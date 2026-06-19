"""Train a studio frontend adapter via audio-only distillation. Multi-GPU (DDP) ready.

This is the training pipeline for `StudioNativeLiveAdapter`'s inner
`HeliumToWav2VecFrontendAdapter` model that the original "phase2_best_*"
checkpoint was trained with -- that pipeline lives outside this repo
(see PERSONAPLEX_IMTALKER_LIVE.md: "niloy629/hdtf_preprocess"). This script
reproduces it for the --frontend_source=mimi_decoder tap added in
liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py.

Teacher target (frozen, never trained):
    Wav2VecModel.extract_projected_frontend(real_audio_16k)  -> [T_w2v, 768]
    This is the Wav2Vec2 CNN + feature_projection output, i.e. the
    representation immediately before Wav2Vec2's positional conv + Transformer
    (see generator/wav2vec2.py).

Student input (frozen Mimi, only the new adapter is trained):
    Mimi.decode_latent(Mimi.encode(real_audio_24k)) -> _to_encoder_framerate
    -> decoder_transformer -> [T_mimi, 512] @ 12.5Hz. Purely acoustic: no
    text/dialogue context is mixed in anywhere in this path (contrast with
    the Helium hidden state, which fuses text_emb into every transformer
    layer -- see personaplex/moshi/moshi/models/lm.py).

Both teacher and student are derived from the SAME raw audio clip, so any
speech corpus works; no motion/video labels are needed. Mirrors the live
serving contract: the adapter is trained to upsample its ~12.5Hz input to
match Wav2Vec2's ~50Hz frontend rate (see helium_w2v_frontend_adapter.py and
StudioNativeLiveAdapter.forward_single).

Multi-GPU: launch with torchrun, one process per GPU. Mimi and Wav2Vec2 are
frozen and simply replicated (in eval mode, no grad) on every rank; only the
small trainable adapter is wrapped in DistributedDataParallel, so gradients
for it are all-reduced across ranks every step. --batch_size is PER GPU, so
effective batch size = batch_size * world_size.

Single GPU (or CPU) still works unchanged: just run with plain `python`.

    # single GPU
    python -m generator.train_frontend_adapter --raw_audio_dir ... --out_dir ...

    # multi-GPU (e.g. 4 GPUs on one node)
    torchrun --standalone --nproc_per_node=4 -m generator.train_frontend_adapter \
        --raw_audio_dir ... --out_dir ...

Checkpointing: every epoch is saved as its own file (epoch_0001.pt, ...,
never overwritten) plus a rolling last.pt (always the most recent epoch, for
easy resuming) and mimi_decoder_best.pt (the single best epoch by val L1,
overwritten only when a new best is found). Pass --resume to continue from a
specific checkpoint.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
from transformers import Wav2Vec2FeatureExtractor

from generator.helium_w2v_frontend_adapter import HeliumToWav2VecFrontendAdapter
from generator.wav2vec2 import Wav2VecModel

MIMI_SR = 24_000
WAV2VEC_SR = 16_000
MIMI_DECODER_HIDDEN_DIM = 512
DEFAULT_WINDOW_SEC = 8.0  # matches the live deque_size=100 steps @ 12.5Hz window


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

class Distributed:
    """Thin wrapper around torch.distributed state, no-op when world_size==1."""

    def __init__(self) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_distributed = self.world_size > 1
        self.is_main = self.rank == 0

        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()

    def reduce_mean(self, value: float) -> float:
        """Average a python float across ranks. No-op if not distributed."""
        if not self.is_distributed:
            return value
        t = torch.tensor([value], device=self.device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item()) / self.world_size

    def shutdown(self) -> None:
        if self.is_distributed:
            dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# Mimi loading + student feature extraction
# ---------------------------------------------------------------------------

def _load_mimi(mimi_hf_repo: str, mimi_weight: str, moshi_root: str, device: torch.device):
    if moshi_root and moshi_root not in sys.path:
        sys.path.insert(0, moshi_root)
    from moshi.models import loaders

    if mimi_weight:
        mimi = loaders.get_mimi(mimi_weight, device=device)
    else:
        from huggingface_hub import hf_hub_download

        weight_path = hf_hub_download(mimi_hf_repo, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(weight_path, device=device)
    mimi.eval()
    for param in mimi.parameters():
        param.requires_grad_(False)
    return mimi


@torch.no_grad()
def mimi_decoder_latent(mimi, wav24k: torch.Tensor) -> torch.Tensor:
    """wav24k: [B, 1, T_samples] (no streaming state) -> [B, T_12.5hz, 512].

    decoder_transformer runs at Mimi's encoder_frame_rate (25Hz), i.e. 2
    frames per 12.5Hz LM step. The live capture in
    liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py
    (_install_mimi_decoder_hidden_capture) keeps only the LAST of each pair
    so the adapter's input cadence matches Helium's 12.5Hz exactly -- we
    downsample the same way here so train/serve cadence matches bit-for-bit.
    Batched over B; Mimi's encode/decode_latent/_to_encoder_framerate/
    decoder_transformer are all batch-agnostic [B, C, T] ops.
    """
    codes = mimi.encode(wav24k)
    emb = mimi.decode_latent(codes)
    emb = mimi._to_encoder_framerate(emb)  # [B, 512, T25] @ 25Hz
    if mimi.decoder_transformer is not None:
        (emb,) = mimi.decoder_transformer(emb)
    bsz, channels, t25 = emb.shape
    t25_even = t25 - (t25 % 2)
    emb = emb[:, :, :t25_even].reshape(bsz, channels, t25_even // 2, 2)[:, :, :, -1]
    return emb.transpose(1, 2).contiguous()  # [B, T_12.5hz, 512]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioFileDataset(Dataset):
    """Recursively collects audio files and returns fixed-length mono 24kHz windows."""

    def __init__(self, audio_dir: str, window_sec: float = DEFAULT_WINDOW_SEC) -> None:
        root = Path(audio_dir)
        exts = ("*.wav", "*.flac", "*.mp3", "*.m4a")
        self.files = sorted({p for ext in exts for p in root.rglob(ext)})
        if not self.files:
            raise RuntimeError(f"No audio files (wav/flac/mp3/m4a) found under {audio_dir}")
        self.window_sec = float(window_sec)
        self.window_samples = int(round(self.window_sec * MIMI_SR))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0, keepdim=True)
        if sr != MIMI_SR:
            wav = torchaudio.functional.resample(wav, sr, MIMI_SR)
        if wav.shape[-1] < self.window_samples:
            wav = F.pad(wav, (0, self.window_samples - wav.shape[-1]))
        else:
            start = random.randint(0, wav.shape[-1] - self.window_samples)
            wav = wav[:, start:start + self.window_samples]
        return wav.squeeze(0).contiguous()


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 31)
    random.seed(seed + worker_id)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    wav24k_batch: torch.Tensor,
    mimi,
    wav2vec,
    feature_extractor,
    adapter: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Fully batched: every clip in the batch has the same fixed window length,
    so Mimi, the Wav2Vec2 feature extractor, and the adapter all run as one
    batched call each (no per-sample Python loop)."""
    wav24k_batch = wav24k_batch.to(device, non_blocking=True)
    bsz = wav24k_batch.shape[0]

    with torch.no_grad():
        mimi_latent = mimi_decoder_latent(mimi, wav24k_batch.unsqueeze(1))  # [B, T_12.5, 512]

        wav16_batch = torchaudio.functional.resample(wav24k_batch, MIMI_SR, WAV2VEC_SR)
        inputs = feature_extractor(
            [w.cpu().numpy() for w in wav16_batch],
            sampling_rate=WAV2VEC_SR,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        attention_mask = inputs.get("attention_mask")
        teacher = wav2vec.extract_projected_frontend(
            input_values=inputs.input_values.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
        )  # [B, T_w2v, 768]

    target_len = int(teacher.shape[1])
    student = adapter(mimi_latent.float(), target_len=target_len)  # [B, T_w2v, 768]
    return F.l1_loss(student.reshape(bsz * target_len, -1), teacher.reshape(bsz * target_len, -1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_audio_dir", required=True, help="Directory of raw speech audio, searched recursively")
    p.add_argument("--wav2vec_model_path", required=True, help="Local path to wav2vec2-base-960h (the frozen teacher)")
    p.add_argument("--moshi_root", default="", help="Directory to prepend to sys.path so `import moshi` resolves to the PersonaPlex fork")
    p.add_argument("--mimi_hf_repo", default="nvidia/personaplex-7b-v1", help="HF repo to download the Mimi checkpoint from if --mimi_weight is empty")
    p.add_argument("--mimi_weight", default="", help="Local Mimi checkpoint path; if empty, downloaded from --mimi_hf_repo")
    p.add_argument("--out_dir", required=True, help="Where to write epoch_*.pt / last.pt / mimi_decoder_best.pt")
    p.add_argument("--num_layers", type=int, default=6, help="Adapter transformer depth; must match what the live script passes via --adapter_num_layers")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=4, help="PER-GPU batch size; effective batch = batch_size * world_size under torchrun")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--window_sec", type=float, default=DEFAULT_WINDOW_SEC)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers PER GPU process")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", default="", help="Optional checkpoint path to resume adapter+optimizer+epoch from")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    ddp = Distributed()

    torch.manual_seed(args.seed + ddp.rank)
    random.seed(args.seed + ddp.rank)

    out_dir = Path(args.out_dir)
    if ddp.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    ddp.barrier()

    if ddp.is_main:
        print(
            f"[train_frontend_adapter] world_size={ddp.world_size} "
            f"effective_batch={args.batch_size * ddp.world_size}",
            flush=True,
        )

    mimi = _load_mimi(args.mimi_hf_repo, args.mimi_weight, args.moshi_root, ddp.device)

    wav2vec = Wav2VecModel.from_pretrained(args.wav2vec_model_path, local_files_only=True).to(ddp.device).eval().float()
    for param in wav2vec.parameters():
        param.requires_grad_(False)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model_path, local_files_only=True)

    adapter = HeliumToWav2VecFrontendAdapter(
        helium_dim=MIMI_DECODER_HIDDEN_DIM, num_layers=args.num_layers, dropout=args.dropout
    ).to(ddp.device).float()

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=ddp.device)
        adapter.load_state_dict(ckpt["adapter"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val_l1", best_val))
        if ddp.is_main:
            print(f"[train_frontend_adapter] resumed from {args.resume} at epoch={start_epoch}", flush=True)

    if ddp.is_distributed:
        adapter = DDP(adapter, device_ids=[ddp.local_rank])

    dataset = AudioFileDataset(args.raw_audio_dir, window_sec=args.window_sec)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    if ddp.is_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=train_sampler,
            drop_last=True, num_workers=args.num_workers, worker_init_fn=_worker_init_fn,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=args.num_workers, worker_init_fn=_worker_init_fn,
            pin_memory=True,
        )
    # Validation only runs on rank 0 (other ranks idle briefly) to keep the
    # best-checkpoint decision simple and avoid cross-rank reduction logic.
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if ddp.is_main:
        print(f"[train_frontend_adapter] train={len(train_set)} val={len(val_set)} clips", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        adapter.train()
        running = 0.0
        n_steps = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(batch, mimi, wav2vec, feature_extractor, adapter, ddp.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            n_steps += 1
        train_loss = ddp.reduce_mean(running / max(1, n_steps))

        val_loss = float("nan")
        if ddp.is_main:
            adapter.eval()
            with torch.no_grad():
                val_running = sum(
                    float(compute_loss(batch, mimi, wav2vec, feature_extractor, adapter, ddp.device).detach())
                    for batch in val_loader
                )
                val_loss = val_running / max(1, len(val_loader))
            print(
                f"[train_frontend_adapter] epoch={epoch} train_l1={train_loss:.5f} val_l1={val_loss:.5f}",
                flush=True,
            )

            ckpt = {
                "adapter": unwrap(adapter).state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "val_l1": val_loss,
                "best_val_l1": min(best_val, val_loss),
                "args": {"num_layers": args.num_layers, "helium_dim": MIMI_DECODER_HIDDEN_DIM, "w2v_dim": 768},
            }
            # Save every epoch, never overwritten -- full training history on disk.
            epoch_path = out_dir / f"epoch_{epoch:04d}.pt"
            torch.save(ckpt, epoch_path)
            # Rolling convenience pointer to the most recent epoch (for --resume).
            torch.save(ckpt, out_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                best_path = out_dir / "mimi_decoder_best.pt"
                torch.save(ckpt, best_path)
                print(f"[train_frontend_adapter] new best val_l1={val_loss:.5f} -> {best_path}", flush=True)

        ddp.barrier()

    if ddp.is_main:
        print(f"[train_frontend_adapter] done. best val_l1={best_val:.5f}", flush=True)
    ddp.shutdown()


if __name__ == "__main__":
    main()
