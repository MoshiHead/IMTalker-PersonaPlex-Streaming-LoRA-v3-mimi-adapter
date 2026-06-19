"""Train a studio frontend adapter via audio-only distillation.

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
serving contract: the adapter is trained to upsample its ~12.5Hz input by 2x
to match Wav2Vec2's ~50Hz frontend rate (see helium_w2v_frontend_adapter.py
and StudioNativeLiveAdapter.forward_single).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import Wav2Vec2FeatureExtractor

from generator.helium_w2v_frontend_adapter import HeliumToWav2VecFrontendAdapter
from generator.wav2vec2 import Wav2VecModel

MIMI_SR = 24_000
WAV2VEC_SR = 16_000
MIMI_DECODER_HIDDEN_DIM = 512
DEFAULT_WINDOW_SEC = 8.0  # matches the live deque_size=100 steps @ 12.5Hz window


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
    """wav24k: [1, 1, T_samples] (no streaming state) -> [1, T_12.5hz, 512].

    decoder_transformer runs at Mimi's encoder_frame_rate (25Hz), i.e. 2
    frames per 12.5Hz LM step. The live capture in
    liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py
    (_install_mimi_decoder_hidden_capture) keeps only the LAST of each pair
    so the adapter's input cadence matches Helium's 12.5Hz exactly -- we
    downsample the same way here so train/serve cadence matches bit-for-bit.
    """
    codes = mimi.encode(wav24k)
    emb = mimi.decode_latent(codes)
    emb = mimi._to_encoder_framerate(emb)  # [1, 512, T25] @ 25Hz
    if mimi.decoder_transformer is not None:
        (emb,) = mimi.decoder_transformer(emb)
    t25 = emb.shape[-1]
    t25_even = t25 - (t25 % 2)
    emb = emb[:, :, :t25_even].reshape(1, emb.shape[1], t25_even // 2, 2)[:, :, :, -1]
    return emb.transpose(1, 2).contiguous()  # [1, T_12.5hz, 512]


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
        print(f"[train_frontend_adapter] found {len(self.files)} audio files under {audio_dir}")

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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_audio_dir", required=True, help="Directory of raw speech audio, searched recursively")
    p.add_argument("--wav2vec_model_path", required=True, help="Local path to wav2vec2-base-960h (the frozen teacher)")
    p.add_argument("--moshi_root", default="", help="Directory to prepend to sys.path so `import moshi` resolves to the PersonaPlex fork")
    p.add_argument("--mimi_hf_repo", default="nvidia/personaplex-7b-v1", help="HF repo to download the Mimi checkpoint from if --mimi_weight is empty")
    p.add_argument("--mimi_weight", default="", help="Local Mimi checkpoint path; if empty, downloaded from --mimi_hf_repo")
    p.add_argument("--out_dir", required=True, help="Where to write last.pt / mimi_decoder_best.pt")
    p.add_argument("--num_layers", type=int, default=6, help="Adapter transformer depth; must match what the live script passes via --adapter_num_layers")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--window_sec", type=float, default=DEFAULT_WINDOW_SEC)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mimi = _load_mimi(args.mimi_hf_repo, args.mimi_weight, args.moshi_root, device)

    wav2vec = Wav2VecModel.from_pretrained(args.wav2vec_model_path, local_files_only=True).to(device).eval().float()
    for param in wav2vec.parameters():
        param.requires_grad_(False)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model_path, local_files_only=True)

    adapter = HeliumToWav2VecFrontendAdapter(
        helium_dim=MIMI_DECODER_HIDDEN_DIM, num_layers=args.num_layers, dropout=args.dropout
    ).to(device).float()

    dataset = AudioFileDataset(args.raw_audio_dir, window_sec=args.window_sec)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[train_frontend_adapter] train={len(train_set)} val={len(val_set)} clips")

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    def compute_loss(wav24k_batch: torch.Tensor) -> torch.Tensor:
        wav24k_batch = wav24k_batch.to(device)
        student_feats = []
        teacher_feats = []
        for i in range(wav24k_batch.shape[0]):
            wav24 = wav24k_batch[i:i + 1].unsqueeze(1)  # [1, 1, T]
            with torch.no_grad():
                mimi_latent = mimi_decoder_latent(mimi, wav24)  # [1, T_12.5, 512]

                wav16 = torchaudio.functional.resample(wav24k_batch[i:i + 1], MIMI_SR, WAV2VEC_SR)
                inputs = feature_extractor(
                    wav16.squeeze(0).cpu().numpy(),
                    sampling_rate=WAV2VEC_SR,
                    return_tensors="pt",
                    padding=True,
                )
                teacher = wav2vec.extract_projected_frontend(
                    input_values=inputs.input_values.to(device),
                )  # [1, T_w2v, 768]

            target_len = int(teacher.shape[1])
            student = adapter(mimi_latent.float(), target_len=target_len)  # [1, T_w2v, 768]
            student_feats.append(student[0])
            teacher_feats.append(teacher[0])

        student_cat = torch.cat(student_feats, dim=0)
        teacher_cat = torch.cat(teacher_feats, dim=0)
        return F.l1_loss(student_cat, teacher_cat)

    best_val = float("inf")
    for epoch in range(args.epochs):
        adapter.train()
        running = 0.0
        for batch in train_loader:
            opt.zero_grad()
            loss = compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
        train_loss = running / max(1, len(train_loader))

        adapter.eval()
        with torch.no_grad():
            val_running = sum(float(compute_loss(batch).detach()) for batch in val_loader)
            val_loss = val_running / max(1, len(val_loader))

        print(
            f"[train_frontend_adapter] epoch={epoch} train_l1={train_loss:.5f} val_l1={val_loss:.5f}",
            flush=True,
        )

        ckpt = {
            "adapter": adapter.state_dict(),
            "args": {"num_layers": args.num_layers, "helium_dim": MIMI_DECODER_HIDDEN_DIM, "w2v_dim": 768},
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / "mimi_decoder_best.pt"
            torch.save(ckpt, best_path)
            print(f"[train_frontend_adapter] new best val_l1={val_loss:.5f} -> {best_path}", flush=True)

    print(f"[train_frontend_adapter] done. best val_l1={best_val:.5f}")


if __name__ == "__main__":
    main()
