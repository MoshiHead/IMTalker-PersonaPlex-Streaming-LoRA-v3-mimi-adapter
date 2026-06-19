import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "generator"))

from generator.FM import FMGenerator
from generator.train_lora import apply_lora_to_model
from render_audio_robert_static_smirk_pca_sweep import (
    generator_args,
    interp_sequence,
    load_audio,
    read_source_image,
)
from static_head_map_anchor_compare import MASK_CONFIGS, method_metrics
from static_head_map_composite import (
    from_tensor,
    load_renderer,
    make_mouth_mask,
    shape_probe,
    to_tensor,
    write_h264_video,
)
from static_head_map_more_ideas import composite_maps


def strip_model_prefix(state):
    if any(k.startswith("model.") for k in state):
        return {k.replace("model.", "", 1): v for k, v in state.items()}
    return state


def load_base_generator(checkpoint_path, opt, device):
    model = FMGenerator(opt).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if "model" in state:
        state = state["model"]
    state = strip_model_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[GEN] base={checkpoint_path} tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}")
    return model


def load_noaudio_lora_generator(base_ckpt, lora_ckpt, opt, device, rank=64, alpha=128, dropout=0.05):
    model = FMGenerator(opt).to(device).eval()
    base = torch.load(base_ckpt, map_location="cpu")
    base_state = base.get("ema_state_dict") or base.get("state_dict", base)
    if "model" in base_state:
        base_state = base_state["model"]
    base_state = strip_model_prefix(base_state)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"[GEN] lora base={base_ckpt} tensors={len(base_state)} missing={len(missing)} unexpected={len(unexpected)}")
    apply_lora_to_model(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        include_pose_lora=True,
        include_audio_lora=False,
        only_pose_lora=False,
    )
    lora = torch.load(lora_ckpt, map_location="cpu")
    lora_state = lora.get("ema_state_dict") or lora.get("state_dict", lora)
    lora_state = strip_model_prefix(lora_state)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    lora_keys = sum(1 for k in lora_state if "lora_" in k)
    print(f"[GEN] lora={lora_ckpt} lora_keys={lora_keys} missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


def load_conditions(pose_path, gaze_path, target_len, device):
    pose = cam = gaze = None
    if pose_path:
        d = torch.load(pose_path, map_location="cpu")
        pose = interp_sequence(d["pose_params"], target_len).to(device)
        cam = interp_sequence(d["cam"], target_len).to(device)
        print(f"[COND] pose={tuple(pose.shape)} cam={tuple(cam.shape)} from {pose_path}")
    if gaze_path:
        g = torch.tensor(np.load(gaze_path), dtype=torch.float32)
        gaze = interp_sequence(g, target_len).to(device)
        print(f"[COND] gaze={tuple(gaze.shape)} from {gaze_path}")
    return pose, cam, gaze


def save_three_col(a_frames, b_frames, c_frames, path, fps):
    out = []
    for a, b, c in zip(a_frames, b_frames, c_frames):
        out.append(
            np.concatenate(
                [
                    cv2.resize(a, (512, 512)),
                    cv2.resize(b, (512, 512)),
                    cv2.resize(c, (512, 512)),
                ],
                axis=1,
            )
        )
    write_h264_video(out, path, fps)


def mux_audio(video_path, audio_path):
    import os

    path = Path(video_path)
    muxed = path.with_name(path.stem + "_audio.mp4")
    cmd = (
        f"ffmpeg -y -loglevel error -i {str(path)!r} -i {str(audio_path)!r} "
        f"-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest -movflags +faststart {str(muxed)!r}"
    )
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"ffmpeg mux failed for {path}")
    return str(muxed)


@torch.no_grad()
def render_compare(renderer, gen_base, gen_lora, source_img, audio_tensor, pose, cam, gaze, masks, device, cfg, nfe, seed):
    source = to_tensor(source_img, device)
    f_r, i_r = renderer.app_encode(source)
    t_ref = renderer.mot_encode(source)
    ma_r = renderer.mot_decode(renderer.id_adapt(t_ref, i_r))

    data = {
        "a": audio_tensor.to(device),
        "ref_x": t_ref,
        "gaze": gaze,
        "pose": pose,
        "cam": cam,
    }
    sample_base = gen_base.sample(data, a_cfg_scale=cfg, nfe=nfe, seed=seed)
    sample_lora = gen_lora.sample(data, a_cfg_scale=cfg, nfe=nfe, seed=seed)
    print(f"[SAMPLE] cfg={cfg} base={tuple(sample_base.shape)} lora={tuple(sample_lora.shape)}")

    original = []
    lora_normal = []
    lora_mask = []
    n = min(sample_base.shape[1], sample_lora.shape[1])
    for idx in range(n):
        t_base = sample_base[:, idx, :]
        ma_base = renderer.mot_decode(renderer.id_adapt(t_base, i_r))
        original.append(from_tensor(renderer.decode(ma_base, ma_r, f_r)))

        t_lora = sample_lora[:, idx, :]
        ma_lora = renderer.mot_decode(renderer.id_adapt(t_lora, i_r))
        lora_normal.append(from_tensor(renderer.decode(ma_lora, ma_r, f_r)))
        ma_mask = composite_maps(ma_lora, ma_r, masks)
        lora_mask.append(from_tensor(renderer.decode(ma_mask, ma_r, f_r)))

        if (idx + 1) % 25 == 0:
            print(f"[FRAME] {idx + 1}/{n}", flush=True)
    return original, lora_normal, lora_mask


def add_rel_metrics(m, base):
    m["head_reduction_vs_original"] = 1 - m["head_motion"] / max(base["head_motion"], 1e-3)
    m["lip_preservation_vs_original"] = m["lip_motion"] / max(base["lip_motion"], 1e-3)
    m["drift_reduction_vs_original"] = 1 - m["mouth_drift"] / max(base["mouth_drift"], 1e-3)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default="/workspace/IMTalker/checkpoints/renderer.ckpt")
    ap.add_argument("--base_generator", default="/workspace/IMTalker/checkpoints/generator.ckpt")
    ap.add_argument("--lora_generator", default="/workspace/exps/imtalker_lora/runs_new_modes_5k/lora_r64_all_noaudio_5k/checkpoints/last.ckpt")
    ap.add_argument("--audio", default="/workspace/IMTalker/assets/audio_3.wav")
    ap.add_argument("--source_image", required=True)
    ap.add_argument("--source_name", required=True)
    ap.add_argument("--pose", default="/workspace/static_imt/lora_preview_conditions/smirk/source_5.pt")
    ap.add_argument("--gaze", default="/workspace/static_imt/lora_preview_conditions/gaze/source_5.npy")
    ap.add_argument("--output_dir", default="/workspace/exps/imtalker_lora/noaudio_lora_mask_cfg123")
    ap.add_argument("--cfgs", default="1,2,3")
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--wav2vec_sec", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gopt = generator_args()
    gopt.wav2vec_sec = args.wav2vec_sec
    gopt.num_frames_for_clip = int(gopt.wav2vec_sec * gopt.fps)
    gopt.nfe = args.nfe

    renderer = load_renderer(args.renderer, device)
    gen_base = load_base_generator(args.base_generator, gopt, device)
    gen_lora = load_noaudio_lora_generator(args.base_generator, args.lora_generator, gopt, device)

    resolutions = shape_probe(renderer, device)
    masks = [make_mouth_mask(r, **MASK_CONFIGS["tight"]) for r in resolutions]
    print(f"[MASK] tight={MASK_CONFIGS['tight']} resolutions={resolutions}")

    source_img = read_source_image(args.source_image)
    audio_tensor, audio_sec = load_audio(args.audio, gopt.sampling_rate, gopt.wav2vec_model_path)
    target_len = int(np.ceil(audio_tensor.shape[-1] * gopt.fps / gopt.sampling_rate))
    pose, cam, gaze = load_conditions(args.pose, args.gaze, target_len, device)

    summary = {
        "columns": ["original", "rank64_all_no_audio_lora", "rank64_all_no_audio_lora_mask"],
        "source_image": args.source_image,
        "audio": args.audio,
        "audio_sec": audio_sec,
        "mask": MASK_CONFIGS["tight"],
        "cfgs": {},
    }
    for cfg in [float(x.strip()) for x in args.cfgs.split(",") if x.strip()]:
        original, lora_normal, lora_mask = render_compare(
            renderer, gen_base, gen_lora, source_img, audio_tensor, pose, cam, gaze, masks, device, cfg, args.nfe, args.seed
        )
        cfg_label = str(int(cfg)) if float(cfg).is_integer() else str(cfg).replace(".", "p")
        video = out_dir / f"{args.source_name}_cfg{cfg_label}_original_noaudio_lora_noaudio_lora_mask.mp4"
        save_three_col(original, lora_normal, lora_mask, video, gopt.fps)
        muxed = mux_audio(video, args.audio)

        m_original = method_metrics(original)
        m_lora = add_rel_metrics(method_metrics(lora_normal), m_original)
        m_mask = add_rel_metrics(method_metrics(lora_mask), m_original)
        summary["cfgs"][str(cfg)] = {
            "video": str(video),
            "video_with_audio": muxed,
            "original": m_original,
            "rank64_all_no_audio_lora": m_lora,
            "rank64_all_no_audio_lora_mask": m_mask,
        }
        print(
            f"[METRIC] {args.source_name} cfg={cfg} "
            f"orig drift={m_original['mouth_drift']:.3f} lip={m_original['lip_motion']:.3f} head={m_original['head_motion']:.3f} | "
            f"lora drift={m_lora['mouth_drift']:.3f} lip={m_lora['lip_motion']:.3f} head={m_lora['head_motion']:.3f} | "
            f"mask drift={m_mask['mouth_drift']:.3f} lip={m_mask['lip_motion']:.3f} head={m_mask['head_motion']:.3f}"
        )
        print(f"[DONE] {muxed}")

    metrics_path = out_dir / f"{args.source_name}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"[DONE] metrics={metrics_path}")


if __name__ == "__main__":
    main()
