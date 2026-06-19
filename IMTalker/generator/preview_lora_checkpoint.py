import argparse
import os
import subprocess

import torch

from generator.generate import InferenceAgent
from generator.train_lora import apply_lora_to_model


def clean_generator_state(ckpt):
    raw = ckpt.get("ema_state_dict") or ckpt.get("state_dict", ckpt.get("model", ckpt))
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    return {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in raw.items()}


def load_lora(agent, opt):
    include_cam = not opt.no_lora_cam_projection
    if getattr(opt, "cam_condition_mode", "normal") in ("remove", "zero_embedding"):
        include_cam = False
    apply_lora_to_model(
        agent.fm,
        rank=opt.lora_rank,
        alpha=opt.lora_alpha,
        dropout=0.0,
        include_pose_lora=not opt.no_lora_pose_projection,
        include_audio_lora=not opt.no_lora_audio_projection,
        include_cam_lora=include_cam,
        only_pose_lora=opt.only_lora_pose_projection,
    )
    ckpt = torch.load(opt.lora_path, map_location="cpu")
    state = clean_generator_state(ckpt)
    missing, unexpected = agent.fm.load_state_dict(state, strict=False)
    lora_keys = sum(1 for k in state if "lora_" in k)
    print(
        f"[preview] loaded lora={opt.lora_path} lora_keys={lora_keys} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    agent.fm.to(opt.rank).eval()


def render(opt, path, use_lora):
    agent = InferenceAgent(opt)
    if use_lora:
        load_lora(agent, opt)
    agent.run_inference(
        path,
        opt.ref_path,
        opt.aud_path,
        pose_path=None,
        gaze_path=None,
        a_cfg_scale=opt.a_cfg_scale,
        nfe=opt.nfe,
        crop=opt.crop,
        seed=opt.seed,
    )


def hstack(paths, labels, out_path):
    inputs = []
    for path in paths:
        inputs.extend(["-i", path])
    filters = []
    for idx, label in enumerate(labels):
        safe = label.replace("'", "")
        filters.append(
            f"[{idx}:v]scale=512:512,setsar=1,"
            f"drawtext=text='{safe}':x=12:y=12:fontsize=22:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=6[v{idx}]"
        )
    filter_complex = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(len(paths))) + f"hstack=inputs={len(paths)}[v]"
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]
    print("[preview ffmpeg]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_path", required=True)
    p.add_argument("--aud_path", required=True)
    p.add_argument("--renderer_path", required=True)
    p.add_argument("--generator_path", required=True)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", default="preview")
    p.add_argument("--wav2vec_model_path", required=True)
    p.add_argument("--rank", default="cuda")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--input_size", type=int, default=512)
    p.add_argument("--dim_w", type=int, default=32)
    p.add_argument("--dim_c", type=int, default=32)
    p.add_argument("--input_nc", type=int, default=3)
    p.add_argument("--audio_marcing", type=int, default=2)
    p.add_argument("--attention_window", type=int, default=5)
    p.add_argument("--audio_dropout_prob", type=float, default=0.1)
    p.add_argument("--ref_dropout_prob", type=float, default=0.1)
    p.add_argument("--emotion_dropout_prob", type=float, default=0.1)
    p.add_argument("--style_dim", type=int, default=512)
    p.add_argument("--dim_a", type=int, default=512)
    p.add_argument("--dim_h", type=int, default=512)
    p.add_argument("--dim_e", type=int, default=7)
    p.add_argument("--dim_motion", type=int, default=32)
    p.add_argument("--fmt_depth", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--no_learned_pe", action="store_true")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--swin_res_threshold", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--low_res_depth", type=int, default=2)
    p.add_argument("--num_prev_frames", type=int, default=10)
    p.add_argument("--wav2vec_sec", type=float, default=2.0)
    p.add_argument("--only_last_features", action="store_true", default=True)
    p.add_argument("--ode_atol", type=float, default=1e-5)
    p.add_argument("--ode_rtol", type=float, default=1e-5)
    p.add_argument("--torchdiffeq_ode_method", default="euler")
    p.add_argument("--a_cfg_scale", type=float, default=1.0)
    p.add_argument("--nfe", type=int, default=5)
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--fix_noise_seed", action="store_true", default=True)
    p.add_argument("--crop", action="store_true")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=float, default=128.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora_pose_projection", action="store_true")
    p.add_argument("--no_lora_audio_projection", action="store_true")
    p.add_argument("--no_lora_cam_projection", action="store_true")
    p.add_argument("--only_lora_pose_projection", action="store_true")
    p.add_argument("--cam_condition_mode", choices=["normal", "remove", "zero_embedding"], default="normal")
    opt = p.parse_args()
    opt.ngpus = 1

    os.makedirs(opt.out_dir, exist_ok=True)
    original = os.path.join(opt.out_dir, f"{opt.name}_original.mp4")
    lora = os.path.join(opt.out_dir, f"{opt.name}_lora.mp4")
    combined = os.path.join(opt.out_dir, f"{opt.name}_original_vs_lora.mp4")

    print("[preview] rendering original", flush=True)
    render(opt, original, use_lora=False)
    torch.cuda.empty_cache()
    print("[preview] rendering lora", flush=True)
    render(opt, lora, use_lora=True)
    torch.cuda.empty_cache()
    hstack([original, lora], ["original", "lora"], combined)
    print(f"[preview done] {combined}", flush=True)


if __name__ == "__main__":
    main()
