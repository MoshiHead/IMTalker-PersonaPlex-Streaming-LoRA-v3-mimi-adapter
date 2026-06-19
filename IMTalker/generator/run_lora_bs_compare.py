import argparse
import os
import subprocess

import torch

from generator.generate import InferenceAgent
from generator.train_lora import apply_lora_to_model


def load_pose_fmt_lora(agent, lora_path, rank, alpha):
    apply_lora_to_model(
        agent.fm,
        rank=rank,
        alpha=alpha,
        dropout=0.0,
        include_pose_lora=True,
        include_fmt_lora=True,
    )
    state = torch.load(lora_path, map_location="cpu")
    if "lora_state_dict" in state:
        state = state["lora_state_dict"]
    missing, unexpected = agent.fm.load_state_dict(state, strict=False)
    print(f"[LoRA] loaded={lora_path} missing={len(missing)} unexpected={len(unexpected)}")
    agent.fm.to(agent.opt.rank).eval()


def render(opt, out_path, lora_path=None):
    agent = InferenceAgent(opt)
    if lora_path:
        load_pose_fmt_lora(agent, lora_path, opt.lora_rank, opt.lora_alpha)
    agent.run_inference(
        out_path,
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
    for p in paths:
        inputs.extend(["-i", p])
    filters = []
    for i, label in enumerate(labels):
        filters.append(
            f"[{i}:v]scale=512:512,setsar=1,"
            f"drawtext=text='{label}':x=16:y=16:fontsize=28:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=8[v{i}]"
        )
    filter_complex = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(len(paths))) + f"hstack=inputs={len(paths)}[v]"
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-shortest", out_path,
    ]
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_path", required=True)
    p.add_argument("--aud_path", required=True)
    p.add_argument("--renderer_path", required=True)
    p.add_argument("--generator_path", required=True)
    p.add_argument("--bs256_lora_path", required=True)
    p.add_argument("--bs512_lora_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wav2vec_model_path", required=True)
    p.add_argument("--rank", default="cuda")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--input_size", type=int, default=512)
    p.add_argument("--input_nc", type=int, default=3)
    p.add_argument("--audio_marcing", type=int, default=2)
    p.add_argument("--attention_window", type=int, default=5)
    p.add_argument("--wav2vec_sec", type=float, default=2.0)
    p.add_argument("--only_last_features", action="store_true", default=True)
    p.add_argument("--style_dim", type=int, default=512)
    p.add_argument("--dim_a", type=int, default=512)
    p.add_argument("--dim_h", type=int, default=512)
    p.add_argument("--dim_e", type=int, default=7)
    p.add_argument("--dim_motion", type=int, default=32)
    p.add_argument("--dim_w", type=int, default=32)
    p.add_argument("--dim_c", type=int, default=32)
    p.add_argument("--fmt_depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--no_learned_pe", action="store_true")
    p.add_argument("--num_prev_frames", type=int, default=10)
    p.add_argument("--ode_atol", type=float, default=1e-5)
    p.add_argument("--ode_rtol", type=float, default=1e-5)
    p.add_argument("--torchdiffeq_ode_method", default="euler")
    p.add_argument("--a_cfg_scale", type=float, default=1.0)
    p.add_argument("--nfe", type=int, default=10)
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--fix_noise_seed", action="store_true", default=True)
    p.add_argument("--swin_res_threshold", type=int, default=128)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--low_res_depth", type=int, default=2)
    p.add_argument("--audio_dropout_prob", type=float, default=0.1)
    p.add_argument("--ref_dropout_prob", type=float, default=0.1)
    p.add_argument("--emotion_dropout_prob", type=float, default=0.1)
    p.add_argument("--crop", action="store_true")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    opt = p.parse_args()
    opt.ngpus = 1

    os.makedirs(opt.out_dir, exist_ok=True)
    paths = [
        os.path.join(opt.out_dir, "original.mp4"),
        os.path.join(opt.out_dir, "bs256.mp4"),
        os.path.join(opt.out_dir, "bs512.mp4"),
    ]
    print("[RUN] original")
    render(opt, paths[0])
    torch.cuda.empty_cache()
    print("[RUN] bs256")
    render(opt, paths[1], opt.bs256_lora_path)
    torch.cuda.empty_cache()
    print("[RUN] bs512")
    render(opt, paths[2], opt.bs512_lora_path)
    combined = os.path.join(opt.out_dir, "original_bs256_bs512_side_by_side.mp4")
    hstack(paths, ["original", "bs256", "bs512"], combined)
    print(f"[DONE] {combined}")


if __name__ == "__main__":
    main()
