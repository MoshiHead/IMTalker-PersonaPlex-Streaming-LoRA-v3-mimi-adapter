import argparse
import os
import subprocess

import torch

from generator.generate import InferenceAgent
from generator.train_lora import apply_lora_to_model


def load_lora(agent, lora_path, fallback_mode, fallback_rank, fallback_alpha):
    state = torch.load(lora_path, map_location="cpu")
    mode = state.get("lora_mode", fallback_mode)
    rank = int(state.get("lora_rank", fallback_rank))
    alpha = float(state.get("lora_alpha", fallback_alpha))
    lora_state = state.get("lora_state_dict", state)

    replaced = apply_lora_to_model(
        agent.fm,
        rank=rank,
        alpha=alpha,
        dropout=0.0,
        include_pose_lora=mode in ("pose", "pose_fmt"),
        include_fmt_lora=mode == "pose_fmt",
        lora_mode=mode,
    )
    missing, unexpected = agent.fm.load_state_dict(lora_state, strict=False)
    print(
        f"[LoRA] path={lora_path} mode={mode} rank={rank} alpha={alpha} "
        f"replaced={len(replaced)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    agent.fm.to(agent.opt.rank).eval()


def render(opt, out_path, lora_path=None, fallback_mode=None):
    agent = InferenceAgent(opt)
    if lora_path:
        load_lora(
            agent,
            lora_path=lora_path,
            fallback_mode=fallback_mode,
            fallback_rank=opt.lora_rank,
            fallback_alpha=opt.lora_alpha,
        )
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
    for path in paths:
        inputs.extend(["-i", path])

    filters = []
    for idx, label in enumerate(labels):
        filters.append(
            f"[{idx}:v]scale=512:512,setsar=1,"
            f"drawtext=text='{label}':x=12:y=12:fontsize=23:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=6[v{idx}]"
        )
    filter_complex = (
        ";".join(filters)
        + ";"
        + "".join(f"[v{i}]" for i in range(len(paths)))
        + f"hstack=inputs={len(paths)}[v]"
    )
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
    print("[ffmpeg]", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", required=True)
    parser.add_argument("--aud_path", required=True)
    parser.add_argument("--renderer_path", required=True)
    parser.add_argument("--generator_path", required=True)
    parser.add_argument("--pose_cam_adaln_lora", required=True)
    parser.add_argument("--adaln_only_lora", required=True)
    parser.add_argument("--decoder_only_lora", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--wav2vec_model_path", required=True)
    parser.add_argument("--rank", default="cuda")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--input_nc", type=int, default=3)
    parser.add_argument("--audio_marcing", type=int, default=2)
    parser.add_argument("--attention_window", type=int, default=5)
    parser.add_argument("--wav2vec_sec", type=float, default=2.0)
    parser.add_argument("--only_last_features", action="store_true", default=True)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--dim_a", type=int, default=512)
    parser.add_argument("--dim_h", type=int, default=512)
    parser.add_argument("--dim_e", type=int, default=7)
    parser.add_argument("--dim_motion", type=int, default=32)
    parser.add_argument("--dim_w", type=int, default=32)
    parser.add_argument("--dim_c", type=int, default=32)
    parser.add_argument("--fmt_depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--no_learned_pe", action="store_true")
    parser.add_argument("--num_prev_frames", type=int, default=10)
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--torchdiffeq_ode_method", default="euler")
    parser.add_argument("--a_cfg_scale", type=float, default=1.0)
    parser.add_argument("--nfe", type=int, default=10)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--fix_noise_seed", action="store_true", default=True)
    parser.add_argument("--swin_res_threshold", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--low_res_depth", type=int, default=2)
    parser.add_argument("--audio_dropout_prob", type=float, default=0.1)
    parser.add_argument("--ref_dropout_prob", type=float, default=0.1)
    parser.add_argument("--emotion_dropout_prob", type=float, default=0.1)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    opt = parser.parse_args()
    opt.ngpus = 1

    os.makedirs(opt.out_dir, exist_ok=True)
    paths = [
        os.path.join(opt.out_dir, "original.mp4"),
        os.path.join(opt.out_dir, "pose_cam_adaln.mp4"),
        os.path.join(opt.out_dir, "adaln_only.mp4"),
        os.path.join(opt.out_dir, "decoder_only.mp4"),
    ]

    print("[RUN] original")
    render(opt, paths[0])
    torch.cuda.empty_cache()

    print("[RUN] pose_cam_adaln")
    render(opt, paths[1], opt.pose_cam_adaln_lora, "pose_cam_adaln")
    torch.cuda.empty_cache()

    print("[RUN] adaln_only")
    render(opt, paths[2], opt.adaln_only_lora, "adaln_only")
    torch.cuda.empty_cache()

    print("[RUN] decoder_only")
    render(opt, paths[3], opt.decoder_only_lora, "decoder_only")

    combined = os.path.join(opt.out_dir, "original_posecam_adaln_decoder_side_by_side.mp4")
    hstack(paths, ["original", "pose_cam_adaln", "adaln_only", "decoder_only"], combined)
    print(f"[DONE] {combined}")


if __name__ == "__main__":
    main()
