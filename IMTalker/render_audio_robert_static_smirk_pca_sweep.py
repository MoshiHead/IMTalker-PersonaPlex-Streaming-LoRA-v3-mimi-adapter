import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

from generator.FM import FMGenerator
from static_head_map_anchor_compare import MASK_CONFIGS, method_metrics
from static_head_map_composite import (
    from_tensor,
    load_renderer,
    make_mouth_mask,
    shape_probe,
    to_tensor,
    write_h264_video,
)
from static_head_map_more_ideas import composite_maps, load_pca


def generator_args():
    return SimpleNamespace(
        rank=0,
        ngpus=1,
        seed=25,
        fix_noise_seed=True,
        input_size=512,
        input_nc=3,
        fps=25.0,
        sampling_rate=16000,
        audio_marcing=2,
        wav2vec_sec=2.0,
        wav2vec_model_path="/workspace/IMTalker/checkpoints/wav2vec2-base-960h",
        attention_window=5,
        only_last_features=True,
        average_emotion=False,
        audio_dropout_prob=0.1,
        ref_dropout_prob=0.1,
        emotion_dropout_prob=0.1,
        style_dim=512,
        dim_a=512,
        dim_h=512,
        dim_e=7,
        dim_motion=32,
        dim_c=32,
        dim_w=32,
        fmt_depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        no_learned_pe=False,
        num_prev_frames=10,
        max_grad_norm=1,
        ode_atol=1e-5,
        ode_rtol=1e-5,
        nfe=10,
        torchdiffeq_ode_method="euler",
        a_cfg_scale=3.0,
        swin_res_threshold=128,
        window_size=8,
    )


def read_source_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img


def load_audio(path, sampling_rate, wav2vec_model_path):
    speech_array, sr = librosa.load(str(path), sr=sampling_rate)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=True)
    audio = processor(speech_array, sampling_rate=sr, return_tensors="pt").input_values[0]
    return audio.unsqueeze(0), len(speech_array) / float(sr)


def load_generator(checkpoint_path, opt, device):
    model = FMGenerator(opt).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if "model" in state:
        state = state["model"]
    stripped = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
    if not stripped:
        stripped = state
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    print(f"[GEN] checkpoint={checkpoint_path} tensors={len(stripped)} missing={len(missing)} unexpected={len(unexpected)}")
    return model


def choose_smirk(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    files = sorted(p.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No SMIRK .pt files under {p}")
    return files[0]


def interp_sequence(x, target_len):
    x = x.float()
    if x.ndim != 2:
        raise ValueError(f"Expected (T,C), got {tuple(x.shape)}")
    if x.shape[0] == target_len:
        return x
    y = F.interpolate(
        x.T.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0).T
    return y.contiguous()


def load_static_smirk(smirk_path, target_len):
    d = torch.load(smirk_path, map_location="cpu")
    pose = interp_sequence(d["pose_params"], target_len)
    cam = interp_sequence(d["cam"], target_len)
    print(f"[SMIRK] {smirk_path} raw_pose={tuple(d['pose_params'].shape)} interp_pose={tuple(pose.shape)} interp_cam={tuple(cam.shape)}")
    return pose, cam


def mux_audio(video_path, audio_path):
    path = Path(video_path)
    muxed = path.with_name(path.stem + "_with_audio.mp4")
    import os

    cmd = (
        f"ffmpeg -y -loglevel error -i {str(path)!r} -i {str(audio_path)!r} "
        f"-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest -movflags +faststart {str(muxed)!r}"
    )
    rc = os.system(cmd)
    return str(muxed if rc == 0 else path)


def save_overview(source_img, normal, methods, order, path, fps):
    src_tile = cv2.resize(source_img, (512, 512))
    out = []
    for i in range(len(normal)):
        tiles = [src_tile, normal[i]]
        for name in order:
            tiles.append(methods[name][i])
        out.append(np.concatenate(tiles, axis=1))
    write_h264_video(out, path, fps)


@torch.no_grad()
def render(renderer, generator, source_img, audio_tensor, pose, cam, masks, pca_mats, device, a_cfg_scale, nfe, seed):
    source = to_tensor(source_img, device)
    f_r, i_r = renderer.app_encode(source)
    t_ref = renderer.mot_encode(source)
    ma_r = renderer.mot_decode(renderer.id_adapt(t_ref, i_r))

    data = {
        "a": audio_tensor.to(device),
        "ref_x": t_ref,
        "gaze": None,
        "pose": pose.to(device),
        "cam": cam.to(device),
    }
    sample = generator.sample(data, a_cfg_scale=a_cfg_scale, nfe=nfe, seed=seed)
    print(f"[AUDIO_SAMPLE] latents={tuple(sample.shape)}")

    normal = []
    methods = {f"pca_k{k}_tight": [] for k in pca_mats}
    for idx in range(sample.shape[1]):
        t_cur = sample[:, idx, :]
        ma_c = renderer.mot_decode(renderer.id_adapt(t_cur, i_r))
        normal.append(from_tensor(renderer.decode(ma_c, ma_r, f_r)))

        delta = t_cur - t_ref
        for k, p in pca_mats.items():
            t_pca = t_ref + (delta @ p) @ p.T
            ma_pca = renderer.mot_decode(renderer.id_adapt(t_pca, i_r))
            maps_pca = composite_maps(ma_pca, ma_r, masks)
            methods[f"pca_k{k}_tight"].append(from_tensor(renderer.decode(maps_pca, ma_r, f_r)))

        if (idx + 1) % 25 == 0:
            print(f"[FRAME] {idx + 1}/{sample.shape[1]}", flush=True)
    return normal, methods, sample.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default="/workspace/IMTalker/checkpoints/renderer.ckpt")
    ap.add_argument("--generator", default="/workspace/IMTalker/checkpoints/generator.ckpt")
    ap.add_argument("--audio", default="/workspace/IMTalker/assets/audio_3.wav")
    ap.add_argument("--source_image", default="/workspace/static_imt/source_images/robert_better1 resize.png")
    ap.add_argument("--smirk", default="/workspace/static_imt/ditto_source_5_static/smirk")
    ap.add_argument("--pca_path", default="/workspace/static_imt/ditto_source_5_static/pca/pca_components_32x32.npy")
    ap.add_argument("--ks", default="28,29,30,31")
    ap.add_argument("--output_dir", default="/workspace/exps/audio_static_smirk_static_source5_full32_pca_tight_robert")
    ap.add_argument("--label", default="AUDIO_STATIC_SMIRK_STATIC_SOURCE5_FULL32_PCA_TIGHT_28_31_ROBERT")
    ap.add_argument("--a_cfg_scale", type=float, default=3.0)
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gopt = generator_args()

    renderer = load_renderer(args.renderer, device)
    generator = load_generator(args.generator, gopt, device)
    resolutions = shape_probe(renderer, device)
    masks = [make_mouth_mask(r, **MASK_CONFIGS["tight"]) for r in resolutions]

    audio_tensor, audio_sec = load_audio(args.audio, gopt.sampling_rate, gopt.wav2vec_model_path)
    target_len = int(np.ceil(audio_tensor.shape[-1] * gopt.fps / gopt.sampling_rate))
    smirk_path = choose_smirk(args.smirk)
    pose, cam = load_static_smirk(smirk_path, target_len)

    pca_mats = {}
    for k in [int(x.strip()) for x in args.ks.split(",") if x.strip()]:
        pca_mats[k] = load_pca(args.pca_path, device, k)
    pca_mats = {k: v for k, v in pca_mats.items() if v is not None}
    if not pca_mats:
        raise RuntimeError(f"No PCA matrices loaded from {args.pca_path}")

    source_img = read_source_image(args.source_image)
    normal, methods, num_frames = render(
        renderer, generator, source_img, audio_tensor, pose, cam, masks, pca_mats,
        device, args.a_cfg_scale, args.nfe, args.seed
    )

    normal_metrics = method_metrics(normal)
    result = {
        "audio": args.audio,
        "audio_sec": audio_sec,
        "source_image": args.source_image,
        "smirk_path": str(smirk_path),
        "pca_path": args.pca_path,
        "frames": num_frames,
        "fps": gopt.fps,
        "columns": ["source", "audio_normal_static_smirk"] + list(methods.keys()),
        "mask": MASK_CONFIGS["tight"],
        "normal": normal_metrics,
        "methods": {},
    }
    for name, frames in methods.items():
        m = method_metrics(frames)
        m["head_reduction_vs_normal"] = 1 - m["head_motion"] / max(normal_metrics["head_motion"], 1e-3)
        m["lip_preservation_vs_normal"] = m["lip_motion"] / max(normal_metrics["lip_motion"], 1e-3)
        m["drift_reduction_vs_normal"] = 1 - m["mouth_drift"] / max(normal_metrics["mouth_drift"], 1e-3)
        result["methods"][name] = m
        print(
            f"[METRIC] {name:16s} drift={m['mouth_drift']:.3f} lip={m['lip_motion']:.3f} "
            f"head={m['head_motion']:.3f} head_red={m['head_reduction_vs_normal']*100:.1f}% "
            f"lip_pres={m['lip_preservation_vs_normal']*100:.1f}%"
        )

    order = list(methods.keys())
    video_path = out_dir / f"robert_better1_resize_{args.label}_overview.mp4"
    save_overview(source_img, normal, methods, order, video_path, gopt.fps)
    muxed = mux_audio(video_path, args.audio)
    result["video"] = str(video_path)
    result["video_with_audio"] = muxed

    metrics_path = out_dir / f"{args.label.lower()}_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))
    print(f"[DONE] video={video_path}")
    print(f"[DONE] muxed={muxed}")
    print(f"[DONE] metrics={metrics_path}")


if __name__ == "__main__":
    main()
