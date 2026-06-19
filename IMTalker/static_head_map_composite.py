import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from renderer.models import IMTRenderer


def renderer_args():
    return SimpleNamespace(
        input_size=512,
        swin_res_threshold=128,
        num_heads=8,
        window_size=8,
    )


def load_renderer(checkpoint_path, device):
    model = IMTRenderer(renderer_args()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean = {k.replace("gen.", "", 1): v for k, v in state_dict.items() if k.startswith("gen.")}
    if not clean:
        clean = state_dict
    missing, unexpected = model.load_state_dict(clean, strict=False)
    model.eval()
    print(f"[LOAD] checkpoint={checkpoint_path}")
    print(f"[LOAD] tensors={len(clean)} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[LOAD] missing sample={missing[:8]}")
    if unexpected:
        print(f"[LOAD] unexpected sample={unexpected[:8]}")
    return model


def to_tensor(frame_rgb, device):
    image = Image.fromarray(frame_rgb)
    tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])(image).unsqueeze(0)
    return tensor.to(device)


def from_tensor(tensor):
    arr = tensor.detach().squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).round().astype(np.uint8)


def read_video(path, max_frames=80):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return frames, float(fps)


def write_h264_video(frames_rgb, path, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".mp4v.tmp.mp4")
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    cmd = (
        f"ffmpeg -y -loglevel error -i {str(tmp)!r} "
        f"-c:v libx264 -pix_fmt yuv420p -movflags +faststart {str(path)!r}"
    )
    rc = os.system(cmd)
    if rc != 0:
        tmp.rename(path)
    else:
        tmp.unlink(missing_ok=True)
    print(f"[SAVE] {path}")


def gaussian_blur_mask(mask, kernel_size):
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(kernel_size / 6.0, 1e-3)
    coords = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
    pad = kernel_size // 2
    mask = F.pad(mask, (pad, pad, 0, 0), mode="replicate")
    mask = F.conv2d(mask, kernel_x)
    mask = F.pad(mask, (0, 0, pad, pad), mode="replicate")
    mask = F.conv2d(mask, kernel_y)
    return mask.clamp(0, 1)


def make_mouth_mask(resolution, v_start, v_end, h_start, h_end, feather):
    mask = torch.zeros(1, 1, resolution, resolution)
    v_lo = max(0, min(resolution - 1, int(v_start * resolution)))
    v_hi = max(v_lo + 1, min(resolution, int(math.ceil(v_end * resolution))))
    h_lo = max(0, min(resolution - 1, int(h_start * resolution)))
    h_hi = max(h_lo + 1, min(resolution, int(math.ceil(h_end * resolution))))
    mask[:, :, v_lo:v_hi, h_lo:h_hi] = 1.0
    k = max(1, int(feather * resolution))
    if k > 1:
        mask = gaussian_blur_mask(mask, k | 1)
    return mask


def composite_motion_maps(maps_current, maps_reference, masks):
    out = []
    for i, (cur, ref) in enumerate(zip(maps_current, maps_reference)):
        mask = masks[i].to(cur.device, cur.dtype)
        if mask.shape[-2:] != cur.shape[-2:]:
            mask = F.interpolate(mask, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        out.append(cur * mask + ref * (1.0 - mask))
    return tuple(out)


def compute_head_motion_pixelspace(frames):
    vals = []
    for a, b in zip(frames[:-1], frames[1:]):
        h = a.shape[0]
        upper = int(h * 0.75)
        vals.append(np.abs(a[:upper].astype(np.float32) - b[:upper].astype(np.float32)).mean())
    return float(np.mean(vals)) if vals else 0.0


def compute_lip_motion_pixelspace(frames):
    vals = []
    for a, b in zip(frames[:-1], frames[1:]):
        h, w = a.shape[:2]
        y0, y1 = int(h * 0.63), int(h * 0.86)
        x0, x1 = int(w * 0.35), int(w * 0.65)
        vals.append(np.abs(a[y0:y1, x0:x1].astype(np.float32) - b[y0:y1, x0:x1].astype(np.float32)).mean())
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def shape_probe(model, device):
    source = torch.randn(1, 3, 512, 512, device=device)
    current = torch.randn(1, 3, 512, 512, device=device)
    f_r, g_r = model.app_encode(source)
    latent_current = model.mot_encode(current)
    latent_ref = model.mot_encode(source)
    adapted_current = model.id_adapt(latent_current, g_r)
    adapted_ref = model.id_adapt(latent_ref, g_r)
    maps_current = model.mot_decode(adapted_current)
    maps_ref = model.mot_decode(adapted_ref)
    print(f"[PROBE] appearance features={[tuple(x.shape) for x in f_r]}")
    print(f"[PROBE] identity={tuple(g_r.shape)} latent_current={tuple(latent_current.shape)} latent_ref={tuple(latent_ref.shape)}")
    print(f"[PROBE] current maps={[tuple(x.shape) for x in maps_current]}")
    print(f"[PROBE] reference maps={[tuple(x.shape) for x in maps_ref]}")
    print("[PROBE] decode(A=current_maps, B=reference_maps, C=reference_appearance_features)")
    return [m.shape[-1] for m in maps_current]


@torch.no_grad()
def render_clip(model, frames, masks, device):
    source_frame = frames[0]
    source = to_tensor(source_frame, device)
    f_r, i_r = model.app_encode(source)
    t_r = model.mot_encode(source)
    ta_r = model.id_adapt(t_r, i_r)
    ma_r = model.mot_decode(ta_r)

    normal_frames = []
    composite_frames = []
    for frame in frames:
        cur = to_tensor(frame, device)
        t_c = model.mot_encode(cur)
        ta_c = model.id_adapt(t_c, i_r)
        ma_c = model.mot_decode(ta_c)
        out_normal = model.decode(ma_c, ma_r, f_r)
        ma_comp = composite_motion_maps(ma_c, ma_r, masks)
        out_comp = model.decode(ma_comp, ma_r, f_r)
        normal_frames.append(from_tensor(out_normal))
        composite_frames.append(from_tensor(out_comp))
    return normal_frames, composite_frames


def save_mask_overlay(frame, mask, path):
    mask_up = F.interpolate(mask, size=(512, 512), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    base = cv2.resize(frame, (512, 512)).astype(np.float32)
    overlay = base.copy()
    red = np.zeros_like(base)
    red[..., 0] = 255
    alpha = np.clip(mask_up[..., None] * 0.55, 0, 0.55)
    overlay = overlay * (1 - alpha) + red * alpha
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(path)
    print(f"[SAVE] {path}")


def discover_videos(args):
    videos = []
    if args.driving_video and Path(args.driving_video).exists():
        videos.append(Path(args.driving_video))
    if args.video_dir and Path(args.video_dir).exists():
        for p in sorted(Path(args.video_dir).glob("*.mp4")):
            if p not in videos:
                videos.append(p)
            if len(videos) >= args.num_clips:
                break
    if not videos:
        raise RuntimeError("no validation videos found")
    return videos[: args.num_clips]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/IMTalker/checkpoints/renderer.ckpt")
    parser.add_argument("--driving_video", default="/workspace/moving_head.mp4")
    parser.add_argument("--video_dir", default="/workspace/synthetic_moving_head_source5/videos")
    parser.add_argument("--output_dir", default="/workspace/exps/map_compositing_clean")
    parser.add_argument("--num_clips", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--v_start", type=float, default=0.58)
    parser.add_argument("--v_end", type=float, default=0.88)
    parser.add_argument("--h_start", type=float, default=0.28)
    parser.add_argument("--h_end", type=float, default=0.72)
    parser.add_argument("--feather", type=float, default=0.18)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_renderer(args.checkpoint, device)
    resolutions = shape_probe(model, device)
    masks = [make_mouth_mask(r, args.v_start, args.v_end, args.h_start, args.h_end, args.feather) for r in resolutions]
    print(f"[MASK] resolutions={resolutions} params=v({args.v_start},{args.v_end}) h({args.h_start},{args.h_end}) feather={args.feather}")

    results = {}
    for video_path in discover_videos(args):
        frames, fps = read_video(video_path, args.max_frames)
        stem = video_path.stem
        save_mask_overlay(frames[0], masks[-1], out_dir / f"{stem}_mouth_mask_overlay.png")
        normal, comp = render_clip(model, frames, masks, device)
        triptych = []
        for drive, norm, cmp_frame in zip(frames, normal, comp):
            drive = cv2.resize(drive, (512, 512))
            triptych.append(np.concatenate([drive, norm, cmp_frame], axis=1))
        write_h264_video(triptych, out_dir / f"{stem}_drive_normal_composited.mp4", fps)

        head_normal = compute_head_motion_pixelspace(normal)
        head_comp = compute_head_motion_pixelspace(comp)
        lip_normal = compute_lip_motion_pixelspace(normal)
        lip_comp = compute_lip_motion_pixelspace(comp)
        metrics = {
            "video": str(video_path),
            "fps": fps,
            "frames": len(frames),
            "head_motion_normal": head_normal,
            "head_motion_composited": head_comp,
            "head_reduction": 1.0 - head_comp / max(head_normal, 1e-3),
            "lip_motion_normal": lip_normal,
            "lip_motion_composited": lip_comp,
            "lip_preservation": lip_comp / max(lip_normal, 1e-3),
        }
        results[stem] = metrics
        print(
            f"[METRIC] {stem} head normal={head_normal:.4f} comp={head_comp:.4f} "
            f"reduction={metrics['head_reduction']*100:.1f}% | "
            f"lip normal={lip_normal:.4f} comp={lip_comp:.4f} preservation={metrics['lip_preservation']*100:.1f}%"
        )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] metrics={out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
