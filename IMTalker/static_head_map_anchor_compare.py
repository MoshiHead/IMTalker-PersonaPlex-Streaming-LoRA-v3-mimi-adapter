import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from static_head_map_composite import (
    compute_head_motion_pixelspace,
    compute_lip_motion_pixelspace,
    discover_videos,
    from_tensor,
    load_renderer,
    make_mouth_mask,
    read_video,
    renderer_args,
    shape_probe,
    to_tensor,
    write_h264_video,
    save_mask_overlay,
)


MASK_CONFIGS = {
    "current": {
        "v_start": 0.58,
        "v_end": 0.88,
        "h_start": 0.28,
        "h_end": 0.72,
        "feather": 0.18,
    },
    "tight": {
        "v_start": 0.66,
        "v_end": 0.80,
        "h_start": 0.38,
        "h_end": 0.62,
        "feather": 0.14,
    },
    "tighter": {
        "v_start": 0.68,
        "v_end": 0.78,
        "h_start": 0.42,
        "h_end": 0.58,
        "feather": 0.10,
    },
    "lips_only": {
        "v_start": 0.70,
        "v_end": 0.77,
        "h_start": 0.43,
        "h_end": 0.57,
        "feather": 0.08,
    },
}


def resize_mask(mask, feat):
    mask = mask.to(feat.device, feat.dtype)
    if mask.shape[-2:] != feat.shape[-2:]:
        mask = F.interpolate(mask, size=feat.shape[-2:], mode="bilinear", align_corners=False)
    return mask


def composite_motion_maps(maps_current, maps_reference, masks):
    out = []
    for cur, ref, mask in zip(maps_current, maps_reference, masks):
        mask = resize_mask(mask, cur)
        out.append(cur * mask + ref * (1.0 - mask))
    return tuple(out)


def mean_anchor_motion_maps(maps_current, maps_reference, masks):
    out = []
    for cur, ref, mask in zip(maps_current, maps_reference, masks):
        mask = resize_mask(mask, cur)
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        cur_mean = (cur * mask).sum(dim=(2, 3), keepdim=True) / denom
        ref_mean = (ref * mask).sum(dim=(2, 3), keepdim=True) / denom
        anchored_cur = cur - cur_mean + ref_mean
        out.append(anchored_cur * mask + ref * (1.0 - mask))
    return tuple(out)


def centroid_anchor_motion_maps(maps_current, maps_reference, masks):
    """Shift the current mouth delta so its activation centroid lands on the mask center."""
    out = []
    for cur, ref, mask in zip(maps_current, maps_reference, masks):
        mask = resize_mask(mask, cur)
        delta = (cur - ref) * mask
        b, c, h, w = delta.shape
        energy = delta.square().mean(dim=1, keepdim=True) * mask
        mass = energy.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        yy = torch.linspace(-1, 1, h, device=cur.device, dtype=cur.dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1, 1, w, device=cur.device, dtype=cur.dtype).view(1, 1, 1, w)
        cy = (energy * yy).sum(dim=(2, 3), keepdim=True) / mass
        cx = (energy * xx).sum(dim=(2, 3), keepdim=True) / mass
        mask_mass = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        target_cy = (mask * yy).sum(dim=(2, 3), keepdim=True) / mask_mass
        target_cx = (mask * xx).sum(dim=(2, 3), keepdim=True) / mask_mass

        shift_x = (target_cx - cx).view(b, 1, 1)
        shift_y = (target_cy - cy).view(b, 1, 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=cur.device, dtype=cur.dtype),
            torch.linspace(-1, 1, w, device=cur.device, dtype=cur.dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        # grid_sample samples source coordinates. To move content toward target, sample opposite.
        grid[..., 0] = grid[..., 0] - shift_x
        grid[..., 1] = grid[..., 1] - shift_y
        shifted_delta = F.grid_sample(delta, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        out.append(ref + shifted_delta * mask)
    return tuple(out)


def compute_mouth_position_drift(frames):
    centers = []
    for frame in frames:
        h, w = frame.shape[:2]
        y0, y1 = int(h * 0.63), int(h * 0.86)
        x0, x1 = int(w * 0.35), int(w * 0.65)
        mouth = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
        # Dynamic dark-region threshold is more stable than a fixed 80 across clips.
        thresh_value = max(35, min(105, int(np.percentile(gray, 28))))
        _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.medianBlur(thresh, 3)
        moments = cv2.moments(thresh)
        if moments["m00"] > 1e-3:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centers.append([cx, cy])
    if len(centers) < 2:
        return 0.0
    centers = np.asarray(centers, dtype=np.float32)
    return float(np.std(centers, axis=0).mean())


@torch.no_grad()
def encode_reference(model, source_frame, device):
    source = to_tensor(source_frame, device)
    f_r, i_r = model.app_encode(source)
    t_r = model.mot_encode(source)
    ta_r = model.id_adapt(t_r, i_r)
    ma_r = model.mot_decode(ta_r)
    return f_r, i_r, ma_r


@torch.no_grad()
def render_methods(model, frames, method_masks, device):
    source_frame = frames[0]
    f_r, i_r, ma_r = encode_reference(model, source_frame, device)
    normal_frames = []
    method_frames = {name: [] for name in method_masks}

    for frame in frames:
        cur = to_tensor(frame, device)
        t_c = model.mot_encode(cur)
        ta_c = model.id_adapt(t_c, i_r)
        ma_c = model.mot_decode(ta_c)
        out_normal = model.decode(ma_c, ma_r, f_r)
        normal_frames.append(from_tensor(out_normal))

        for name, (kind, masks) in method_masks.items():
            if kind == "plain":
                maps = composite_motion_maps(ma_c, ma_r, masks)
            elif kind == "mean_anchor":
                maps = mean_anchor_motion_maps(ma_c, ma_r, masks)
            elif kind == "centroid_anchor":
                maps = centroid_anchor_motion_maps(ma_c, ma_r, masks)
            else:
                raise ValueError(kind)
            method_frames[name].append(from_tensor(model.decode(maps, ma_r, f_r)))
    return normal_frames, method_frames


def build_method_masks(resolutions):
    methods = {}
    for config_name, params in MASK_CONFIGS.items():
        masks = [make_mouth_mask(r, **params) for r in resolutions]
        methods[config_name] = ("plain", masks)
    methods["mean_anchor_current"] = (
        "mean_anchor",
        [make_mouth_mask(r, **MASK_CONFIGS["current"]) for r in resolutions],
    )
    methods["mean_anchor_tight"] = (
        "mean_anchor",
        [make_mouth_mask(r, **MASK_CONFIGS["tight"]) for r in resolutions],
    )
    methods["centroid_anchor_tight"] = (
        "centroid_anchor",
        [make_mouth_mask(r, **MASK_CONFIGS["tight"]) for r in resolutions],
    )
    return methods


def method_metrics(frames):
    return {
        "mouth_drift": compute_mouth_position_drift(frames),
        "lip_motion": compute_lip_motion_pixelspace(frames),
        "head_motion": compute_head_motion_pixelspace(frames),
    }


def save_method_video(driving, normal, method_frames, path, fps):
    out = []
    for d, n, m in zip(driving, normal, method_frames):
        d = cv2.resize(d, (512, 512))
        out.append(np.concatenate([d, n, m], axis=1))
    write_h264_video(out, path, fps)


def save_overview_video(driving, normal, method_frames_dict, method_order, path, fps):
    out = []
    for idx, d in enumerate(driving):
        tiles = [cv2.resize(d, (512, 512)), normal[idx]]
        for name in method_order:
            tiles.append(method_frames_dict[name][idx])
        out.append(np.concatenate(tiles, axis=1))
    write_h264_video(out, path, fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/IMTalker/checkpoints/renderer.ckpt")
    parser.add_argument("--driving_video", default="/workspace/moving_head.mp4")
    parser.add_argument("--video_dir", default="/workspace/synthetic_moving_head_source5/videos")
    parser.add_argument("--output_dir", default="/workspace/exps/map_compositing_anchored")
    parser.add_argument("--num_clips", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=80)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_renderer(args.checkpoint, device)
    resolutions = shape_probe(model, device)
    methods = build_method_masks(resolutions)

    print("[STEP0] Existing compositing baseline:")
    print("  composite = current * mask + reference * (1 - mask)")
    print("  previous mask params:", MASK_CONFIGS["current"])
    print("  map resolutions:", resolutions)
    print("[METHODS]", ", ".join(methods.keys()))

    # Save mask overlays using first available clip.
    videos = discover_videos(args)
    first_frames, _ = read_video(videos[0], max_frames=1)
    for name, params in MASK_CONFIGS.items():
        mask = make_mouth_mask(resolutions[-1], **params)
        save_mask_overlay(first_frames[0], mask, out_dir / f"mask_{name}_overlay.png")

    all_results = {}
    method_order = list(methods.keys())
    for video_path in videos:
        frames, fps = read_video(video_path, args.max_frames)
        stem = video_path.stem
        print(f"[CLIP] {stem} frames={len(frames)} fps={fps}")
        normal, method_frames = render_methods(model, frames, methods, device)

        normal_metrics = method_metrics(normal)
        clip_results = {"normal": normal_metrics, "methods": {}}
        for name in method_order:
            metrics = method_metrics(method_frames[name])
            metrics["lip_preservation_vs_normal"] = metrics["lip_motion"] / max(normal_metrics["lip_motion"], 1e-3)
            metrics["head_reduction_vs_normal"] = 1.0 - metrics["head_motion"] / max(normal_metrics["head_motion"], 1e-3)
            metrics["drift_reduction_vs_normal"] = 1.0 - metrics["mouth_drift"] / max(normal_metrics["mouth_drift"], 1e-3)
            clip_results["methods"][name] = metrics
            print(
                f"[METRIC] {stem:28s} {name:22s} "
                f"drift={metrics['mouth_drift']:.3f} "
                f"lip={metrics['lip_motion']:.3f} "
                f"head={metrics['head_motion']:.3f} "
                f"head_red={metrics['head_reduction_vs_normal']*100:.1f}% "
                f"lip_pres={metrics['lip_preservation_vs_normal']*100:.1f}%"
            )
            save_method_video(
                frames,
                normal,
                method_frames[name],
                out_dir / f"{stem}_{name}_drive_normal_method.mp4",
                fps,
            )

        save_overview_video(
            frames,
            normal,
            method_frames,
            method_order,
            out_dir / f"{stem}_ALL_METHODS_overview.mp4",
            fps,
        )
        all_results[stem] = clip_results

    # Pick winner on first/high clip: prioritize low drift, keep lip >= 60%, head reduction positive.
    first_stem = videos[0].stem
    candidates = []
    for name, metrics in all_results[first_stem]["methods"].items():
        lip_pres = metrics["lip_preservation_vs_normal"]
        head_red = metrics["head_reduction_vs_normal"]
        drift = metrics["mouth_drift"]
        penalty = 0.0
        if lip_pres < 0.60:
            penalty += (0.60 - lip_pres) * 100.0
        if head_red < 0.50:
            penalty += (0.50 - head_red) * 100.0
        candidates.append((drift + penalty, name, metrics))
    candidates.sort(key=lambda x: x[0])
    recommendation = {
        "clip_used": first_stem,
        "recommended_method": candidates[0][1],
        "recommended_metrics": candidates[0][2],
        "note": "Visual inspection still decides; metric uses dark-mouth centroid and can be noisy.",
    }
    all_results["recommendation"] = recommendation

    with open(out_dir / "anchored_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("[RECOMMENDATION]", json.dumps(recommendation, indent=2))
    print(f"[DONE] {out_dir / 'anchored_metrics.json'}")


if __name__ == "__main__":
    main()
