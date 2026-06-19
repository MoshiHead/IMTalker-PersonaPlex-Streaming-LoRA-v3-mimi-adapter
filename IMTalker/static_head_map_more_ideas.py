import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from static_head_map_anchor_compare import (
    MASK_CONFIGS,
    build_method_masks,
    compute_mouth_position_drift,
    method_metrics,
    resize_mask,
)
from static_head_map_composite import (
    discover_videos,
    from_tensor,
    load_renderer,
    make_mouth_mask,
    read_video,
    shape_probe,
    to_tensor,
    write_h264_video,
)


def composite_maps(cur_maps, ref_maps, masks):
    out = []
    for cur, ref, mask in zip(cur_maps, ref_maps, masks):
        mask = resize_mask(mask, cur)
        out.append(cur * mask + ref * (1 - mask))
    return tuple(out)


def filtered_delta_maps(cur_maps, ref_maps, masks, mode="highpass", gain=1.0):
    out = []
    for cur, ref, mask in zip(cur_maps, ref_maps, masks):
        mask = resize_mask(mask, cur)
        delta = (cur - ref) * mask
        h, w = delta.shape[-2:]
        k = max(3, int(min(h, w) * 0.25) | 1)
        if mode == "highpass":
            low = F.avg_pool2d(delta, kernel_size=k, stride=1, padding=k // 2)
            filt = (delta - low) * gain
        elif mode == "lowfreq_removed_light":
            low = F.avg_pool2d(delta, kernel_size=k, stride=1, padding=k // 2)
            filt = (delta - 0.5 * low) * gain
        elif mode == "gated":
            filt = delta * gain
        else:
            raise ValueError(mode)
        out.append(ref + filt * mask)
    return tuple(out)


def feature_centroid(delta, mask):
    b, c, h, w = delta.shape
    energy = delta.square().mean(dim=1, keepdim=True) * mask
    mass = energy.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    yy = torch.linspace(-1, 1, h, device=delta.device, dtype=delta.dtype).view(1, 1, h, 1)
    xx = torch.linspace(-1, 1, w, device=delta.device, dtype=delta.dtype).view(1, 1, 1, w)
    cy = (energy * yy).sum(dim=(2, 3), keepdim=True) / mass
    cx = (energy * xx).sum(dim=(2, 3), keepdim=True) / mass
    return cx, cy


def shift_feature(feat, shift_x, shift_y):
    b, c, h, w = feat.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype),
        torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    grid[..., 0] = grid[..., 0] - shift_x.view(b, 1, 1)
    grid[..., 1] = grid[..., 1] - shift_y.view(b, 1, 1)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class TemporalFeatureLock:
    def __init__(self, alpha=0.9):
        self.targets = None
        self.alpha = alpha

    def __call__(self, cur_maps, ref_maps, masks):
        if self.targets is None:
            self.targets = []
            for cur, ref, mask in zip(cur_maps, ref_maps, masks):
                mask = resize_mask(mask, cur)
                delta = (cur - ref) * mask
                cx, cy = feature_centroid(delta, mask)
                self.targets.append((cx.detach(), cy.detach()))

        out = []
        new_targets = []
        for idx, (cur, ref, mask) in enumerate(zip(cur_maps, ref_maps, masks)):
            mask = resize_mask(mask, cur)
            delta = (cur - ref) * mask
            cx, cy = feature_centroid(delta, mask)
            tx, ty = self.targets[idx]
            shifted = shift_feature(delta, tx - cx, ty - cy)
            out.append(ref + shifted * mask)
            # Slow EMA target to prevent sudden jumps but allow articulation changes.
            new_targets.append((
                (self.alpha * tx + (1 - self.alpha) * cx).detach(),
                (self.alpha * ty + (1 - self.alpha) * cy).detach(),
            ))
        self.targets = new_targets
        return tuple(out)


_LOCAL_MASK_CACHE = {}


def local_attention_mask(resolution, radius, device, dtype):
    key = (resolution, radius, str(device), str(dtype))
    cached = _LOCAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    coords = torch.stack(torch.meshgrid(
        torch.arange(resolution),
        torch.arange(resolution),
        indexing="ij",
    ), dim=-1).view(-1, 2)
    dist = (coords[:, None, :] - coords[None, :, :]).abs()
    keep = ((dist[..., 0] <= radius) & (dist[..., 1] <= radius)).float()
    mask = keep.view(1, 1, resolution * resolution, resolution * resolution).to(device=device, dtype=dtype)
    _LOCAL_MASK_CACHE[key] = mask
    return mask


def decode_local(model, A, B, C, radius_by_res):
    num_levels = len(model.spatial_dims)
    aligned_features = [None] * num_levels
    attention_map = None
    for i in range(num_levels):
        attention_block = model.imt[i]
        if attention_block.is_standard_attention:
            a = A[i]
            bsz, ch, h, w = a.shape
            A_seq = A[i].flatten(2).transpose(1, 2)
            B_seq = B[i].flatten(2).transpose(1, 2)
            C_seq = C[i].flatten(2).transpose(1, 2)
            radius = radius_by_res.get(h, max(1, h // 8))
            mask = local_attention_mask(h, radius, a.device, a.dtype)
            out_seq, attention_map = attention_block.block_efc(A_seq, B_seq, C_seq, mask=mask)
            aligned_features[i] = out_seq.transpose(1, 2).view(bsz, ch, h, w)
        else:
            aligned_features[i] = attention_block.fine_stage(C[i], attn=attention_map)
    return model.frame_decoder(aligned_features)


def load_pca(path, device, k):
    if not path or not Path(path).exists():
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[0] != 32:
        print(f"[PCA] skip invalid shape {arr.shape} path={path}")
        return None
    kk = min(k, arr.shape[1])
    p = torch.from_numpy(arr[:, :kk]).float().to(device)
    print(f"[PCA] loaded {path} shape={arr.shape} using k={kk}")
    return p


@torch.no_grad()
def render_all(model, frames, masks_tight, masks_current, device, pca_mats):
    source = to_tensor(frames[0], device)
    f_r, i_r = model.app_encode(source)
    t_ref = model.mot_encode(source)
    ma_r = model.mot_decode(model.id_adapt(t_ref, i_r))

    method_frames = {
        "tight": [],
        "filtered_light_current": [],
        "highpass_current": [],
        "gated_tight_075": [],
        "temporal_lock_tight": [],
        "temporal_local_tight": [],
        "local_attention_tight": [],
    }
    for k in pca_mats:
        method_frames[f"pca_k{k}_tight"] = []
        method_frames[f"pca_k{k}_local_tight"] = []
    normal = []
    temporal = TemporalFeatureLock(alpha=0.92)
    temporal_local = TemporalFeatureLock(alpha=0.92)
    pca_energy = {k: {"lip_orig": [], "lip_clean": [], "total_orig": [], "total_clean": []} for k in pca_mats}

    for frame in frames:
        cur = to_tensor(frame, device)
        t_cur = model.mot_encode(cur)
        ma_c = model.mot_decode(model.id_adapt(t_cur, i_r))
        normal.append(from_tensor(model.decode(ma_c, ma_r, f_r)))

        maps_tight = composite_maps(ma_c, ma_r, masks_tight)
        method_frames["tight"].append(from_tensor(model.decode(maps_tight, ma_r, f_r)))

        maps_filt = filtered_delta_maps(ma_c, ma_r, masks_current, mode="lowfreq_removed_light", gain=1.0)
        method_frames["filtered_light_current"].append(from_tensor(model.decode(maps_filt, ma_r, f_r)))

        maps_hp = filtered_delta_maps(ma_c, ma_r, masks_current, mode="highpass", gain=1.5)
        method_frames["highpass_current"].append(from_tensor(model.decode(maps_hp, ma_r, f_r)))

        maps_gate = filtered_delta_maps(ma_c, ma_r, masks_tight, mode="gated", gain=0.75)
        method_frames["gated_tight_075"].append(from_tensor(model.decode(maps_gate, ma_r, f_r)))

        maps_temp = temporal(ma_c, ma_r, masks_tight)
        method_frames["temporal_lock_tight"].append(from_tensor(model.decode(maps_temp, ma_r, f_r)))

        maps_temp_local = temporal_local(ma_c, ma_r, masks_tight)
        method_frames["temporal_local_tight"].append(from_tensor(
            decode_local(model, maps_temp_local, ma_r, f_r, {8: 1, 16: 2, 32: 3, 64: 4})
        ))

        method_frames["local_attention_tight"].append(from_tensor(
            decode_local(model, maps_tight, ma_r, f_r, {8: 1, 16: 2, 32: 3, 64: 4})
        ))

        for k, p in pca_mats.items():
            delta = t_cur - t_ref
            t_pca = t_ref + (delta @ p) @ p.T
            clean_delta = t_pca - t_ref
            pca_energy[k]["lip_orig"].append(torch.linalg.norm(delta @ p).item())
            pca_energy[k]["lip_clean"].append(torch.linalg.norm(clean_delta @ p).item())
            pca_energy[k]["total_orig"].append(torch.linalg.norm(delta).item())
            pca_energy[k]["total_clean"].append(torch.linalg.norm(clean_delta).item())
            ma_pca = model.mot_decode(model.id_adapt(t_pca, i_r))
            maps_pca = composite_maps(ma_pca, ma_r, masks_tight)
            method_frames[f"pca_k{k}_tight"].append(from_tensor(model.decode(maps_pca, ma_r, f_r)))
            method_frames[f"pca_k{k}_local_tight"].append(from_tensor(
                decode_local(model, maps_pca, ma_r, f_r, {8: 1, 16: 2, 32: 3, 64: 4})
            ))
        if len(normal) % 10 == 0:
            print(f"[FRAME] rendered {len(normal)}/{len(frames)}", flush=True)

    energy_summary = {}
    for k, vals in pca_energy.items():
        lip_orig = float(np.mean(vals["lip_orig"])) if vals["lip_orig"] else 0.0
        lip_clean = float(np.mean(vals["lip_clean"])) if vals["lip_clean"] else 0.0
        total_orig = float(np.mean(vals["total_orig"])) if vals["total_orig"] else 0.0
        total_clean = float(np.mean(vals["total_clean"])) if vals["total_clean"] else 0.0
        energy_summary[k] = {
            "lip_energy_orig": lip_orig,
            "lip_energy_clean": lip_clean,
            "lip_energy_ratio_clean_over_orig": lip_clean / max(lip_orig, 1e-8),
            "total_delta_energy_orig": total_orig,
            "total_delta_energy_clean": total_clean,
            "total_delta_energy_ratio_clean_over_orig": total_clean / max(total_orig, 1e-8),
        }

    return normal, method_frames, energy_summary


def save_overview(frames, normal, methods, order, path, fps):
    out = []
    for i, d in enumerate(frames):
        tiles = [cv2.resize(d, (512, 512)), normal[i]]
        for name in order:
            tiles.append(methods[name][i])
        out.append(np.concatenate(tiles, axis=1))
    write_h264_video(out, path, fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/IMTalker/checkpoints/renderer.ckpt")
    parser.add_argument("--driving_video", default="/workspace/moving_head.mp4")
    parser.add_argument("--video_dir", default="/workspace/synthetic_moving_head_source5/videos")
    parser.add_argument("--output_dir", default="/workspace/exps/map_compositing_more_ideas")
    parser.add_argument("--num_clips", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--pca_path", default="/workspace/static_imt/pca/lip_projection_matrix_full.npy")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_renderer(args.checkpoint, device)
    resolutions = shape_probe(model, device)

    masks_tight = [make_mouth_mask(r, **MASK_CONFIGS["tight"]) for r in resolutions]
    masks_current = [make_mouth_mask(r, **MASK_CONFIGS["current"]) for r in resolutions]
    pca_mats = {}
    for k in (6, 8, 12):
        p = load_pca(args.pca_path, device, k)
        if p is not None:
            pca_mats[k] = p

    videos = discover_videos(args)
    results = {}
    for video_path in videos:
        frames, fps = read_video(video_path, args.max_frames)
        stem = video_path.stem
        print(f"[CLIP] {stem} frames={len(frames)} fps={fps}")
        normal, methods, pca_energy = render_all(model, frames, masks_tight, masks_current, device, pca_mats)
        order = list(methods.keys())
        normal_metrics = method_metrics(normal)
        results[stem] = {"normal": normal_metrics, "pca_lip_energy_check": pca_energy, "methods": {}}
        for k, energy in pca_energy.items():
            print(
                f"[PCA_ENERGY] {stem} k={k} lip_ratio={energy['lip_energy_ratio_clean_over_orig']:.3f} "
                f"total_ratio={energy['total_delta_energy_ratio_clean_over_orig']:.3f}"
            )
        for name in order:
            m = method_metrics(methods[name])
            m["head_reduction_vs_normal"] = 1 - m["head_motion"] / max(normal_metrics["head_motion"], 1e-3)
            m["lip_preservation_vs_normal"] = m["lip_motion"] / max(normal_metrics["lip_motion"], 1e-3)
            m["drift_reduction_vs_normal"] = 1 - m["mouth_drift"] / max(normal_metrics["mouth_drift"], 1e-3)
            results[stem]["methods"][name] = m
            print(
                f"[METRIC] {stem:28s} {name:24s} drift={m['mouth_drift']:.3f} "
                f"lip={m['lip_motion']:.3f} head={m['head_motion']:.3f} "
                f"head_red={m['head_reduction_vs_normal']*100:.1f}% "
                f"lip_pres={m['lip_preservation_vs_normal']*100:.1f}%"
            )
        save_overview(frames, normal, methods, order, out_dir / f"{stem}_MORE_IDEAS_overview.mp4", fps)

    first = videos[0].stem
    candidates = []
    for name, m in results[first]["methods"].items():
        penalty = 0.0
        if m["lip_preservation_vs_normal"] < 0.55:
            penalty += (0.55 - m["lip_preservation_vs_normal"]) * 100
        if m["head_reduction_vs_normal"] < 0.70:
            penalty += (0.70 - m["head_reduction_vs_normal"]) * 100
        candidates.append((m["mouth_drift"] + penalty, name, m))
    candidates.sort(key=lambda x: x[0])
    results["recommendation"] = {
        "clip_used": first,
        "recommended_method": candidates[0][1],
        "recommended_metrics": candidates[0][2],
        "note": "Visual inspection decides. PCA uses locally found lip projection matrix if present.",
    }
    with open(out_dir / "more_ideas_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[RECOMMENDATION]", json.dumps(results["recommendation"], indent=2))
    print(f"[DONE] {out_dir / 'more_ideas_metrics.json'}")


if __name__ == "__main__":
    main()
