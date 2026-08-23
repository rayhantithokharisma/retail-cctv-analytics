from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.config import VideoConfig
from src.geometry import facing_normal, foot_point, head_yaw_asym, speed_bh_per_s
from src.smoothing import interpolate_gaps, median_filter_track


@dataclass
class Identity:
    identity_id: int
    raw_track_ids: list[int]
    frames: np.ndarray  # int32[T]
    t_s: np.ndarray  # float32[T]
    foot_xy: np.ndarray  # float32[T, 2], smoothed
    heights: np.ndarray  # float32[T]
    facing_normal: np.ndarray  # float32[T, 2], NaN row where unavailable
    head_yaw: np.ndarray  # float32[T], NaN where unavailable
    speed_bh_s: np.ndarray  # float32[T], smoothed
    kpts_raw: np.ndarray  # float32[T, 17, 3] UNsmoothed


def build_identities(
    obs: pd.DataFrame,
    stitch_map: dict[int, int],
    cfg: VideoConfig
) -> list[Identity]:
    """Merge obs rows by identity via stitch_map, sort by frame, run smoothing
    over foot_xy/speed/facing, leave kpts_raw untouched."""
    if obs.empty:
        return []

    obs_copy = obs.copy()
    obs_copy["identity_id"] = obs_copy["raw_track_id"].map(
        lambda tid: stitch_map.get(tid, tid)
    )

    smoothing_cfg = getattr(cfg, "smoothing", {})
    if isinstance(smoothing_cfg, dict):
        median_win = smoothing_cfg.get("median_win", 15)
        max_interp_gap = smoothing_cfg.get("max_interp_gap", 10)
    else:
        median_win = getattr(smoothing_cfg, "median_win", 15)
        max_interp_gap = getattr(smoothing_cfg, "max_interp_gap", 10)

    identities = []
    has_kp = "kp_x_00" in obs.columns

    for identity_id, group in obs_copy.groupby("identity_id"):
        # Deduplicate same frame if tracklets overlapped
        group = group.sort_values(["frame_idx", "det_conf"], ascending=[True, False])
        group = group.drop_duplicates(subset=["frame_idx"], keep="first")

        raw_track_ids = sorted(list(group["raw_track_id"].unique()))
        frames = group["frame_idx"].to_numpy(dtype=np.int32)
        t_s = group["t_s"].to_numpy(dtype=np.float32)
        heights = (group["y2"] - group["y1"]).to_numpy(dtype=np.float32)
        boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
        num_rows = len(group)

        # Extract raw keypoints
        kpts_raw = np.full((num_rows, 17, 3), fill_value=np.nan, dtype=np.float32)
        if has_kp:
            kp_cols_x = [f"kp_x_{i:02d}" for i in range(17)]
            kp_cols_y = [f"kp_y_{i:02d}" for i in range(17)]
            kp_cols_c = [f"kp_c_{i:02d}" for i in range(17)]
            kps_x = group[kp_cols_x].to_numpy(dtype=np.float32)
            kps_y = group[kp_cols_y].to_numpy(dtype=np.float32)
            kps_c = group[kp_cols_c].to_numpy(dtype=np.float32)
            for i in range(num_rows):
                kpts_raw[i, :, 0] = kps_x[i]
                kpts_raw[i, :, 1] = kps_y[i]
                kpts_raw[i, :, 2] = kps_c[i]

        # Compute raw foot points, facing normals, and head yaw
        foot_xy_raw = np.zeros((num_rows, 2), dtype=np.float32)
        facing_raw = np.full((num_rows, 2), fill_value=np.nan, dtype=np.float32)
        head_yaw_raw = np.full(num_rows, fill_value=np.nan, dtype=np.float32)

        for i in range(num_rows):
            kp_i = kpts_raw[i] if has_kp else None
            foot_xy_raw[i] = foot_point(boxes[i], kp_i)
            fn = facing_normal(kp_i)
            if fn is not None:
                facing_raw[i] = fn
            hy = head_yaw_asym(kp_i)
            if hy is not None:
                head_yaw_raw[i] = hy

        # Smooth foot positions
        foot_xy_filled, _ = interpolate_gaps(foot_xy_raw, frames, max_gap=max_interp_gap)
        foot_xy_smooth = median_filter_track(foot_xy_filled, frames, window_frames=median_win).astype(np.float32)

        # Compute raw speed from smoothed foot positions
        speed_raw = np.zeros(num_rows, dtype=np.float32)
        for i in range(1, num_rows):
            dt = float(t_s[i] - t_s[i - 1])
            if dt > 0:
                speed_raw[i] = speed_bh_per_s(foot_xy_smooth[i - 1], foot_xy_smooth[i], dt, cfg.scale_map)
        if num_rows > 1:
            speed_raw[0] = speed_raw[1]

        speed_smooth = median_filter_track(speed_raw, frames, window_frames=median_win).astype(np.float32)

        # Smooth facing normal and head yaw
        facing_filled, _ = interpolate_gaps(facing_raw, frames, max_gap=max_interp_gap)
        # Normalize facing normal
        facing_smooth = np.full_like(facing_filled, fill_value=np.nan, dtype=np.float32)
        valid_fn = ~np.isnan(facing_filled).any(axis=1)
        if valid_fn.any():
            norms = np.linalg.norm(facing_filled[valid_fn], axis=1, keepdims=True)
            facing_smooth[valid_fn] = (facing_filled[valid_fn] / (norms + 1e-9)).astype(np.float32)

        head_yaw_filled, _ = interpolate_gaps(head_yaw_raw, frames, max_gap=max_interp_gap)
        head_yaw_smooth = head_yaw_filled.astype(np.float32)

        identities.append(
            Identity(
                identity_id=int(identity_id),
                raw_track_ids=raw_track_ids,
                frames=frames,
                t_s=t_s,
                foot_xy=foot_xy_smooth,
                heights=heights,
                facing_normal=facing_smooth,
                head_yaw=head_yaw_smooth,
                speed_bh_s=speed_smooth,
                kpts_raw=kpts_raw,
            )
        )

    return sorted(identities, key=lambda ident: ident.identity_id)
