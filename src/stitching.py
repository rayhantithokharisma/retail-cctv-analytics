from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from src.config import ScaleMap, StitchConfig
from src.geometry import cos_angle, foot_point


@dataclass
class Tracklet:
    raw_track_id: int
    frames: np.ndarray  # int32[T]
    foot_xy: np.ndarray  # float32[T, 2]
    heights: np.ndarray  # float32[T]
    embed: np.ndarray | None  # float32[D] mean embedding, None if no ReID available


def build_tracklets(obs: pd.DataFrame, embeddings: np.ndarray) -> list[Tracklet]:
    """Group obs by raw_track_id, compute foot_point per row via geometry.foot_point,
    take heights = y2-y1, embed = mean of embeddings[embed_ref] over rows with embed_ref>=0."""
    if obs.empty:
        return []

    tracklets = []
    has_kp = "kp_x_00" in obs.columns

    for track_id, group in obs.groupby("raw_track_id"):
        group = group.sort_values("frame_idx")
        frames = group["frame_idx"].to_numpy(dtype=np.int32)
        heights = (group["y2"] - group["y1"]).to_numpy(dtype=np.float32)

        boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
        num_rows = len(group)
        foot_pts = np.zeros((num_rows, 2), dtype=np.float32)

        if has_kp:
            kp_cols_x = [f"kp_x_{i:02d}" for i in range(17)]
            kp_cols_y = [f"kp_y_{i:02d}" for i in range(17)]
            kp_cols_c = [f"kp_c_{i:02d}" for i in range(17)]
            kps_x = group[kp_cols_x].to_numpy(dtype=np.float32)
            kps_y = group[kp_cols_y].to_numpy(dtype=np.float32)
            kps_c = group[kp_cols_c].to_numpy(dtype=np.float32)

            for i in range(num_rows):
                kp_i = np.stack([kps_x[i], kps_y[i], kps_c[i]], axis=1)
                foot_pts[i] = foot_point(boxes[i], kp_i)
        else:
            for i in range(num_rows):
                foot_pts[i] = foot_point(boxes[i], None)

        # Mean embedding across valid embed_ref
        embed = None
        if "embed_ref" in group.columns and embeddings is not None and len(embeddings) > 0:
            refs = group["embed_ref"].to_numpy()
            valid_refs = refs[(refs >= 0) & (refs < len(embeddings))]
            if len(valid_refs) > 0:
                valid_feats = embeddings[valid_refs]
                mean_feat = np.mean(valid_feats, axis=0)
                norm = np.linalg.norm(mean_feat)
                if norm > 1e-6:
                    embed = (mean_feat / norm).astype(np.float32)

        tracklets.append(
            Tracklet(
                raw_track_id=int(track_id),
                frames=frames,
                foot_xy=foot_pts,
                heights=heights,
                embed=embed,
            )
        )
    return tracklets


def extrapolate(tracklet: Tracklet, to_frame: int, fps: float = 30.0) -> np.ndarray:
    """Fit velocity via linear regression on the tracklet's LAST 10 points (frame vs foot_xy),
    project to `to_frame`. Returns predicted (x,y)."""
    n_pts = len(tracklet.frames)
    if n_pts == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    if n_pts == 1:
        return tracklet.foot_xy[0].copy()

    k = min(10, n_pts)
    frames_fit = tracklet.frames[-k:].astype(np.float64)
    pts_fit = tracklet.foot_xy[-k:].astype(np.float64)

    t_rel = frames_fit - frames_fit[-1]
    denom = np.sum(t_rel**2)
    if denom < 1e-9:
        return pts_fit[-1].astype(np.float32)

    # Slope per frame: sum(t_rel * (pts - mean_pts)) / denom
    t_target = float(to_frame - frames_fit[-1])
    vx = np.sum(t_rel * (pts_fit[:, 0] - pts_fit[-1, 0])) / denom
    vy = np.sum(t_rel * (pts_fit[:, 1] - pts_fit[-1, 1])) / denom

    pred_x = pts_fit[-1, 0] + vx * t_target
    pred_y = pts_fit[-1, 1] + vy * t_target
    return np.array([pred_x, pred_y], dtype=np.float32)


def get_velocity_vector(tracklet: Tracklet, from_tail: bool = True) -> np.ndarray:
    """Computes velocity vector (dx, dy) for heading comparison."""
    n = len(tracklet.frames)
    if n < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    k = min(10, n)
    if from_tail:
        pts = tracklet.foot_xy[-k:]
    else:
        pts = tracklet.foot_xy[:k]
    vec = pts[-1] - pts[0]
    return vec.astype(np.float32)


def gate_pair(a: Tracklet, b: Tracklet, cfg: StitchConfig, scale_map: ScaleMap) -> bool:
    """All four gates from plan §4.2 step 1-4, in cheapest-first order:
    1. 0 < b.frames[0]-a.frames[-1] <= cfg.max_gap_frames
    2. motion gate using extrapolate()
    3. scale gate on heights
    4. direction gate: angle between a's and b's velocity headings <= cfg.direction_gate_deg
    Short-circuit False on first failing gate."""
    # 1. Temporal gap
    gap_frames = int(b.frames[0] - a.frames[-1])
    if gap_frames <= 0 or gap_frames > cfg.max_gap_frames:
        return False

    # 2. Motion gate
    p_hat_a = extrapolate(a, to_frame=int(b.frames[0]))
    p_b = b.foot_xy[0]
    mid_y = float((p_hat_a[1] + p_b[1]) / 2.0)
    h_bar = scale_map.expected_height(mid_y)
    motion_residual_bh = float(np.linalg.norm(p_hat_a - p_b) / h_bar)
    if motion_residual_bh > cfg.motion_gate_bh:
        return False

    # 3. Scale gate
    h_a = float(np.median(a.heights))
    h_b = float(np.median(b.heights))
    max_h = max(h_a, h_b, 1e-6)
    scale_diff = abs(h_a - h_b) / max_h
    if scale_diff > cfg.scale_gate_frac:
        return False

    # 4. Direction gate
    v_a = get_velocity_vector(a, from_tail=True)
    v_b = get_velocity_vector(b, from_tail=False)
    norm_a = np.linalg.norm(v_a)
    norm_b = np.linalg.norm(v_b)

    # Only enforce direction if both have enough displacement (> 5 px)
    if norm_a > 5.0 and norm_b > 5.0:
        cos_val = cos_angle(v_a, v_b)
        cos_thresh = np.cos(np.radians(cfg.direction_gate_deg))
        if cos_val < cos_thresh:
            return False

    return True


def pair_cost(a: Tracklet, b: Tracklet, cfg: StitchConfig, scale_map: ScaleMap) -> float:
    """Weighted sum from plan §4.2 — appearance term is 1.0 (max cost) if either embed is None,
    never silently skipped."""
    if a.embed is not None and b.embed is not None:
        sim = float(np.dot(a.embed, b.embed) / (np.linalg.norm(a.embed) * np.linalg.norm(b.embed) + 1e-9))
        sim = max(-1.0, min(1.0, sim))
        app_cost = 1.0 - sim
    else:
        app_cost = 1.0

    p_hat_a = extrapolate(a, to_frame=int(b.frames[0]))
    p_b = b.foot_xy[0]
    mid_y = float((p_hat_a[1] + p_b[1]) / 2.0)
    h_bar = scale_map.expected_height(mid_y)
    motion_residual_bh = float(np.linalg.norm(p_hat_a - p_b) / h_bar)
    motion_cost = motion_residual_bh / max(cfg.motion_gate_bh, 1e-6)

    h_a = float(np.median(a.heights))
    h_b = float(np.median(b.heights))
    max_h = max(h_a, h_b, 1e-6)
    scale_cost = (abs(h_a - h_b) / max_h) / max(cfg.scale_gate_frac, 1e-6)

    cost = cfg.w_appearance * app_cost + cfg.w_motion * motion_cost + cfg.w_scale * scale_cost
    return float(cost)


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def merge_tracklets(tracklets: list[Tracklet]) -> Tracklet:
    """Merges a list of tracklets belonging to the same cluster into one combined tracklet."""
    sorted_ts = sorted(tracklets, key=lambda t: t.frames[0])
    frames = np.concatenate([t.frames for t in sorted_ts])
    foot_xy = np.vstack([t.foot_xy for t in sorted_ts])
    heights = np.concatenate([t.heights for t in sorted_ts])

    # Re-sort in case of overlapping frames
    order = np.argsort(frames)
    frames = frames[order]
    foot_xy = foot_xy[order]
    heights = heights[order]

    valid_embeds = [t.embed for t in sorted_ts if t.embed is not None]
    if valid_embeds:
        mean_embed = np.mean(valid_embeds, axis=0)
        norm = np.linalg.norm(mean_embed)
        embed = (mean_embed / norm).astype(np.float32) if norm > 1e-6 else None
    else:
        embed = None

    return Tracklet(
        raw_track_id=sorted_ts[0].raw_track_id,
        frames=frames,
        foot_xy=foot_xy,
        heights=heights,
        embed=embed,
    )


def stitch(tracklets: list[Tracklet], cfg: StitchConfig, scale_map: ScaleMap) -> dict[int, int]:
    """Build bipartite candidate graph (a ends before b starts, gate_pair passes),
    solve via scipy.optimize.linear_sum_assignment on the cost matrix (pad with cfg.cost_max
    +eps for non-edges so they're never chosen), accept edges with cost < cfg.cost_max,
    union-find merge, REPEAT to fixed point (cap at 5 iterations, log if it doesn't converge).
    Returns raw_track_id -> identity_id map."""
    if not tracklets:
        return {}

    track_ids = [t.raw_track_id for t in tracklets]
    uf = UnionFind(track_ids)

    for iteration in range(5):
        # Group tracklets by current root
        clusters = {}
        for t in tracklets:
            root = uf.find(t.raw_track_id)
            clusters.setdefault(root, []).append(t)

        merged_clusters = {root: merge_tracklets(ts) for root, ts in clusters.items()}
        root_keys = list(merged_clusters.keys())
        N = len(root_keys)
        if N <= 1:
            break

        # Find candidate pairs (earlier -> later)
        candidate_pairs = []
        for i in range(N):
            a = merged_clusters[root_keys[i]]
            for j in range(N):
                if i == j:
                    continue
                b = merged_clusters[root_keys[j]]
                if a.frames[-1] < b.frames[0] and gate_pair(a, b, cfg, scale_map):
                    c = pair_cost(a, b, cfg, scale_map)
                    if c < cfg.cost_max:
                        candidate_pairs.append((i, j, c))

        if not candidate_pairs:
            break

        # Solve bipartite matching over candidate pairs
        # Sources = i, Targets = j
        cost_matrix = np.full((N, N), fill_value=cfg.cost_max + 1.0, dtype=np.float64)
        for i, j, c in candidate_pairs:
            cost_matrix[i, j] = c

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        merged_count = 0
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < cfg.cost_max:
                uf.union(root_keys[r], root_keys[c])
                merged_count += 1

        if merged_count == 0:
            break

    # Build dense 1-indexed (or 0-indexed) identity IDs
    root_to_id = {}
    id_map = {}
    next_id = 1
    for t_id in track_ids:
        root = uf.find(t_id)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        id_map[t_id] = root_to_id[root]

    return id_map
