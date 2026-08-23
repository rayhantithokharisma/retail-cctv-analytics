import numpy as np
from src.config import Task1Config
from src.geometry import cos_angle, nearest_point_on_segment, point_in_polygon, signed_side
from src.identity import Identity


def orientation_score(
    identity: Identity,
    cfg: Task1Config
) -> np.ndarray:
    """Per-frame A(t). target -> nearest point on entrance line (geometry.nearest_point_on_segment).
    A_body = clamp01((cos(n, to_target) - cos(floor_deg)) / (1 - cos(floor_deg)))
    A_head = same mapping but blend head_yaw sign onto A_body per plan §5's head-yaw definition
    A = max(A_body, 0.7*A_body + 0.3*A_head), with NaN-safe fallback to A_body alone
    when head signal is unavailable that frame."""
    n_frames = len(identity.frames)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    line_p1 = cfg.entrance_line.p1
    line_p2 = cfg.entrance_line.p2
    cos_floor = np.cos(np.radians(cfg.orientation_cos_floor_deg))

    A = np.zeros(n_frames, dtype=np.float32)

    for t in range(n_frames):
        foot = identity.foot_xy[t]
        target = nearest_point_on_segment(foot, line_p1, line_p2)
        to_target = np.array([target[0] - foot[0], target[1] - foot[1]], dtype=np.float64)
        norm = np.linalg.norm(to_target)
        if norm > 1e-6:
            to_target = to_target / norm
        else:
            to_target = np.array([0.0, 1.0], dtype=np.float64)

        fn = identity.facing_normal[t]
        if fn is not None and not np.isnan(fn).any():
            cos_n = cos_angle(fn, to_target)
            A_body = float(np.clip((cos_n - cos_floor) / (1.0 - cos_floor + 1e-9), 0.0, 1.0))
        else:
            A_body = 0.0

        hy = identity.head_yaw[t]
        if hy is not None and not np.isnan(hy):
            # Head yaw asymmetry: positive is turning toward camera/store
            A_head = float(np.clip((hy + 1.0) / 2.0, 0.0, 1.0))
            A[t] = max(A_body, 0.7 * A_body + 0.3 * A_head)
        else:
            A[t] = A_body

    return A


def deceleration_score(identity: Identity, cfg: Task1Config) -> np.ndarray:
    """v_cruise = np.nanpercentile(identity.speed_bh_s, 80)
    S = clamp01((v_cruise - v) / (0.7*v_cruise)); force S=1 where v < 0.25."""
    v = identity.speed_bh_s
    n_frames = len(v)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    valid_v = v[~np.isnan(v)]
    if len(valid_v) == 0:
        return np.zeros(n_frames, dtype=np.float32)

    v_cruise = float(np.nanpercentile(valid_v, 80))
    if v_cruise < 1e-4:
        v_cruise = 0.5

    S = np.clip((v_cruise - v) / (0.7 * v_cruise + 1e-9), 0.0, 1.0).astype(np.float32)
    # Force S=1 where speed is effectively stopped (< 0.25 bh/s)
    S[v < 0.25] = 1.0
    return S


def approach_score(identity: Identity, cfg: Task1Config) -> np.ndarray:
    """dist_bh(t) = -signed_side(entrance_line, foot) / body_height (outside is negative signed_side).
    full credit (1.0) once (dist_bh[0] - dist_bh[t]) >= 1.5, OR foot inside storefront_zone;
    linear ramp below that."""
    n_frames = len(identity.frames)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    in_storefront = point_in_polygon(identity.foot_xy, cfg.storefront_zone)
    P = np.zeros(n_frames, dtype=np.float32)

    # Calculate distance to entrance line in body heights
    # Outside store: signed_side < 0. Distance from line = -signed_side / bh
    dist_bh = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        foot = identity.foot_xy[t]
        side_val = signed_side(cfg.entrance_line, foot)
        med_h = float(np.median(identity.heights)) if len(identity.heights) > 0 else 100.0
        # Line length normalization factor for cross product
        d_line = np.linalg.norm(np.array(cfg.entrance_line.p2) - np.array(cfg.entrance_line.p1))
        # Perpendicular distance in pixels = abs(cross) / d_line
        perp_dist_px = -side_val / max(d_line, 1.0)
        dist_bh[t] = perp_dist_px / max(med_h, 1.0)

    dist_bh_0 = dist_bh[0]

    for t in range(n_frames):
        if in_storefront[t] or dist_bh[t] <= 0.0:
            P[t] = 1.0
        else:
            delta = dist_bh_0 - dist_bh[t]
            P[t] = float(np.clip(delta / 1.5, 0.0, 1.0))

    return P


def dwell_score(identity: Identity, cfg: Task1Config) -> np.ndarray:
    """clamp01(continuous seconds inside storefront_zone / 3.0), computed via a running
    counter that resets to 0 the instant the point exits the zone (not a global count)."""
    n_frames = len(identity.frames)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    in_zone = point_in_polygon(identity.foot_xy, cfg.storefront_zone)
    dwell = np.zeros(n_frames, dtype=np.float32)
    cur_dwell_s = 0.0

    for t in range(n_frames):
        if in_zone[t]:
            if t > 0:
                dt = float(identity.t_s[t] - identity.t_s[t - 1])
                cur_dwell_s += max(0.0, dt)
            else:
                cur_dwell_s = 0.033
        else:
            cur_dwell_s = 0.0
        dwell[t] = float(np.clip(cur_dwell_s / 3.0, 0.0, 1.0))

    return dwell



