import numpy as np
import pandas as pd
from src.config import Task1Config
from src.features import (
    approach_score,
    deceleration_score,
    dwell_score,
    orientation_score,
)
from src.geometry import point_in_polygon, signed_side
from src.identity import Identity
from src.smoothing import median_filter_track


def is_candidate(identity: Identity, cfg: Task1Config, is_staff: bool = False) -> bool:
    """>=1.0s of frames with foot in hallway_polygon AND median(heights)>=cfg.min_bbox_height
    AND not staff."""
    if is_staff:
        return False

    n_frames = len(identity.frames)
    if n_frames == 0:
        return False

    med_h = float(np.median(identity.heights))
    if med_h < cfg.min_bbox_height:
        return False

    in_hallway = point_in_polygon(identity.foot_xy, cfg.hallway)
    dur_in_hallway_s = np.sum(in_hallway) / 30.0  # standard fps

    return dur_in_hallway_s >= 1.0


def interest_timeline(identity: Identity, cfg: Task1Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes interest timeline and individual channel components.
    Returns (I_smooth, A, S, P, D)."""
    A = orientation_score(identity, cfg)
    S = deceleration_score(identity, cfg)
    P = approach_score(identity, cfg)
    D = dwell_score(identity, cfg)

    I = cfg.w_orientation * A + cfg.w_decel * S + cfg.w_approach * P + cfg.w_dwell * D
    I_smooth = median_filter_track(I, identity.frames, window_frames=15)
    return I_smooth, A, S, P, D


def decide_interested(I: np.ndarray, cfg: Task1Config, fps: float = 30.0) -> bool:
    """True iff any contiguous run of frames with I>=threshold spans >= interest_min_duration_s."""
    if len(I) == 0:
        return False

    required_frames = int(round(cfg.interest_min_duration_s * fps))
    cur_streak = 0
    for val in I:
        if val >= cfg.interest_threshold:
            cur_streak += 1
            if cur_streak >= required_frames:
                return True
        else:
            cur_streak = 0
    return False


def decide_entered(identity: Identity, cfg: Task1Config, fps: float = 30.0) -> bool:
    """Find sign changes of signed_side(entrance_line, foot) that pass through the
    [-deadzone_bh, +deadzone_bh] band (not a bare sign flip — must actually traverse the band).
    For each candidate crossing into store-side: check the person STAYS store-side for
    >= entered_dwell_s continuously AND penetration depth (max signed distance past the line
    over that dwell window) >= entered_depth_bh. First such crossing wins (latched True)."""
    n_frames = len(identity.frames)
    if n_frames == 0:
        return False

    med_h = float(np.median(identity.heights)) if len(identity.heights) > 0 else 100.0
    d_line = float(np.linalg.norm(np.array(cfg.entrance_line.p2) - np.array(cfg.entrance_line.p1)))
    d_line = max(d_line, 1.0)

    # Compute signed distance in body heights (store-side > 0, hallway < 0)
    sides = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        # cross product: >0 is store-side, <0 is hallway
        cross = signed_side(cfg.entrance_line, identity.foot_xy[t])
        perp_px = cross / d_line
        sides[t] = perp_px / max(med_h, 1.0)

    deadzone = float(cfg.entered_deadzone_bh)
    min_dwell_frames = int(round(cfg.entered_dwell_s * fps))

    # Look for transitions from outside (< -deadzone) to inside (> 0)
    was_outside = False
    for t in range(n_frames):
        if sides[t] < -deadzone:
            was_outside = True
        elif was_outside and sides[t] > 0:
            # Candidate crossing point! Check dwell & penetration
            dwell_slice = sides[t : t + min_dwell_frames]
            if len(dwell_slice) >= min_dwell_frames and np.all(dwell_slice > 0):
                max_depth = float(np.max(dwell_slice))
                if max_depth >= cfg.entered_depth_bh:
                    return True

    return False


def run_task1(
    identities: list[Identity],
    cfg: Task1Config,
    staff_ids: set[int] | None = None
) -> pd.DataFrame:
    """Returns the audit dataframe from plan §6.1 Outputs — one row per candidate identity
    with: id, first_t, last_t, seconds_in_hallway, peak_interest, channel contributions at
    peak, interested, entered, low_confidence."""
    staff_set = staff_ids or set()
    records = []

    for ident in identities:
        is_staff = ident.identity_id in staff_set
        if not is_candidate(ident, cfg, is_staff=is_staff):
            continue

        in_h = point_in_polygon(ident.foot_xy, cfg.hallway)
        seconds_in_hallway = float(np.sum(in_h) / 30.0)

        I_smooth, A, S, P, D = interest_timeline(ident, cfg)

        peak_idx = int(np.argmax(I_smooth)) if len(I_smooth) > 0 else 0
        peak_interest = float(I_smooth[peak_idx]) if len(I_smooth) > 0 else 0.0

        interested = decide_interested(I_smooth, cfg)
        entered = decide_entered(ident, cfg)

        # Low confidence flag if average keypoint confidence is low or track very short
        avg_conf = float(np.mean(ident.kpts_raw[:, :, 2])) if len(ident.kpts_raw) > 0 else 0.0
        low_confidence = avg_conf < 0.35 or len(ident.frames) < 45

        records.append({
            "identity_id": ident.identity_id,
            "first_t_s": float(ident.t_s[0]),
            "last_t_s": float(ident.t_s[-1]),
            "seconds_in_hallway": seconds_in_hallway,
            "peak_interest": peak_interest,
            "orientation_at_peak": float(A[peak_idx]) if len(A) > 0 else 0.0,
            "decel_at_peak": float(S[peak_idx]) if len(S) > 0 else 0.0,
            "approach_at_peak": float(P[peak_idx]) if len(P) > 0 else 0.0,
            "dwell_at_peak": float(D[peak_idx]) if len(D) > 0 else 0.0,
            "interested": interested,
            "entered": entered,
            "low_confidence": low_confidence,
        })

    if not records:
        return pd.DataFrame(columns=[
            "identity_id", "first_t_s", "last_t_s", "seconds_in_hallway",
            "peak_interest", "orientation_at_peak", "decel_at_peak",
            "approach_at_peak", "dwell_at_peak", "interested", "entered",
            "low_confidence",
        ])

    return pd.DataFrame(records)
