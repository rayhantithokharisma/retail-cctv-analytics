"""Task 2 — Per-Shelf Customer Interest.

Simple, auditable rule (per person, per frame):
  a frame is a *candidate* for shelf S when
    1. the foot point is within `reach_dist_bh` body-heights of S's annotation rect
       (body height = the person's own bbox height that frame, so the threshold is
       perspective-invariant across the floor plane), AND
    2. the torso facing normal points at the rect centre within
       `facing_cos_floor_deg` (visual attention toward the shelf — this is what
       decides between two shelves when a person stands between them), AND
    3. the person is slower than `engage_speed_bh_s` (browsing, not walking past).

Each frame gets at most one shelf: the nearest candidate by distance.
A HysteresisStateMachine (t_on / t_off) turns the per-frame boolean into episodes:
continuous attention >= t_on_s = ONE interest event; the episode ends after t_off_s
without candidates, and a later return counts as a NEW event (brief: no double
counting of one continuous interaction, but repeated visits do count).
"""
from collections import Counter
import numpy as np
import pandas as pd
from src.config import ShelfConfig, Task2Config
from src.events import HysteresisStateMachine
from src.geometry import cos_angle, point_in_polygon
from src.identity import Identity


def rect_distance(px: float, py: float, rect: tuple[float, float, float, float]) -> float:
    """Euclidean distance from point to axis-aligned rect (0 when inside)."""
    x, y, w, h = rect
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return float(np.hypot(dx, dy))


def candidate_score(
    identity: Identity,
    t_idx: int,
    shelf: ShelfConfig,
    cfg: Task2Config,
) -> tuple[float, float] | None:
    """Returns (dist_bh, facing_cos) when the frame is a candidate for `shelf`,
    else None. See module docstring for the three gates."""
    foot = identity.foot_xy[t_idx]
    if np.isnan(foot).any():
        return None

    # Scene-artifact exclusions (mirror reflections, cashier counter / staff desk):
    # verified against the footage — people here are not browsing any shelf.
    for zone in cfg.ignore_zones:
        if point_in_polygon(foot.reshape(1, 2), zone)[0]:
            return None

    # body height = own bbox height this frame (fallback: median over the track)
    h = float(identity.heights[t_idx]) if t_idx < len(identity.heights) else np.nan
    if not np.isfinite(h) or h <= 1.0:
        h = float(np.nanmedian(identity.heights)) if len(identity.heights) else 100.0

    min_h = shelf.min_bbox_height if shelf.min_bbox_height is not None else cfg.min_bbox_height
    if h < min_h:
        return None

    # Gate 1+3: proximity (body-height normalised) and browsing speed
    dist_bh = rect_distance(float(foot[0]), float(foot[1]), shelf.rect) / h
    if dist_bh > cfg.reach_dist_bh:
        return None
    v = float(identity.speed_bh_s[t_idx])
    if np.isfinite(v) and v > cfg.engage_speed_bh_s:
        return None

    # Gate 2: torso must face the shelf (decides which shelf when between two)
    fn = identity.facing_normal[t_idx]
    if fn is None or np.isnan(fn).any():
        return None
    cx, cy = shelf.centroid
    to_shelf = np.array([cx - foot[0], cy - foot[1]], dtype=np.float64)
    floor_deg = shelf.facing_cos_floor_deg if shelf.facing_cos_floor_deg is not None else cfg.facing_cos_floor_deg
    facing_cos = cos_angle(fn, to_shelf)
    if facing_cos < np.cos(np.radians(floor_deg)):
        return None

    return (dist_bh, facing_cos)


def assign_shelf_per_frame(identity: Identity, cfg: Task2Config) -> np.ndarray:
    """Per frame: the single shelf the person interacts with (nearest candidate),
    or None. This is the answer to 'which shelf is this person interacting with'."""
    n = len(identity.frames)
    assignment = np.empty(n, dtype=object)
    for t_idx in range(n):
        best_name, best_dist = None, np.inf
        for shelf in cfg.shelves:
            res = candidate_score(identity, t_idx, shelf, cfg)
            if res is not None and res[0] < best_dist:
                best_name, best_dist = shelf.name, res[0]
        assignment[t_idx] = best_name
    return assignment


def mode_filter_assignment(raw_assignment: np.ndarray, window_frames: int = 30) -> np.ndarray:
    """1s rolling majority vote over the raw per-frame assignment (None-safe).
    Kills A/B flicker when a person drifts between two fixtures, so one person
    has at most one active shelf at a time."""
    n = len(raw_assignment)
    if n == 0:
        return np.array([], dtype=object)
    filtered = np.empty(n, dtype=object)
    half = window_frames // 2
    for i in range(n):
        window = [x for x in raw_assignment[max(0, i - half):min(n, i + half + 1)] if x is not None]
        filtered[i] = Counter(window).most_common(1)[0][0] if window else None
    return filtered


def shelf_interest_events(
    identities: list[Identity],
    cfg: Task2Config,
    fps: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs Task 2.
    Returns:
      df_events: one row per interest episode (identity, shelf, start/end, duration,
                 median foot distance) — the audit trail.
      df_counts: official deliverable — interest-event count per shelf + total row.
    """
    sm = HysteresisStateMachine(t_on_s=cfg.event.t_on_s, t_off_s=cfg.event.t_off_s, fps=fps)
    event_records = []

    for ident in identities:
        if len(ident.frames) == 0:
            continue
        raw_assignment = assign_shelf_per_frame(ident, cfg)
        assignment = mode_filter_assignment(raw_assignment, window_frames=int(round(fps)))
        for shelf in cfg.shelves:
            cond = assignment == shelf.name
            episodes = sm.run(cond, ident.t_s)
            for ep_start, ep_end in episodes:
                mask = (ident.t_s >= ep_start) & (ident.t_s <= ep_end)
                dists = np.array([
                    rect_distance(float(ident.foot_xy[i][0]), float(ident.foot_xy[i][1]), shelf.rect)
                    / max(float(ident.heights[i]), 1.0)
                    for i in np.where(mask)[0]
                ]) if np.any(mask) else np.array([np.nan])
                event_records.append({
                    "identity_id": ident.identity_id,
                    "shelf_name": shelf.name,
                    "start_t": float(ep_start),
                    "end_t": float(ep_end),
                    "duration_s": float(ep_end - ep_start),
                    "median_dist_bh": float(np.nanmedian(dists)),
                })

    df_events = pd.DataFrame(event_records) if event_records else pd.DataFrame(
        columns=["identity_id", "shelf_name", "start_t", "end_t", "duration_s", "median_dist_bh"]
    )

    count_rows = []
    total = 0
    for shelf in cfg.shelves:
        n_events = int((df_events["shelf_name"] == shelf.name).sum()) if not df_events.empty else 0
        total += n_events
        count_rows.append({"shelf": shelf.name, "interest_events": n_events})
    count_rows.append({"shelf": "total", "interest_events": total})
    df_counts = pd.DataFrame(count_rows)

    return df_events, df_counts


def shelf_summary(df_events: pd.DataFrame, cfg: Task2Config) -> pd.DataFrame:
    """Per-shelf summary: event count, unique visitors, total/avg dwell seconds."""
    rows = []
    for shelf in cfg.shelves:
        s_df = df_events[df_events["shelf_name"] == shelf.name] if not df_events.empty else pd.DataFrame()
        n = len(s_df)
        rows.append({
            "shelf_name": shelf.name,
            "total_engagements": n,
            "unique_visitors": int(s_df["identity_id"].nunique()) if n else 0,
            "total_dwell_s": float(s_df["duration_s"].sum()) if n else 0.0,
            "avg_dwell_s": float(s_df["duration_s"].mean()) if n else 0.0,
        })
    return pd.DataFrame(rows)
