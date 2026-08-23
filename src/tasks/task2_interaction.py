import numpy as np
import pandas as pd
from src.config import Task2Config
from src.events import HysteresisStateMachine
from src.identity import Identity


def group_dwell_episodes(
    identities: list[Identity],
    cfg: Task2Config,
    fps: float = 30.0
) -> list[dict]:
    """Find group dwell episodes: >=2 people within 1.5 body-heights of each other
    AND speed < cfg.engage_speed_bh_s for a continuous duration >= cfg.event.t_on_s.
    Uses HysteresisStateMachine for temporal stability."""
    if len(identities) < 2:
        return []

    # Map frames to active identities
    all_frames = sorted(set(f for ident in identities for f in ident.frames))
    if not all_frames:
        return []

    frame_to_time = {}
    for ident in identities:
        for f, t in zip(ident.frames, ident.t_s):
            frame_to_time[f] = t

    frame_ident_map = {}
    for ident in identities:
        for idx, f in enumerate(ident.frames):
            frame_ident_map.setdefault(f, []).append((ident, idx))

    # Evaluate pairwise close & slow conditions
    pairs = []
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            id1 = identities[i]
            id2 = identities[j]
            # Common frames
            set2 = set(id2.frames)
            common = [f for f in id1.frames if f in set2]
            if len(common) < int(round(cfg.event.t_on_s * fps)):
                continue

            id1_fdict = {f: idx for idx, f in enumerate(id1.frames)}
            id2_fdict = {f: idx for idx, f in enumerate(id2.frames)}

            n_c = len(common)
            cond = np.zeros(n_c, dtype=bool)
            t_c = np.array([frame_to_time[f] for f in common])
            centroids = np.zeros((n_c, 2), dtype=float)

            for k, f in enumerate(common):
                i1 = id1_fdict[f]
                i2 = id2_fdict[f]

                foot1 = id1.foot_xy[i1]
                foot2 = id2.foot_xy[i2]
                med_h = (float(id1.heights[i1]) + float(id2.heights[i2])) / 2.0
                med_h = max(med_h, 1.0)

                dist_bh = np.linalg.norm(foot1 - foot2) / med_h
                s1 = id1.speed_bh_s[i1]
                s2 = id2.speed_bh_s[i2]

                s1_slow = np.isnan(s1) or s1 <= cfg.engage_speed_bh_s
                s2_slow = np.isnan(s2) or s2 <= cfg.engage_speed_bh_s

                if dist_bh <= 1.5 and s1_slow and s2_slow:
                    cond[k] = True
                    centroids[k] = (foot1 + foot2) / 2.0

            sm = HysteresisStateMachine(t_on_s=cfg.event.t_on_s, t_off_s=cfg.event.t_off_s, fps=fps)
            episodes = sm.run(cond, t_c)
            for ep_s, ep_e in episodes:
                dur = ep_e - ep_s
                if dur >= cfg.event.t_on_s:
                    mask = (t_c >= ep_s) & (t_c <= ep_e) & cond
                    c_xy = np.mean(centroids[mask], axis=0) if np.any(mask) else np.array([0.0, 0.0])
                    pairs.append({
                        "member_ids": [id1.identity_id, id2.identity_id],
                        "start_t": ep_s,
                        "end_t": ep_e,
                        "duration_s": dur,
                        "centroid_x": float(c_xy[0]),
                        "centroid_y": float(c_xy[1]),
                    })

    # Group overlapping pair episodes into multi-member clusters
    group_records = []
    for idx, p in enumerate(pairs):
        group_records.append({
            "group_id": idx + 1,
            "member_ids": p["member_ids"],
            "start_t": p["start_t"],
            "end_t": p["end_t"],
            "duration_s": p["duration_s"],
            "centroid_x": p["centroid_x"],
            "centroid_y": p["centroid_y"],
        })

    return group_records


def run_task2_group_dwell(
    identities: list[Identity],
    cfg: Task2Config,
    fps: float = 30.0
) -> pd.DataFrame:
    """Group dwell episodes as a DataFrame (auxiliary output, not part of the
    official Task 2 per-shelf interest deliverable)."""
    group_dwells = group_dwell_episodes(identities, cfg, fps=fps)
    return pd.DataFrame(group_dwells) if group_dwells else pd.DataFrame(
        columns=["group_id", "member_ids", "start_t", "end_t", "duration_s", "centroid_x", "centroid_y"]
    )
