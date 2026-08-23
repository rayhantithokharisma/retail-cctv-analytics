import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon as ShapelyPolygon
from src.config import StaffConfig, Task3Config
from src.events import HysteresisStateMachine
from src.geometry import cos_angle, point_in_polygon
from src.identity import Identity


def detect_staff(
    identities: list[Identity],
    staff_cfg: StaffConfig,
    fps: float = 30.0
) -> set[int]:
    """Identify staff members using counter prior, dwell prior, and apron color prior.
    Score = w_apron * P_apron + w_counter * P_counter + w_dwell * P_dwell.
    Returns set of staff identity_ids."""
    staff_ids = set()

    for ident in identities:
        n_frames = len(ident.frames)
        if n_frames == 0:
            continue

        in_counter = point_in_polygon(ident.foot_xy, staff_cfg.staff_zone)
        cumulative_counter_s = float(np.sum(in_counter) / fps)
        p_counter = float(np.clip(cumulative_counter_s / max(staff_cfg.counter_prior_norm_s, 1e-3), 0.0, 1.0))

        duration_s = float(ident.t_s[-1] - ident.t_s[0]) if n_frames > 1 else 0.0
        p_dwell = float(np.clip(duration_s / max(staff_cfg.dwell_prior_norm_s, 1e-3), 0.0, 1.0))

        # P_apron fallback to 0.5 if no color detector run
        p_apron = 0.5

        score = (
            staff_cfg.w_apron * p_apron
            + staff_cfg.w_counter * p_counter
            + staff_cfg.w_dwell * p_dwell
        )

        if score >= staff_cfg.staff_score_threshold:
            staff_ids.add(ident.identity_id)

    return staff_ids


def staff_customer_interactions(
    identities: list[Identity],
    staff_ids: set[int],
    task3_cfg: Task3Config,
    fps: float = 30.0
) -> list[dict]:
    """Find interaction episodes between staff and customer identities.
    For each (staff, cust) pair concurrent in time:
    per-frame distance in bh <= task3_cfg.proximity_bh (0.9 bh)
    AND cos_angle(staff_facing, to_cust) >= cos(task3_cfg.orientation_cos_floor_deg)
    AND cos_angle(cust_facing, to_staff) >= cos(task3_cfg.orientation_cos_floor_deg)
    AND both speeds <= task3_cfg.co_stationary_bh_s.
    HysteresisStateMachine(t_on_s=task3_cfg.event.t_on_s, t_off_s=task3_cfg.event.t_off_s).run(condition).
    Returns list of dicts: staff_id, customer_id, start_t, end_t, duration_s, mean_distance_bh."""
    if not staff_ids:
        return []

    staff_map = {ident.identity_id: ident for ident in identities if ident.identity_id in staff_ids}
    customer_list = [ident for ident in identities if ident.identity_id not in staff_ids]

    cos_floor = np.cos(np.radians(task3_cfg.orientation_cos_floor_deg))
    sm = HysteresisStateMachine(t_on_s=task3_cfg.event.t_on_s, t_off_s=task3_cfg.event.t_off_s, fps=fps)
    interactions = []

    for staff_id, s_ident in staff_map.items():
        s_frames_dict = {f: i for i, f in enumerate(s_ident.frames)}

        for c_ident in customer_list:
            # Find overlapping frames
            common_frames = [f for f in c_ident.frames if f in s_frames_dict]
            if len(common_frames) < int(round(task3_cfg.event.t_on_s * fps)):
                continue

            c_indices = [np.where(c_ident.frames == f)[0][0] for f in common_frames]
            s_indices = [s_frames_dict[f] for f in common_frames]

            n_common = len(common_frames)
            cond = np.zeros(n_common, dtype=bool)
            dist_bh_arr = np.zeros(n_common, dtype=float)
            t_common = np.array([c_ident.t_s[ci] for ci in c_indices])

            for k in range(n_common):
                ci = c_indices[k]
                si = s_indices[k]

                c_foot = c_ident.foot_xy[ci]
                s_foot = s_ident.foot_xy[si]
                med_h = (float(c_ident.heights[ci]) + float(s_ident.heights[si])) / 2.0
                med_h = max(med_h, 1.0)

                dist_px = np.linalg.norm(c_foot - s_foot)
                dist_bh = dist_px / med_h
                dist_bh_arr[k] = dist_bh

                if dist_bh > task3_cfg.proximity_bh:
                    continue

                # Stationary check
                c_speed = c_ident.speed_bh_s[ci]
                s_speed = s_ident.speed_bh_s[si]
                if not (np.isnan(c_speed) or c_speed <= task3_cfg.co_stationary_bh_s):
                    continue
                if not (np.isnan(s_speed) or s_speed <= task3_cfg.co_stationary_bh_s):
                    continue

                # Facing check
                to_cust = c_foot - s_foot
                norm_sc = np.linalg.norm(to_cust)
                if norm_sc > 1e-6:
                    to_cust = to_cust / norm_sc
                    to_staff = -to_cust
                else:
                    to_cust = np.array([0.0, 1.0])
                    to_staff = np.array([0.0, -1.0])

                s_fn = s_ident.facing_normal[si]
                c_fn = c_ident.facing_normal[ci]

                s_facing_ok = True
                if s_fn is not None and not np.isnan(s_fn).any():
                    s_facing_ok = cos_angle(s_fn, to_cust) >= cos_floor

                c_facing_ok = True
                if c_fn is not None and not np.isnan(c_fn).any():
                    c_facing_ok = cos_angle(c_fn, to_staff) >= cos_floor

                if s_facing_ok and c_facing_ok:
                    cond[k] = True

            episodes = sm.run(cond, t_common)
            for ep_start, ep_end in episodes:
                mask = (t_common >= ep_start) & (t_common <= ep_end)
                dur = float(ep_end - ep_start)
                mean_dist = float(np.mean(dist_bh_arr[mask])) if np.any(mask) else 0.0
                interactions.append({
                    "staff_id": staff_id,
                    "customer_id": c_ident.identity_id,
                    "start_t": ep_start,
                    "end_t": ep_end,
                    "duration_s": dur,
                    "mean_distance_bh": mean_dist,
                })

    return interactions


def run_task3(
    identities: list[Identity],
    staff_cfg: StaffConfig,
    task3_cfg: Task3Config,
    fps: float = 30.0
) -> tuple[set[int], list[dict], pd.DataFrame]:
    """Runs staff detection and staff-customer interaction identification."""
    staff_ids = detect_staff(identities, staff_cfg, fps=fps)
    interactions = staff_customer_interactions(identities, staff_ids, task3_cfg, fps=fps)
    df_interactions = pd.DataFrame(interactions) if interactions else pd.DataFrame(
        columns=["staff_id", "customer_id", "start_t", "end_t", "duration_s", "mean_distance_bh"]
    )
    return staff_ids, interactions, df_interactions


def staff_session_summary(staff_ids: set[int], interactions: list[dict]) -> pd.DataFrame:
    """Brief-required Task 3 table: one row per detected staff instance — INCLUDING
    staff with zero sessions — plus a final row with the average sessions per staff."""
    rows = []
    for sid in sorted(staff_ids):
        n_sessions = sum(1 for inter in interactions if inter["staff_id"] == sid)
        rows.append({"staff_id": sid, "interaction_sessions": n_sessions})
    avg = (sum(r["interaction_sessions"] for r in rows) / len(rows)) if rows else 0.0
    rows.append({"staff_id": "AVERAGE", "interaction_sessions": round(avg, 3)})
    return pd.DataFrame(rows)
