from collections import Counter
import cv2
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from src.config import StaffConfig
from src.geometry import point_in_polygon
from src.identity import Identity


def detect_staff_apron(
    frame_bgr: np.ndarray,
    torso_kpts: np.ndarray,
    cfg: StaffConfig
) -> float:
    """Computes fraction of torso pixels matching apron HSV color bounds."""
    if frame_bgr is None or torso_kpts is None or len(torso_kpts) < 13:
        return 0.0

    # Torso keypoints: Lshoulder(5), Rshoulder(6), Lhip(11), Rhip(12)
    l_sh = torso_kpts[5]
    r_sh = torso_kpts[6]
    l_hip = torso_kpts[11]
    r_hip = torso_kpts[12]

    # Check confidences
    if l_sh[2] < 0.3 or r_sh[2] < 0.3 or l_hip[2] < 0.3 or r_hip[2] < 0.3:
        return 0.0

    x_min = int(max(0, min(l_sh[0], r_sh[0], l_hip[0], r_hip[0])))
    x_max = int(min(frame_bgr.shape[1], max(l_sh[0], r_sh[0], l_hip[0], r_hip[0])))
    y_min = int(max(0, min(l_sh[1], r_sh[1], l_hip[1], r_hip[1])))
    y_max = int(min(frame_bgr.shape[0], max(l_sh[1], r_sh[1], l_hip[1], r_hip[1])))

    if x_max - x_min < 5 or y_max - y_min < 5:
        return 0.0

    torso_crop = frame_bgr[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)

    total_pixels = torso_crop.shape[0] * torso_crop.shape[1]
    matching_pixels = 0

    for color_def in cfg.apron_colours:
        h_lo, h_hi = color_def["h"]
        s_lo, s_hi = color_def["s"]
        v_lo, v_hi = color_def["v"]

        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        matching_pixels += int(np.count_nonzero(mask))

    return float(np.clip(matching_pixels / max(total_pixels, 1), 0.0, 1.0))


def classify_staff(
    identities: list[Identity],
    cfg: StaffConfig,
    video_accessor=None,
    fps: float = 30.0
) -> set[int]:
    """Classifies staff identities by combining apron color, counter zone dwell, and total dwell."""
    staff_ids = set()

    for ident in identities:
        n_frames = len(ident.frames)
        if n_frames == 0:
            continue

        in_counter = point_in_polygon(ident.foot_xy, cfg.staff_zone)
        cumulative_counter_s = float(np.sum(in_counter) / fps)
        p_counter = float(np.clip(cumulative_counter_s / max(cfg.counter_prior_norm_s, 1e-3), 0.0, 1.0))

        duration_s = float(ident.t_s[-1] - ident.t_s[0]) if n_frames > 1 else 0.0
        p_dwell = float(np.clip(duration_s / max(cfg.dwell_prior_norm_s, 1e-3), 0.0, 1.0))

        # Sample apron color if accessor available
        apron_scores = []
        if video_accessor is not None and n_frames > 0:
            sample_step = max(1, n_frames // 10)
            for s_idx in range(0, n_frames, sample_step):
                f_idx = int(ident.frames[s_idx])
                frame = video_accessor.get_frame(f_idx)
                if frame is not None:
                    kpts = ident.kpts_raw[s_idx]
                    sc = detect_staff_apron(frame, kpts, cfg)
                    apron_scores.append(sc)

        p_apron = float(np.mean(apron_scores)) if apron_scores else 0.5

        score = (
            cfg.w_apron * p_apron
            + cfg.w_counter * p_counter
            + cfg.w_dwell * p_dwell
        )

        if score >= cfg.staff_score_threshold:
            staff_ids.add(ident.identity_id)

    return staff_ids
