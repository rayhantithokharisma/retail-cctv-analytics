from typing import Sequence
import numpy as np
from src.config import Line, Polygon, ScaleMap


def foot_point(
    bbox: Sequence[float] | np.ndarray,
    kpts: np.ndarray | None = None
) -> tuple[float, float]:
    """bbox-bottom-centre; if both ankle kpts conf>=0.5, use their midpoint instead."""
    x1, y1, x2, y2 = bbox[:4]
    bbox_bottom_centre = ((x1 + x2) / 2.0, float(y2))

    if kpts is not None and len(kpts) >= 17:
        l_ankle = kpts[15]
        r_ankle = kpts[16]
        if l_ankle[2] >= 0.5 and r_ankle[2] >= 0.5:
            return (float((l_ankle[0] + r_ankle[0]) / 2.0), float((l_ankle[1] + r_ankle[1]) / 2.0))

    return (float(bbox_bottom_centre[0]), float(bbox_bottom_centre[1]))


def signed_side(line: Line, q: tuple[float, float] | list[float] | np.ndarray) -> float:
    """wraps Line.signed_side"""
    return line.signed_side(q)


def facing_normal(kpts: np.ndarray | None) -> np.ndarray | None:
    """Returns unit vector n, or None if either shoulder conf < 0.5.
    s = L_shoulder.xy - R_shoulder.xy ; n = normalize([-s_y, s_x])."""
    if kpts is None or len(kpts) < 7:
        return None

    l_shoulder = kpts[5]
    r_shoulder = kpts[6]

    if l_shoulder[2] < 0.5 or r_shoulder[2] < 0.5:
        return None

    s = l_shoulder[:2] - r_shoulder[:2]
    s_x, s_y = s[0], s[1]
    n = np.array([-s_y, s_x], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return None
    return n / norm


def head_yaw_asym(kpts: np.ndarray | None) -> float | None:
    """asym = 0.5*((c_Lear-c_Rear)+(c_Leye-c_Reye)), clamped [-1,1]. None if all four confs < 0.15."""
    if kpts is None or len(kpts) < 5:
        return None

    c_leye = kpts[1, 2]
    c_reye = kpts[2, 2]
    c_lear = kpts[3, 2]
    c_rear = kpts[4, 2]

    if c_leye < 0.15 and c_reye < 0.15 and c_lear < 0.15 and c_rear < 0.15:
        return None

    asym = 0.5 * ((c_lear - c_rear) + (c_leye - c_reye))
    return float(np.clip(asym, -1.0, 1.0))


def body_height_at(y: float, scale_map: ScaleMap) -> float:
    return scale_map.expected_height(y)


def speed_bh_per_s(
    p0: tuple[float, float] | np.ndarray,
    p1: tuple[float, float] | np.ndarray,
    dt_s: float,
    scale_map: ScaleMap
) -> float:
    """||p1-p0|| / expected_height(p1.y) / dt_s"""
    if dt_s <= 0:
        return 0.0
    p0_arr = np.asarray(p0, dtype=np.float64)
    p1_arr = np.asarray(p1, dtype=np.float64)
    dist = np.linalg.norm(p1_arr - p0_arr)
    bh = scale_map.expected_height(float(p1_arr[1]))
    return float(dist / bh / dt_s)


def point_in_polygon(points_xy: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Vectorised via polygon.to_path.contains_points(points_xy)."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    return polygon.to_path.contains_points(pts)


def nearest_point_on_segment(
    p: tuple[float, float] | np.ndarray,
    a: tuple[float, float] | np.ndarray,
    b: tuple[float, float] | np.ndarray
) -> tuple[float, float]:
    p_arr = np.asarray(p, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    ab = b_arr - a_arr
    ab_norm_sq = float(np.dot(ab, ab))
    if ab_norm_sq < 1e-9:
        return (float(a_arr[0]), float(a_arr[1]))
    t = float(np.dot(p_arr - a_arr, ab) / ab_norm_sq)
    t = max(0.0, min(1.0, t))
    proj = a_arr + t * ab
    return (float(proj[0]), float(proj[1]))


def cos_angle(u: np.ndarray, v: np.ndarray) -> float:
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    norm_u = np.linalg.norm(u_arr)
    norm_v = np.linalg.norm(v_arr)
    if norm_u < 1e-6 or norm_v < 1e-6:
        return 0.0
    cos_val = np.dot(u_arr, v_arr) / (norm_u * norm_v)
    return float(np.clip(cos_val, -1.0, 1.0))
