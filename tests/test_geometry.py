import numpy as np
import pytest
from src.config import Line, Polygon, ScaleMap
from src.geometry import (
    body_height_at,
    cos_angle,
    facing_normal,
    foot_point,
    head_yaw_asym,
    nearest_point_on_segment,
    point_in_polygon,
    signed_side,
    speed_bh_per_s,
)


def test_entrance_line_sign():
    line = Line(p1=(362, 267), p2=(1117, 498), store_side_sign=1)
    val_inside = line.signed_side((500, 550))
    val_outside = line.signed_side((800, 250))

    # Matches plan §1.1 measured values (+181,787 and -114,013)
    assert val_inside > 0
    assert val_outside < 0
    assert abs(val_inside - 181787.0) < 1.0
    assert abs(val_outside - (-114013.0)) < 1.0


def test_facing_normal_requires_both_shoulders():
    # 17 keypoints with (x, y, conf)
    kpts_valid = np.zeros((17, 3), dtype=np.float32)
    kpts_valid[5] = [200, 100, 0.9]  # L-shoulder
    kpts_valid[6] = [100, 100, 0.9]  # R-shoulder

    n = facing_normal(kpts_valid)
    assert n is not None
    # s = L - R = [100, 0], n = normalize([-0, 100]) = [0, 1] (pointing down)
    assert np.allclose(n, [0.0, 1.0])

    kpts_missing_left = kpts_valid.copy()
    kpts_missing_left[5, 2] = 0.1  # L-shoulder conf 0.1
    assert facing_normal(kpts_missing_left) is None

    kpts_missing_right = kpts_valid.copy()
    kpts_missing_right[6, 2] = 0.4  # R-shoulder conf 0.4 < 0.5
    assert facing_normal(kpts_missing_right) is None


def test_scale_map_monotonic():
    # Scale map fitted from plan §1.1
    # ~ (148 - 89) / (635 - 200) = 59 / 435 ≈ 0.136, b ≈ 62
    scale_map = ScaleMap(a=0.136, b=62.0)
    h_top = body_height_at(200.0, scale_map)
    h_mid = body_height_at(400.0, scale_map)
    h_bot = body_height_at(600.0, scale_map)
    assert h_top < h_mid < h_bot


def test_speed_zero_for_stationary():
    scale_map = ScaleMap(a=0.136, b=62.0)
    speed = speed_bh_per_s((500.0, 300.0), (500.0, 300.0), dt_s=1.0, scale_map=scale_map)
    assert speed == 0.0


def test_foot_point():
    bbox = [100, 100, 200, 300]
    # No kpts -> bbox bottom centre (150, 300)
    assert foot_point(bbox) == (150.0, 300.0)

    # Low conf ankles -> fallback to bbox bottom centre
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[15] = [140, 290, 0.3]
    kpts[16] = [160, 295, 0.4]
    assert foot_point(bbox, kpts) == (150.0, 300.0)

    # High conf ankles -> midpoint of ankles (150, 292.5)
    kpts[15] = [140, 290, 0.8]
    kpts[16] = [160, 295, 0.8]
    assert foot_point(bbox, kpts) == (150.0, 292.5)


def test_head_yaw_asym():
    kpts = np.zeros((17, 3), dtype=np.float32)
    # nose=0, Leye=1, Reye=2, Lear=3, Rear=4
    kpts[1] = [100, 50, 0.9]
    kpts[2] = [90, 50, 0.1]
    kpts[3] = [110, 50, 0.9]
    kpts[4] = [80, 50, 0.1]
    asym = head_yaw_asym(kpts)
    assert asym is not None
    assert asym > 0.5  # Turned left -> high L conf, low R conf

    # All below floor -> None
    kpts[:, 2] = 0.1
    assert head_yaw_asym(kpts) is None


def test_nearest_point_on_segment():
    a = (0.0, 0.0)
    b = (10.0, 0.0)
    assert nearest_point_on_segment((5.0, 5.0), a, b) == (5.0, 0.0)
    assert nearest_point_on_segment((-2.0, 0.0), a, b) == (0.0, 0.0)
    assert nearest_point_on_segment((15.0, 0.0), a, b) == (10.0, 0.0)


def test_point_in_polygon():
    poly = Polygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)])
    pts = np.array([[5, 5], [15, 15], [0, 0]])
    mask = point_in_polygon(pts, poly)
    assert mask[0] == True
    assert mask[1] == False


def test_cos_angle():
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    assert abs(cos_angle(u, v) - 0.0) < 1e-6
    assert abs(cos_angle(u, u) - 1.0) < 1e-6
    assert abs(cos_angle(u, -u) - (-1.0)) < 1e-6
