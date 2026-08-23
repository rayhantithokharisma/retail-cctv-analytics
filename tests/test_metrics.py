import numpy as np
import pytest
from src.config import load_entrance_config, load_interior_config
from src.identity import Identity
from src.tasks.task1_interest import decide_entered, decide_interested, is_candidate, run_task1
from src.tasks.task2_interaction import group_dwell_episodes
from src.tasks.task2_shelf import assign_shelf_per_frame, rect_distance, shelf_interest_events
from src.tasks.task3_staff import detect_staff


@pytest.fixture
def entrance_cfg():
    return load_entrance_config("configs/entrance.yaml")


@pytest.fixture
def interior_cfg():
    return load_interior_config("configs/interior.yaml")


def test_decide_interested_threshold_and_duration(entrance_cfg):
    # interest_min_duration_s = 0.8s (24 frames @ 30fps)
    # 15 frames above 0.60 @ 30fps = 0.50s (< 0.8s required) -> False
    I1 = np.array([0.60] * 15 + [0.20] * 30)
    assert not decide_interested(I1, entrance_cfg.task1, fps=30.0)

    # 35 frames above 0.60 @ 30fps = 1.17s (>= 0.8s required) -> True
    I2 = np.array([0.60] * 35 + [0.20] * 30)
    assert decide_interested(I2, entrance_cfg.task1, fps=30.0)


def test_decide_entered_deadzone_and_penetration(entrance_cfg):
    # Construct synthetic track crossing entrance line
    # Line is from (362, 267) to (1117, 498)
    # At Q_x = 700: line y is ~370.4. Q_y > 370.4 is store side (>0).
    # Path starts in hallway at y=250 (outside deadzone) and walks into store to y=550 (deep inside)
    n = 120
    frames = np.arange(n)
    t_s = frames / 30.0
    foot_xy = np.zeros((n, 2))
    foot_xy[:, 0] = 700.0
    foot_xy[:, 1] = np.concatenate([np.linspace(250.0, 360.0, 30), np.linspace(480.0, 550.0, 90)])

    kpts_raw = np.ones((n, 17, 3), dtype=np.float32)
    ident = Identity(
        identity_id=1,
        raw_track_ids=[1],
        frames=frames,
        t_s=t_s,
        foot_xy=foot_xy,
        heights=np.full(n, 100.0),
        facing_normal=np.full((n, 2), np.nan),
        head_yaw=np.full(n, np.nan),
        speed_bh_s=np.full(n, 0.5),
        kpts_raw=kpts_raw,
    )

    entered = decide_entered(ident, entrance_cfg.task1, fps=30.0)
    assert entered is True


def test_detect_staff_in_counter_zone(entrance_cfg):
    # Create identity standing in counter zone for 20s (600 frames)
    # Counter zone in entrance: [(480, 420), (600, 420), (600, 520), (480, 520)]
    n = 600
    frames = np.arange(n)
    t_s = frames / 30.0
    foot_xy = np.full((n, 2), [100.0, 600.0])  # inside entrance staff_zone

    kpts = np.ones((n, 17, 3), dtype=np.float32)
    staff_ident = Identity(
        identity_id=99,
        raw_track_ids=[99],
        frames=frames,
        t_s=t_s,
        foot_xy=foot_xy,
        heights=np.full(n, 150.0),
        facing_normal=np.full((n, 2), np.nan),
        head_yaw=np.full(n, np.nan),
        speed_bh_s=np.zeros(n),
        kpts_raw=kpts,
    )

    staff_ids = detect_staff([staff_ident], entrance_cfg.staff, fps=30.0)
    assert 99 in staff_ids


def test_group_dwell_clustering(interior_cfg):
    # Two people stationary close to each other for 4 seconds (120 frames)
    n = 120
    frames = np.arange(n)
    t_s = frames / 30.0

    foot1 = np.full((n, 2), [500.0, 400.0])
    foot2 = np.full((n, 2), [550.0, 400.0])  # 50px apart with bh=100 -> 0.5 bh (< 1.5 bh)

    kpts = np.ones((n, 17, 3), dtype=np.float32)
    id1 = Identity(
        identity_id=1,
        raw_track_ids=[1],
        frames=frames,
        t_s=t_s,
        foot_xy=foot1,
        heights=np.full(n, 100.0),
        facing_normal=np.full((n, 2), np.nan),
        head_yaw=np.full(n, np.nan),
        speed_bh_s=np.full(n, 0.1),
        kpts_raw=kpts,
    )
    id2 = Identity(
        identity_id=2,
        raw_track_ids=[2],
        frames=frames,
        t_s=t_s,
        foot_xy=foot2,
        heights=np.full(n, 100.0),
        facing_normal=np.full((n, 2), np.nan),
        head_yaw=np.full(n, np.nan),
        speed_bh_s=np.full(n, 0.1),
        kpts_raw=kpts,
    )

    episodes = group_dwell_episodes([id1, id2], interior_cfg.task2, fps=30.0)
    assert len(episodes) == 1
    assert episodes[0]["duration_s"] >= 3.0


def _make_browser(n, foot, facing, height=150.0, speed=0.1):
    """Synthetic identity standing at `foot` facing direction `facing`."""
    frames = np.arange(n)
    return Identity(
        identity_id=1,
        raw_track_ids=[1],
        frames=frames,
        t_s=frames / 30.0,
        foot_xy=np.tile(np.asarray(foot, dtype=float), (n, 1)),
        heights=np.full(n, height),
        facing_normal=np.tile(np.asarray(facing, dtype=float), (n, 1)),
        head_yaw=np.full(n, np.nan),
        speed_bh_s=np.full(n, speed),
        kpts_raw=np.ones((n, 17, 3), dtype=np.float32),
    )


def test_rect_distance():
    rect = (100.0, 100.0, 50.0, 40.0)  # x, y, w, h
    assert rect_distance(125.0, 120.0, rect) == 0.0  # inside
    assert abs(rect_distance(160.0, 120.0, rect) - 10.0) < 1e-6  # right of it
    assert abs(rect_distance(125.0, 160.0, rect) - 20.0) < 1e-6  # below it
    assert abs(rect_distance(90.0, 90.0, rect) - np.hypot(10, 10)) < 1e-6  # corner


def test_shelf_assignment_proximity_and_facing(interior_cfg):
    # shelf-b rect from VIA: x=540, y=443, w=101, h=51 -> centroid ~(590, 468)
    # person 0.7 BH below-left of it, facing up-right toward the centroid
    shelf_b = next(s for s in interior_cfg.task2.shelves if s.name == "shelf-b")
    foot = (470.0, 570.0)
    to_centroid = np.array([shelf_b.centroid[0] - foot[0], shelf_b.centroid[1] - foot[1]])
    facing = to_centroid / np.linalg.norm(to_centroid)

    ident = _make_browser(90, foot, facing)
    assignment = assign_shelf_per_frame(ident, interior_cfg.task2)
    assert all(a == "shelf-b" for a in assignment)

    # same spot but facing AWAY from the shelf -> no assignment (facing gate)
    ident_away = _make_browser(90, foot, -facing)
    assignment_away = assign_shelf_per_frame(ident_away, interior_cfg.task2)
    assert all(a is None for a in assignment_away)

    # facing the shelf but too far away -> no assignment (proximity gate)
    ident_far = _make_browser(90, (100.0, 700.0), facing)
    assignment_far = assign_shelf_per_frame(ident_far, interior_cfg.task2)
    assert all(a is None for a in assignment_far)


def test_shelf_interest_events_count_and_revisit(interior_cfg):
    shelf_b = next(s for s in interior_cfg.task2.shelves if s.name == "shelf-b")
    foot = np.array([470.0, 570.0])
    to_centroid = np.array([shelf_b.centroid[0] - foot[0], shelf_b.centroid[1] - foot[1]])
    facing = to_centroid / np.linalg.norm(to_centroid)

    # browse 3s (90f), walk away fast for 6s (180f, far + fast), come back 3s (90f)
    n1, n2, n3 = 90, 180, 90
    n = n1 + n2 + n3
    frames = np.arange(n)
    foot_xy = np.concatenate([
        np.tile(foot, (n1, 1)),
        np.tile(np.array([100.0, 700.0]), (n2, 1)),
        np.tile(foot, (n3, 1)),
    ])
    ident = Identity(
        identity_id=7,
        raw_track_ids=[7],
        frames=frames,
        t_s=frames / 30.0,
        foot_xy=foot_xy,
        heights=np.full(n, 150.0),
        facing_normal=np.tile(facing, (n, 1)),
        head_yaw=np.full(n, np.nan),
        speed_bh_s=np.concatenate([np.full(n1, 0.1), np.full(n2, 1.0), np.full(n3, 0.1)]),
        kpts_raw=np.ones((n, 17, 3), dtype=np.float32),
    )

    df_events, df_counts = shelf_interest_events([ident], interior_cfg.task2, fps=30.0)
    # two separate visits -> two events for shelf-b (t_off=3.0s < 6s away)
    assert len(df_events) == 2
    assert set(df_events["shelf_name"]) == {"shelf-b"}
    row = df_counts[df_counts["shelf"] == "shelf-b"].iloc[0]
    assert row["interest_events"] == 2
    assert df_counts[df_counts["shelf"] == "total"].iloc[0]["interest_events"] == 2
