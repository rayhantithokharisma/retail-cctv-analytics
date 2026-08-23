import numpy as np
import pytest
from src.config import ScaleMap, StitchConfig
from src.stitching import Tracklet, gate_pair, pair_cost, stitch


@pytest.fixture
def base_stitch_config():
    return StitchConfig(
        max_gap_frames=60,  # 2.0s @ 30fps
        motion_gate_bh=2.5,
        scale_gate_frac=0.40,
        direction_gate_deg=60.0,
        cost_max=0.45,
        w_appearance=0.55,
        w_motion=0.30,
        w_scale=0.15,
    )


@pytest.fixture
def scale_map():
    return ScaleMap(a=0.136, b=62.0)


def create_tracklet(track_id, start_frame, num_frames, start_pos, velocity, height=100.0, embed=None):
    frames = np.arange(start_frame, start_frame + num_frames, dtype=np.int32)
    foot_xy = np.zeros((num_frames, 2), dtype=np.float32)
    for i in range(num_frames):
        foot_xy[i] = [start_pos[0] + velocity[0] * i, start_pos[1] + velocity[1] * i]
    heights = np.full(num_frames, height, dtype=np.float32)
    if embed is not None:
        embed = np.asarray(embed, dtype=np.float32)
        embed = embed / np.linalg.norm(embed)
    return Tracklet(
        raw_track_id=track_id,
        frames=frames,
        foot_xy=foot_xy,
        heights=heights,
        embed=embed,
    )


def test_merge_two_tracklets_close_1_5s(base_stitch_config, scale_map):
    # 1.5s gap @ 30fps = 45 frames
    embed1 = np.random.randn(128).astype(np.float32)
    embed2 = embed1 + np.random.randn(128) * 0.01  # very similar

    t1 = create_tracklet(1, start_frame=0, num_frames=30, start_pos=(100, 300), velocity=(2.0, 0.0), height=100.0, embed=embed1)
    # At frame 29, pos is (100 + 29*2 = 158, 300)
    # Extrapolating 45 frames ahead to frame 75: pos ≈ 158 + 46*2 = 250
    t2 = create_tracklet(2, start_frame=75, num_frames=30, start_pos=(250, 300), velocity=(2.0, 0.0), height=102.0, embed=embed2)

    assert gate_pair(t1, t2, base_stitch_config, scale_map) is True
    id_map = stitch([t1, t2], base_stitch_config, scale_map)
    assert id_map[1] == id_map[2]


def test_no_merge_gap_exceeds_max(base_stitch_config, scale_map):
    # 3s gap = 90 frames > max_gap_frames (60)
    embed1 = np.ones(128, dtype=np.float32)
    embed2 = np.ones(128, dtype=np.float32)

    t1 = create_tracklet(1, start_frame=0, num_frames=30, start_pos=(100, 300), velocity=(2.0, 0.0), height=100.0, embed=embed1)
    t2 = create_tracklet(2, start_frame=120, num_frames=30, start_pos=(340, 300), velocity=(2.0, 0.0), height=100.0, embed=embed2)

    assert gate_pair(t1, t2, base_stitch_config, scale_map) is False
    id_map = stitch([t1, t2], base_stitch_config, scale_map)
    assert id_map[1] != id_map[2]


def test_no_merge_opposite_direction(base_stitch_config, scale_map):
    embed1 = np.ones(128, dtype=np.float32)
    embed2 = np.ones(128, dtype=np.float32)

    # t1 moving right, t2 moving left
    t1 = create_tracklet(1, start_frame=0, num_frames=30, start_pos=(100, 300), velocity=(2.0, 0.0), height=100.0, embed=embed1)
    t2 = create_tracklet(2, start_frame=40, num_frames=30, start_pos=(180, 300), velocity=(-2.0, 0.0), height=100.0, embed=embed2)

    assert gate_pair(t1, t2, base_stitch_config, scale_map) is False
    id_map = stitch([t1, t2], base_stitch_config, scale_map)
    assert id_map[1] != id_map[2]


def test_three_way_fragmentation_merges_all(base_stitch_config, scale_map):
    # A (0..30) -> B (45..75) -> C (90..120)
    embed = np.random.randn(128).astype(np.float32)
    t_a = create_tracklet(1, start_frame=0, num_frames=30, start_pos=(100, 300), velocity=(2.0, 0.0), height=100.0, embed=embed)
    t_b = create_tracklet(2, start_frame=45, num_frames=30, start_pos=(190, 300), velocity=(2.0, 0.0), height=101.0, embed=embed)
    t_c = create_tracklet(3, start_frame=90, num_frames=30, start_pos=(280, 300), velocity=(2.0, 0.0), height=99.0, embed=embed)

    id_map = stitch([t_a, t_b, t_c], base_stitch_config, scale_map)
    assert id_map[1] == id_map[2] == id_map[3]
