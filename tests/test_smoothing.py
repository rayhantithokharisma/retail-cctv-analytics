import numpy as np
import pytest
from src.smoothing import interpolate_gaps, median_filter_track, savgol_filter_track


def test_median_filter_absorbs_single_frame_spike():
    # Constant baseline speed of 0.8 bh/s with a single-frame spike of 50.0 bh/s at index 15
    n = 31
    speed = np.full(n, 0.8, dtype=np.float64)
    speed[15] = 50.0

    frames = np.arange(n, dtype=np.int32)
    smoothed = median_filter_track(speed, frames, window_frames=15)

    assert smoothed[15] == 0.8
    assert np.allclose(smoothed, 0.8)


def test_interpolate_gaps_8_vs_15_frames():
    # Sequence of frames with an 8-frame gap and a 15-frame gap
    # Indices: 0..5 (valid), 6..12 (NaNs, gap=8 to index 13), 13..18 (valid), 19..32 (NaNs, gap=15 to index 33), 33..35 (valid)
    n = 36
    frames = np.arange(n, dtype=np.int32)
    series = np.full(n, np.nan, dtype=np.float64)

    # 0..5: value 10.0
    series[0:6] = 10.0
    # 13: value 18.0 (gap of 13-5 = 8 frames)
    series[13] = 18.0
    # 14..18: value 18.0
    series[14:19] = 18.0
    # 33: value 30.0 (gap of 33-18 = 15 frames)
    series[33:36] = 30.0

    filled, valid_mask = interpolate_gaps(series, frames, max_gap=10)

    # 8-frame gap (index 6..12) should be interpolated and valid
    for idx in range(6, 13):
        assert valid_mask[idx] == True
        assert not np.isnan(filled[idx])
        assert 10.0 < filled[idx] < 18.0

    # 15-frame gap (index 19..32) should NOT be interpolated and stay invalid
    for idx in range(19, 33):
        assert valid_mask[idx] == False
        assert np.isnan(filled[idx])


def test_savgol_filter_preserves_trend():
    # Linear ramp + noise
    x = np.linspace(0, 10, 51)
    y = 2.0 * x + 1.0
    smoothed = savgol_filter_track(y, window=11, order=2)
    assert np.allclose(smoothed, y, atol=1e-3)
