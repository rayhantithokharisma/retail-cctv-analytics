import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def interpolate_gaps(
    series: np.ndarray,
    frames: np.ndarray,
    max_gap: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolates gaps <= max_gap between valid observations.
    Gaps > max_gap stay NaN and valid_mask=False there.
    Returns (filled_series, valid_mask)."""
    series_arr = np.asarray(series, dtype=np.float64)
    frames_arr = np.asarray(frames, dtype=np.int32)
    n = len(series_arr)

    if n == 0:
        return series_arr.copy(), np.zeros(0, dtype=bool)

    is_2d = series_arr.ndim == 2
    if not is_2d:
        series_2d = series_arr[:, np.newaxis]
    else:
        series_2d = series_arr

    num_cols = series_2d.shape[1]
    filled_2d = series_2d.copy()
    valid_mask = ~np.isnan(series_2d).any(axis=1)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) >= 2:
        for k in range(len(valid_indices) - 1):
            idx_start = valid_indices[k]
            idx_end = valid_indices[k + 1]
            gap_frames = int(frames_arr[idx_end] - frames_arr[idx_start])
            gap_steps = idx_end - idx_start

            if 1 < gap_steps and gap_frames <= max_gap:
                # Interpolate intermediate steps
                v_start = filled_2d[idx_start]
                v_end = filled_2d[idx_end]
                for s in range(1, gap_steps):
                    alpha = float(s) / float(gap_steps)
                    filled_2d[idx_start + s] = (1.0 - alpha) * v_start + alpha * v_end
                    valid_mask[idx_start + s] = True

    if not is_2d:
        return filled_2d[:, 0], valid_mask
    return filled_2d, valid_mask


def median_filter_track(
    series: np.ndarray,
    frames: np.ndarray | None = None,
    window_frames: int = 15
) -> np.ndarray:
    """Applies a median filter with window_frames over the series."""
    series_arr = np.asarray(series, dtype=np.float64)
    if len(series_arr) == 0:
        return series_arr.copy()

    if window_frames % 2 == 0:
        window_frames += 1

    is_2d = series_arr.ndim == 2
    if is_2d:
        out = np.zeros_like(series_arr)
        for col in range(series_arr.shape[1]):
            s = pd.Series(series_arr[:, col])
            out[:, col] = s.rolling(window=window_frames, center=True, min_periods=1).median().to_numpy()
        return out
    else:
        s = pd.Series(series_arr)
        return s.rolling(window=window_frames, center=True, min_periods=1).median().to_numpy()


def savgol_filter_track(
    series: np.ndarray,
    frames: np.ndarray | None = None,
    window: int = 21,
    order: int = 2
) -> np.ndarray:
    """Applies Savitzky-Golay filter to smooth series while preserving derivatives."""
    series_arr = np.asarray(series, dtype=np.float64)
    n = len(series_arr)
    if n == 0:
        return series_arr.copy()

    if window % 2 == 0:
        window += 1
    if n < window:
        window = n if n % 2 == 1 else n - 1

    if window <= order or window < 3:
        return series_arr.copy()

    is_2d = series_arr.ndim == 2
    if is_2d:
        out = np.zeros_like(series_arr)
        for col in range(series_arr.shape[1]):
            col_data = series_arr[:, col]
            # Replace NaNs with forward/backward fill for filtering
            s = pd.Series(col_data).interpolate(method="linear").bfill().ffill()
            smoothed = savgol_filter(s.to_numpy(), window_length=window, polyorder=order)
            out[:, col] = np.where(np.isnan(col_data), np.nan, smoothed)
        return out
    else:
        s = pd.Series(series_arr).interpolate(method="linear").bfill().ffill()
        smoothed = savgol_filter(s.to_numpy(), window_length=window, polyorder=order)
        return np.where(np.isnan(series_arr), np.nan, smoothed)
