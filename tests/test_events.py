import numpy as np
import pytest
from src.events import HysteresisStateMachine


def test_clean_episode():
    # fps=10, t_on=2.0s (20 frames), t_off=3.0s (30 frames)
    fps = 10.0
    sm = HysteresisStateMachine(t_on_s=2.0, t_off_s=3.0, fps=fps)

    # 10 frames False, 40 frames True, 40 frames False
    cond = np.array([False] * 10 + [True] * 40 + [False] * 40)
    t_s = np.arange(len(cond)) / fps

    episodes = sm.run(cond, t_s)
    assert len(episodes) == 1
    start_t, end_t = episodes[0]
    # Starts at index 10 (1.0s), ends at index 49 (4.9s)
    assert abs(start_t - 1.0) < 1e-3
    assert abs(end_t - 4.9) < 1e-3


def test_flicker_in_middle_stays_single_episode():
    fps = 10.0
    sm = HysteresisStateMachine(t_on_s=2.0, t_off_s=3.0, fps=fps)

    # 30 frames True, 10 frames False (1.0s < 3.0s t_off), 30 frames True, 40 frames False
    cond = np.array([True] * 30 + [False] * 10 + [True] * 30 + [False] * 40)
    t_s = np.arange(len(cond)) / fps

    episodes = sm.run(cond, t_s)
    assert len(episodes) == 1
    start_t, end_t = episodes[0]
    assert abs(start_t - 0.0) < 1e-3
    assert abs(end_t - 6.9) < 1e-3


def test_back_to_back_episodes():
    fps = 10.0
    sm = HysteresisStateMachine(t_on_s=2.0, t_off_s=3.0, fps=fps)

    # Ep 1: 30 True, Gap: 40 False (> 3.0s), Ep 2: 30 True, 40 False
    cond = np.array([True] * 30 + [False] * 40 + [True] * 30 + [False] * 40)
    t_s = np.arange(len(cond)) / fps

    episodes = sm.run(cond, t_s)
    assert len(episodes) == 2
    assert abs(episodes[0][0] - 0.0) < 1e-3
    assert abs(episodes[0][1] - 2.9) < 1e-3
    # Ep 2 starts at index 70 (7.0s), ends at index 99 (9.9s)
    assert abs(episodes[1][0] - 7.0) < 1e-3
    assert abs(episodes[1][1] - 9.9) < 1e-3


def test_truncated_by_end():
    fps = 10.0
    sm = HysteresisStateMachine(t_on_s=2.0, t_off_s=3.0, fps=fps)

    # 10 False, 30 True (ends while still True)
    cond = np.array([False] * 10 + [True] * 30)
    t_s = np.arange(len(cond)) / fps

    episodes = sm.run(cond, t_s)
    assert len(episodes) == 1
    assert abs(episodes[0][0] - 1.0) < 1e-3
    assert abs(episodes[0][1] - 3.9) < 1e-3
