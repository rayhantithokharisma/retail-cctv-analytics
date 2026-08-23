import numpy as np


class HysteresisStateMachine:
    def __init__(self, t_on_s: float, t_off_s: float, fps: float = 30.0):
        self.t_on_s = float(t_on_s)
        self.t_off_s = float(t_off_s)
        self.fps = float(fps)
        self.n_on = max(1, int(round(self.t_on_s * self.fps)))
        self.n_off = max(1, int(round(self.t_off_s * self.fps)))

    def run(self, condition: np.ndarray, t_s: np.ndarray) -> list[tuple[float, float]]:
        """Returns [(start_t, end_t), ...] episodes. IDLE->ACTIVE after t_on_s continuously
        true; ACTIVE->IDLE after t_off_s continuously false (closing the episode at the LAST
        true frame before the gap started, not at the gap's end). Episode still open at the
        final input frame is closed there and included."""
        cond_arr = np.asarray(condition, dtype=bool)
        t_arr = np.asarray(t_s, dtype=float)
        n = len(cond_arr)
        if n == 0:
            return []

        episodes = []
        is_active = False
        c_on = 0
        c_off = 0
        episode_start_idx = 0
        last_true_idx = 0

        for i in range(n):
            if not is_active:
                if cond_arr[i]:
                    if c_on == 0:
                        episode_start_idx = i
                    c_on += 1
                    if c_on >= self.n_on:
                        is_active = True
                        last_true_idx = i
                        c_off = 0
                else:
                    c_on = 0
            else:  # is_active
                if cond_arr[i]:
                    last_true_idx = i
                    c_off = 0
                else:
                    c_off += 1
                    if c_off >= self.n_off:
                        is_active = False
                        episodes.append((float(t_arr[episode_start_idx]), float(t_arr[last_true_idx])))
                        c_on = 0
                        c_off = 0

        if is_active:
            episodes.append((float(t_arr[episode_start_idx]), float(t_arr[last_true_idx])))

        return episodes
