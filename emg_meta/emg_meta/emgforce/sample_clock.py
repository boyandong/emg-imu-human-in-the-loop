from __future__ import annotations

import numpy as np


class SampleIndexClock:
    """Counts received, unique EMG samples; session indices restart at zero."""

    def __init__(self) -> None:
        self.total = 0
        self.origin = 0

    @property
    def session_index(self) -> int:
        return self.total - self.origin

    def begin_session(self) -> None:
        self.origin = self.total

    def allocate(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if count < 0:
            raise ValueError("count 不能为负")
        global_index = np.arange(self.total, self.total + count, dtype=np.int64)
        self.total += count
        return global_index, global_index - self.origin

