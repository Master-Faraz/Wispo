"""A short tail of recent microphone samples.

The voice detector decides a frame is speech only after it has heard that
frame. The start of the word is already in the past by then. This buffer
keeps the last `capacity` samples so that audio can be handed to the
transcriber together with the frame that crossed the threshold.
"""

from __future__ import annotations

import numpy as np


class AudioRing:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._samples = np.zeros(0, dtype=np.float32)

    def write(self, samples: np.ndarray) -> None:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size == 0:
            return
        if incoming.size >= self.capacity:
            self._samples = incoming[-self.capacity :].copy()
            return
        self._samples = np.concatenate((self._samples, incoming))
        if self._samples.size > self.capacity:
            self._samples = self._samples[-self.capacity :]

    def snapshot(self) -> np.ndarray:
        return self._samples.copy()

    def clear(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._samples.size)
