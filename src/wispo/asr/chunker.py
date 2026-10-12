"""Cut a stream of speech samples into overlapping windows.

Microphone frames are small (about 30 ms). whisper is happier with about a
second or more, and a word that lands on a cut would be heard only as a
fragment. So frames are stored until `chunk_ms` of new audio has arrived,
then the last `length_ms` of audio is returned as one window.

The next window still contains the end of this one. That repeated audio is
the overlap. The transcript accumulator, not this class, removes the
repeated words.
"""

from __future__ import annotations

import numpy as np


class SlidingChunker:
    def __init__(self, sample_rate: int, chunk_ms: int, length_ms: int) -> None:
        if chunk_ms <= 0 or length_ms <= 0:
            raise ValueError("chunk_ms and length_ms must be positive")
        if length_ms < chunk_ms:
            raise ValueError("length_ms must be at least chunk_ms")
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.length_samples = int(sample_rate * length_ms / 1000)
        self._pending = np.zeros(0, dtype=np.float32)
        self._window = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray | None:
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size == 0:
            return None
        self._pending = np.concatenate((self._pending, incoming))
        if self._pending.size < self.chunk_samples:
            return None
        self._window = np.concatenate((self._window, self._pending))
        self._pending = np.zeros(0, dtype=np.float32)
        if self._window.size > self.length_samples:
            self._window = self._window[-self.length_samples :]
        return self._window.copy()

    def flush(self) -> np.ndarray | None:
        """Return whatever speech is left when the person stops talking."""
        if self._pending.size:
            self._window = np.concatenate((self._window, self._pending))
            self._pending = np.zeros(0, dtype=np.float32)
            if self._window.size > self.length_samples:
                self._window = self._window[-self.length_samples :]
        if self._window.size == 0:
            return None
        return self._window.copy()

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._window = np.zeros(0, dtype=np.float32)
