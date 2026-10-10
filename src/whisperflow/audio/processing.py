"""Audio processing utilities including AGC and level normalization."""

from __future__ import annotations

import numpy as np


def normalize_audio(
    samples: np.ndarray,
    target_rms: float = 0.08,
    peak_limit: float = 0.90,
    max_gain: float = 3.5,
) -> np.ndarray:
    """Soft-limiting AGC normalizer for 16kHz float32 audio.

    Brings quiet or distant microphone speech to a calibrated level
    for optimal whisper mel-filterbank extraction, while preventing clipping.
    """
    if samples.size == 0:
        return samples
    arr = np.asarray(samples, dtype=np.float32)
    current_rms = float(np.sqrt(np.mean(arr ** 2)))
    # Avoid amplifying pure noise floor or silence
    if current_rms < 1e-4:
        return arr
    gain = min(target_rms / current_rms, max_gain)
    normalized = arr * gain
    peak = float(np.max(np.abs(normalized)))
    if peak > peak_limit:
        normalized = normalized * (peak_limit / peak)
    return np.ascontiguousarray(normalized, dtype=np.float32)
