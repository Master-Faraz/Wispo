import numpy as np
from whisperflow.audio.processing import normalize_audio


def test_normalize_empty():
    empty = np.zeros(0, dtype=np.float32)
    result = normalize_audio(empty)
    assert result.size == 0


def test_normalize_silence():
    silence = np.zeros(1600, dtype=np.float32)
    result = normalize_audio(silence)
    assert np.allclose(result, silence)


def test_normalize_boosts_quiet_speech():
    # Quiet sine wave (RMS ~ 0.014)
    t = np.linspace(0, 1, 16000, endpoint=False)
    quiet = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    normalized = normalize_audio(quiet, target_rms=0.08)
    orig_rms = np.sqrt(np.mean(quiet ** 2))
    new_rms = np.sqrt(np.mean(normalized ** 2))
    assert new_rms > orig_rms
    assert np.max(np.abs(normalized)) <= 0.90


def test_normalize_limits_peak():
    # Loud burst that would clip
    loud = np.ones(100, dtype=np.float32) * 0.8
    normalized = normalize_audio(loud, target_rms=0.5, peak_limit=0.85)
    assert np.max(np.abs(normalized)) <= 0.85
