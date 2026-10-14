import numpy as np

from wispo.audio.buffer import AudioRing
from wispo.audio.vad import EnergyVad


def test_ring_keeps_the_newest_samples():
    ring = AudioRing(4)
    ring.write(np.array([1, 2, 3], dtype=np.float32))
    ring.write(np.array([4, 5], dtype=np.float32))
    assert ring.snapshot().tolist() == [2, 3, 4, 5]


def test_vad_marks_a_loud_frame_as_speech_and_holds_briefly():
    vad = EnergyVad(
        sample_rate=16000,
        frame_ms=30,
        rms_threshold=0.01,
        noise_multiplier=3.0,
        hangover_ms=60,
        highpass_hz=0,
    )
    frame = int(16000 * 30 / 1000)
    silence = np.zeros(frame, dtype=np.float32)
    tone = (0.4 * np.sin(np.linspace(0, 40, frame))).astype(np.float32)

    assert vad.process(silence) is False
    assert vad.process(tone) is True
    # Two frames of hangover (60 ms / 30 ms) stay open after the voice stops.
    assert vad.process(silence) is True
    assert vad.process(silence) is True
    assert vad.process(silence) is False
