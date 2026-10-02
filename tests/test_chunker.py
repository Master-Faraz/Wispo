import numpy as np

from whisperflow.asr.chunker import SlidingChunker


def test_window_is_emitted_once_a_chunk_has_arrived():
    chunker = SlidingChunker(sample_rate=16000, chunk_ms=1000, length_ms=2000)
    assert chunker.push(np.ones(15999, dtype=np.float32)) is None
    window = chunker.push(np.ones(1, dtype=np.float32))
    assert window is not None
    assert window.size == 16000


def test_window_keeps_only_the_newest_length():
    chunker = SlidingChunker(sample_rate=1000, chunk_ms=1000, length_ms=1500)
    first = chunker.push(np.zeros(1000, dtype=np.float32))
    assert first is not None and first.size == 1000
    second = chunker.push(np.ones(1000, dtype=np.float32))
    assert second is not None
    assert second.size == 1500
    # The oldest 500 samples of the first window were dropped.
    assert np.all(second[:500] == 0)
    assert np.all(second[500:] == 1)


def test_flush_returns_audio_shorter_than_one_chunk():
    chunker = SlidingChunker(sample_rate=1000, chunk_ms=1000, length_ms=1000)
    assert chunker.push(np.ones(400, dtype=np.float32)) is None
    leftover = chunker.flush()
    assert leftover is not None
    assert leftover.size == 400
