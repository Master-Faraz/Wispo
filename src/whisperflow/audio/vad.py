"""Decide whether the latest microphone frame contains speech.

whisper.cpp's stream example uses a different check: it waits until the last
part of a long buffer is quieter than the rest, then transcribes the whole
buffer at once. That only produces text after the person pauses.

Stage 1 needs text while the person is still talking, so this detector marks
each short frame as speech or silence. A high-pass copy is used for the
loudness measurement only. The samples stored for whisper stay untouched,
because the model was trained on ordinary microphone audio, not filtered audio.
"""

from __future__ import annotations

import math

import numpy as np


class EnergyVad:
    def __init__(
        self,
        sample_rate: int,
        frame_ms: int,
        rms_threshold: float,
        noise_multiplier: float,
        hangover_ms: int,
        highpass_hz: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.noise_multiplier = noise_multiplier
        self.highpass_hz = highpass_hz
        self._hangover_frames = max(1, int(math.ceil(hangover_ms / frame_ms)))
        self._noise = rms_threshold
        self._speech = False
        self._hangover_left = 0
        self._filter_y = 0.0
        self._filter_prev = 0.0
        self._filter_ready = False

    @property
    def speech(self) -> bool:
        return self._speech

    def reset(self) -> None:
        self._speech = False
        self._hangover_left = 0
        self._filter_ready = False

    def process(self, samples: np.ndarray) -> bool:
        frame = np.asarray(samples, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return self._speech
        loud = self._is_speech_frame(frame)
        if loud:
            self._speech = True
            self._hangover_left = self._hangover_frames
            return True
        # Update the noise floor only from frames that are not speech, so a
        # long sentence does not teach the detector that voice is the room tone.
        rms = self._rms(self._highpass(frame))
        self._noise = (0.95 * self._noise) + (0.05 * rms)
        if not self._speech:
            return False
        # hangover_left is how many quiet frames still count as the same
        # utterance. The frame that uses up the last one is still speech.
        if self._hangover_left <= 0:
            self._speech = False
            return False
        self._hangover_left -= 1
        return True

    def _is_speech_frame(self, frame: np.ndarray) -> bool:
        rms = self._rms(self._highpass(frame))
        floor = max(self.rms_threshold, self._noise * self.noise_multiplier)
        return rms > floor

    def _rms(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(frame))))

    def _highpass(self, frame: np.ndarray) -> np.ndarray:
        # Same one-pole high-pass as whisper.cpp examples/common.cpp. It knocks
        # down mains hum and handling noise before the energy comparison.
        if self.highpass_hz <= 0:
            return frame
        rc = 1.0 / (2.0 * math.pi * self.highpass_hz)
        dt = 1.0 / self.sample_rate
        alpha = dt / (rc + dt)
        out = np.empty_like(frame)
        if not self._filter_ready:
            self._filter_y = float(frame[0])
            self._filter_prev = float(frame[0])
            self._filter_ready = True
        y = self._filter_y
        prev = self._filter_prev
        for i, sample in enumerate(frame):
            value = float(sample)
            y = alpha * (y + value - prev)
            prev = value
            out[i] = y
        self._filter_y = y
        self._filter_prev = prev
        return out
