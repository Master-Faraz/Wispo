"""Stage 1 session: microphone to text, with the model kept loaded.

Data flow while the CLI is listening:

    microphone frame (16 kHz mono float)
        -> preroll buffer
        -> energy VAD
        -> if speech: sliding window
        -> whisper.cpp (model already in memory, SYCL or CPU)
        -> transcript accumulator
        -> partial line

When the VAD reports that speech has ended, the whole utterance is sent
through whisper once more. That second call is the Final line. It is more
stable than the last partial because it hears the phrase in one piece
instead of a window that still overlaps the previous guess.

start_recording() and stop_recording() are the only controls. Nothing in
this file knows about a keyboard shortcut. That is Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from whisperflow.asr.accumulator import TranscriptAccumulator
from whisperflow.asr.chunker import SlidingChunker
from whisperflow.asr.device import DeviceInfo, select_device
from whisperflow.asr.whisper_cpp import WhisperCppEngine
from whisperflow.audio.buffer import AudioRing
from whisperflow.audio.capture import AudioCapture
from whisperflow.audio.sources import resolve_mic
from whisperflow.audio.vad import EnergyVad
from whisperflow.config import Config

# whisper's useful context is 30 seconds. A longer final pass would be cut
# by the model anyway, so past that we trust the merged windows.
_MAX_FINAL_SECONDS = 30


@dataclass
class Partial:
    text: str
    committed: str
    partial: str
    audio_seconds: float
    infer_seconds: float


@dataclass
class Final:
    text: str
    audio_seconds: float
    infer_seconds: float


class TranscriptionApp:
    def __init__(self, config: Config, device: DeviceInfo | None = None) -> None:
        self.config = config
        self.device = device or select_device(config.asr.device)
        self.engine = WhisperCppEngine(
            model_path=config.resolve(config.asr.model_path),
            use_gpu=self.device.use_gpu,
            gpu_device=config.asr.gpu_device,
            threads=config.asr.threads,
        )
        mic = resolve_mic(config.audio.device)
        self.capture = AudioCapture(
            device=mic.alsa_device,
            sample_rate=config.audio.sample_rate,
            frame_ms=config.audio.frame_ms,
            pipewire_target=mic.pipewire_target,
            description=mic.description,
        )
        self.vad = EnergyVad(
            sample_rate=config.audio.sample_rate,
            frame_ms=config.audio.frame_ms,
            rms_threshold=config.vad.rms_threshold,
            noise_multiplier=config.vad.noise_multiplier,
            hangover_ms=config.vad.hangover_ms,
            highpass_hz=config.vad.highpass_hz,
        )
        preroll_samples = int(config.audio.sample_rate * config.vad.preroll_ms / 1000)
        self.preroll = AudioRing(max(1, preroll_samples))
        self.chunker = SlidingChunker(
            sample_rate=config.audio.sample_rate,
            chunk_ms=config.streaming.chunk_ms,
            length_ms=max(config.streaming.length_ms, config.streaming.chunk_ms),
        )
        self.accumulator = TranscriptAccumulator()
        self._recording = False
        self._in_speech = False
        self._utterance_frames: list[np.ndarray] = []
        self._saw_window = False
        self.last_audio_level: float = 0.0

    @property
    def _utterance(self) -> np.ndarray:
        if not self._utterance_frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._utterance_frames)

    def load_model(self, warmup: bool = True) -> None:
        self.engine.start()
        if warmup:
            self.engine.warmup()
            if not getattr(self, "_audio_warmed", False):
                try:
                    self.capture.start()
                    self.capture.read(timeout=0.1)
                    self.capture.stop()
                    self._audio_warmed = True
                except Exception:
                    pass

    def start_recording(self) -> None:
        self.load_model(warmup=True)
        self._reset_utterance()
        self.last_audio_level = 0.0
        self.capture.start()
        self._recording = True

    def stop_recording(self) -> str:
        """Stop the microphone and return the current phrase. The model stays loaded."""
        text = ""
        if self._recording:
            leftovers = self.capture.drain()
            if leftovers:
                self._utterance_frames.extend(leftovers)
                tail = np.concatenate(leftovers)
                if self._saw_window or self._in_speech:
                    self.chunker.push(tail)
        if self._recording and self._in_speech:
            final = self._finish_utterance()
            text = final.text if final else ""
        self.capture.stop()
        self.engine.stop()
        self._recording = False
        self._reset_utterance()
        return text

    def read_once(
        self,
        force_speech: bool = False,
        transcribe: bool = True,
    ) -> Partial | Final | None:
        """Pull one microphone frame and maybe return a new partial or a final phrase.

        force_speech is for hold-to-talk. The key already means "this audio
        matters", so the voice detector must not drop the phrase.
        transcribe controls whether intermediate chunk inference is executed.
        """
        if not self._recording:
            raise RuntimeError("start_recording() before read_once()")
        frame = self.capture.read()
        extra_frames = self.capture.drain()
        combined = np.concatenate([frame, *extra_frames]) if extra_frames else frame
        if combined.size:
            self.last_audio_level = float(np.sqrt(np.mean(combined ** 2)))
        if force_speech:
            opening = not self._in_speech
            if opening:
                self._begin_utterance()
            return self._on_speech(combined, opening=opening, transcribe=transcribe)
        frames = [frame, *extra_frames] if extra_frames else [frame]
        result: Partial | Final | None = None
        for f in frames:
            was_speech = self._in_speech
            is_speech = self.vad.process(f)
            if is_speech and not was_speech:
                self._begin_utterance()
            if is_speech:
                partial = self._on_speech(f, opening=not was_speech, transcribe=transcribe)
                if partial is not None:
                    result = partial
            elif was_speech:
                final = self._finish_utterance()
                if final is not None:
                    result = final
            else:
                self.preroll.write(f)
        return result

    def _begin_utterance(self) -> None:
        self._in_speech = True
        self._saw_window = False
        self.accumulator.reset()
        self.chunker.reset()
        self._utterance_frames.clear()

    def _on_speech(
        self,
        frame: np.ndarray,
        opening: bool,
        transcribe: bool = True,
    ) -> Partial | None:
        # The preroll is the audio from just before the detector flipped.
        # It is sent once, ahead of the frame that crossed the threshold.
        if opening:
            pre = self.preroll.snapshot()
            self.preroll.clear()
            if pre.size:
                frame = np.concatenate((pre, frame))
        self._utterance_frames.append(frame)
        if not transcribe:
            return None
        window = self.chunker.push(frame)
        if window is None:
            return None
        self._saw_window = True
        text = self.engine.transcribe_chunk(window)
        if not text:
            return None
        self.accumulator.merge(text)
        return Partial(
            text=self.accumulator.text,
            committed=self.accumulator.committed,
            partial=self.accumulator.partial,
            audio_seconds=self.engine.last_audio_seconds,
            infer_seconds=self.engine.last_infer_seconds,
        )

    def _finish_utterance(self) -> Final | None:
        self._in_speech = False
        leftover = self.chunker.flush()
        audio = (
            np.concatenate(self._utterance_frames)
            if self._utterance_frames
            else np.zeros(0, dtype=np.float32)
        )
        max_samples = _MAX_FINAL_SECONDS * self.config.audio.sample_rate
        if audio.size > max_samples:
            if self._saw_window and leftover is not None and leftover.size:
                text = self.engine.finalize(leftover)
                if text:
                    self.accumulator.merge(text)
                phrase = self.accumulator.finish()
            else:
                final_phrase = self.engine.finalize(audio)
                if final_phrase:
                    self.accumulator.reset()
                    self.accumulator.merge(final_phrase)
                    phrase = self.accumulator.finish()
                else:
                    phrase = self.accumulator.finish()
        elif audio.size >= int(0.3 * self.config.audio.sample_rate):
            final_phrase = self.engine.finalize(audio)
            if final_phrase:
                self.accumulator.reset()
                self.accumulator.merge(final_phrase)
                phrase = self.accumulator.finish()
            else:
                phrase = self.accumulator.finish()
        else:
            phrase = self.accumulator.finish() if self._saw_window else ""
        self._reset_utterance()
        if not phrase:
            return None
        return Final(
            text=phrase,
            audio_seconds=self.engine.last_audio_seconds,
            infer_seconds=self.engine.last_infer_seconds,
        )

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._saw_window = False
        self._utterance_frames.clear()
        self.chunker.reset()
        self.accumulator.reset()
        self.vad.reset()
        self.preroll.clear()
