"""Read the Linux microphone as 16 kHz mono float samples.

whisper.cpp expects that exact shape: 16 kHz, one channel, float32 in
[-1, 1]. ALSA is used directly because libasound is already on the machine
and the capture path should not grow a PortAudio dependency.

One read() call returns one frame. The engine feeds that frame to the VAD
and, when the VAD says speech, into the chunker.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import threading
from ctypes import c_char_p, c_int, c_uint, c_ulong, c_void_p

import numpy as np

# Values from /usr/include/alsa/pcm.h. They are stable ABI, not Python constants.
_STREAM_CAPTURE = 1
_FORMAT_S16_LE = 2
_ACCESS_RW_INTERLEAVED = 3

_alsa = ctypes.CDLL("libasound.so.2")
_alsa.snd_pcm_open.argtypes = [ctypes.POINTER(c_void_p), c_char_p, c_int, c_int]
_alsa.snd_pcm_open.restype = c_int
_alsa.snd_pcm_set_params.argtypes = [c_void_p, c_int, c_int, c_uint, c_uint, c_int, c_uint]
_alsa.snd_pcm_set_params.restype = c_int
_alsa.snd_pcm_readi.argtypes = [c_void_p, c_void_p, c_ulong]
_alsa.snd_pcm_readi.restype = ctypes.c_long
_alsa.snd_pcm_recover.argtypes = [c_void_p, c_int, c_int]
_alsa.snd_pcm_recover.restype = c_int
_alsa.snd_pcm_drop.argtypes = [c_void_p]
_alsa.snd_pcm_drop.restype = c_int
_alsa.snd_pcm_close.argtypes = [c_void_p]
_alsa.snd_pcm_close.restype = c_int
_alsa.snd_strerror.argtypes = [c_int]
_alsa.snd_strerror.restype = c_char_p


def _alsa_error(code: int, action: str) -> RuntimeError:
    message = _alsa.snd_strerror(code)
    text = message.decode("utf-8", errors="replace") if message else str(code)
    return RuntimeError(f"{action}: {text}")


class AudioCapture:
    def __init__(
        self,
        device: str,
        sample_rate: int,
        frame_ms: int,
        pipewire_target: str | None = None,
        description: str | None = None,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("whisper.cpp requires 16000 Hz audio")
        self.device = device
        self.pipewire_target = pipewire_target
        self.description = description or device
        self.sample_rate = sample_rate
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        self._pcm = c_void_p()
        self._open = False
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=150)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: Exception | None = None

    def start(self) -> None:
        if self._open:
            return
        # The PipeWire ALSA plugin reads this when the device is opened and
        # connects the stream to that source.
        if self.pipewire_target:
            os.environ["PIPEWIRE_PROPS"] = json.dumps({"target.object": self.pipewire_target})
        handle = c_void_p()
        rc = _alsa.snd_pcm_open(ctypes.byref(handle), self.device.encode(), _STREAM_CAPTURE, 0)
        if rc < 0:
            raise _alsa_error(rc, f"opening microphone {self.description!r}")
        # soft_resample=1 lets PipeWire/ALSA give us 16 kHz even when hardware runs at another rate.
        # Use 1,500,000 µs (1.5 seconds) buffer to ensure hardware never overflows during inference.
        latency_us = 1_500_000
        rc = _alsa.snd_pcm_set_params(
            handle,
            _FORMAT_S16_LE,
            _ACCESS_RW_INTERLEAVED,
            1,
            self.sample_rate,
            1,
            latency_us,
        )
        if rc < 0:
            _alsa.snd_pcm_close(handle)
            raise _alsa_error(rc, "configuring microphone")
        self._pcm = handle
        self._open = True
        self._error = None
        self._stop_event.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="wispo-capture",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        raw = np.empty(self.frame_samples, dtype=np.int16)
        while not self._stop_event.is_set():
            remaining = self.frame_samples
            offset = 0
            read_failed = False
            while remaining > 0 and not self._stop_event.is_set():
                got = _alsa.snd_pcm_readi(
                    self._pcm,
                    raw[offset:].ctypes.data_as(c_void_p),
                    remaining,
                )
                if got < 0:
                    # If the stop event is set, the PCM handle may already be
                    # dropped by stop(). Do not call snd_pcm_recover on it.
                    if self._stop_event.is_set():
                        break
                    recovered = _alsa.snd_pcm_recover(self._pcm, int(got), 1)
                    if recovered < 0:
                        self._error = _alsa_error(recovered, "reading microphone")
                        self._open = False
                        read_failed = True
                        break
                    continue
                offset += int(got)
                remaining -= int(got)
            if read_failed or self._stop_event.is_set():
                break
            frame = (raw.astype(np.float32) / 32768.0).copy()
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass

    def read(self, timeout: float = 0.5) -> np.ndarray:
        if self._error is not None:
            err = self._error
            self._error = None
            self.stop()
            raise err
        if not self._open:
            raise RuntimeError("microphone is not started")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if not self._open:
                raise RuntimeError("microphone is not started")
            return np.zeros(self.frame_samples, dtype=np.float32)

    def drain(self) -> list[np.ndarray]:
        """Drain and return all currently buffered frames without waiting."""
        frames: list[np.ndarray] = []
        while True:
            try:
                frames.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return frames

    def stop(self) -> None:
        self._stop_event.set()
        # Drop the stream BEFORE joining the thread. snd_pcm_readi() is a
        # blocking kernel call; the capture thread may be blocked inside it
        # for up to the configured 1.5s hardware latency. snd_pcm_drop()
        # interrupts it immediately so the thread exits cleanly.
        if self._pcm.value is not None:
            try:
                _alsa.snd_pcm_drop(self._pcm)
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._pcm.value is not None:
            try:
                _alsa.snd_pcm_close(self._pcm)
            except Exception:
                pass
        self._pcm = c_void_p()
        self._open = False

    def __enter__(self) -> AudioCapture:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
