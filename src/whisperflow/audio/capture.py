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
import os
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

    def start(self) -> None:
        if self._open:
            return
        # The PipeWire ALSA plugin reads this when the device is opened and
        # connects the stream to that source. Without it, "default" follows
        # the session default, which on this PC is a silent Bluetooth loopback.
        if self.pipewire_target:
            os.environ["PIPEWIRE_PROPS"] = json.dumps({"target.object": self.pipewire_target})
        handle = c_void_p()
        rc = _alsa.snd_pcm_open(ctypes.byref(handle), self.device.encode(), _STREAM_CAPTURE, 0)
        if rc < 0:
            raise _alsa_error(rc, f"opening microphone {self.description!r}")
        # soft_resample=1 lets PipeWire/ALSA give us 16 kHz even when the
        # hardware capture device runs at another rate.
        latency_us = 200_000
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

    def read(self) -> np.ndarray:
        if not self._open:
            raise RuntimeError("microphone is not started")
        raw = np.empty(self.frame_samples, dtype=np.int16)
        remaining = self.frame_samples
        offset = 0
        while remaining > 0:
            got = _alsa.snd_pcm_readi(
                self._pcm,
                raw[offset:].ctypes.data_as(c_void_p),
                remaining,
            )
            if got < 0:
                # A desktop session sometimes interrupts capture (suspend, device
                # switch). recover() puts the stream back into a readable state.
                recovered = _alsa.snd_pcm_recover(self._pcm, int(got), 1)
                if recovered < 0:
                    raise _alsa_error(recovered, "reading microphone")
                continue
            offset += int(got)
            remaining -= int(got)
        return raw.astype(np.float32) / 32768.0

    def stop(self) -> None:
        if not self._open:
            return
        _alsa.snd_pcm_drop(self._pcm)
        _alsa.snd_pcm_close(self._pcm)
        self._pcm = c_void_p()
        self._open = False

    def __enter__(self) -> AudioCapture:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
