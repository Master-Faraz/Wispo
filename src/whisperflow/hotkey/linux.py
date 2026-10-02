"""Hold-to-talk from the Linux input device.

A terminal only reports that a key was pressed, not that it was released.
Reading /dev/input/event* gives both edges, so key-down can start the
microphone and key-up can stop it. The account needs to be in the `input`
group for those device files to be readable.

The event layout is the kernel's input_event: two 64-bit time fields, then
type, code, and value. value 1 is press, 0 is release, 2 is auto-repeat.
"""

from __future__ import annotations

import os
import select
import struct
import threading
from pathlib import Path

# linux/input-event-codes.h
KEY_CODES = {
    "rightctrl": 97,
    "leftctrl": 29,
    "rightalt": 100,
    "leftalt": 56,
    "space": 57,
}

_EVENT = struct.Struct("llHHi")
_EV_KEY = 0x01


class HoldKey:
    def __init__(self, key: str) -> None:
        name = key.lower()
        if name not in KEY_CODES:
            known = ", ".join(sorted(KEY_CODES))
            raise ValueError(f"unknown key {key!r}; choose one of: {known}")
        self.key = name
        self.code = KEY_CODES[name]
        self.held = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="whisperflow-hotkey", daemon=True)
        self._fds: list[int] = []

    def start(self) -> None:
        self._fds = _open_keyboards(self.code)
        if not self._fds:
            raise RuntimeError(
                f"no keyboard exposing {self.key} was readable. "
                "The account needs to be in the input group."
            )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.held.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for fd in self._fds:
            os.close(fd)
        self._fds = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select(self._fds, [], [], 0.05)
            for fd in readable:
                try:
                    data = os.read(fd, _EVENT.size * 64)
                except OSError:
                    continue
                for offset in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
                    _sec, _usec, event_type, code, value = _EVENT.unpack_from(data, offset)
                    if event_type != _EV_KEY or code != self.code:
                        continue
                    if value == 1:
                        self.held.set()
                    elif value == 0:
                        self.held.clear()


def _open_keyboards(code: int) -> list[int]:
    opened: list[int] = []
    for path in _keyboard_paths(code):
        try:
            opened.append(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
        except OSError:
            continue
    return opened


def _keyboard_paths(code: int) -> list[str]:
    """Event nodes whose key bitmap includes this key, from /proc/bus/input/devices."""
    text = Path("/proc/bus/input/devices").read_text(encoding="utf-8", errors="replace")
    paths: list[str] = []
    blocks = text.split("\n\n")
    for block in blocks:
        key_bits = 0
        handlers = ""
        for line in block.splitlines():
            if line.startswith("B: KEY="):
                key_bits = _parse_key_bits(line.split("=", 1)[1])
            elif line.startswith("H: Handlers="):
                handlers = line.split("=", 1)[1]
        if key_bits & (1 << code) == 0:
            continue
        for token in handlers.split():
            if token.startswith("event"):
                paths.append(f"/dev/input/{token}")
    return paths


def _parse_key_bits(hex_words: str) -> int:
    # The kernel prints the bitmap in unsigned-long chunks, most significant
    # chunk first. The last chunk holds keys 0 through 63.
    bits = 0
    shift = 0
    for word in reversed(hex_words.split()):
        bits |= int(word, 16) << shift
        shift += 64
    return bits
