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
import subprocess
import threading
from pathlib import Path

# linux/input-event-codes.h
KEY_CODES = {
    "rightctrl": 97,
    "leftctrl": 29,
    "rightalt": 100,
    "leftalt": 56,
    "space": 57,
    "leftmeta": 125,
    "rightmeta": 126,
}

# Either Ctrl key, and either Windows/Super key. The portal shortcut path
# cannot express this: a trigger with only modifiers was stored as key 0x00
# and never fired.
CTRL_CODES = {KEY_CODES["leftctrl"], KEY_CODES["rightctrl"]}
META_CODES = {KEY_CODES["leftmeta"], KEY_CODES["rightmeta"]}
SHIFT_CODES = {42, 54}
ALT_CODES = {KEY_CODES["leftalt"], KEY_CODES["rightalt"]}

# One shortcut slot can be satisfied by either physical key.
_MODIFIER_GROUPS = {
    "shift": SHIFT_CODES,
    "ctrl": CTRL_CODES,
    "alt": ALT_CODES,
    "super": META_CODES,
}
_SINGLE_KEYS = {
    "esc": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "0": 11,
    "backspace": 14,
    "tab": 15,
    "q": 16,
    "w": 17,
    "e": 18,
    "r": 19,
    "t": 20,
    "y": 21,
    "u": 22,
    "i": 23,
    "o": 24,
    "p": 25,
    "enter": 28,
    "a": 30,
    "s": 31,
    "d": 32,
    "f": 33,
    "g": 34,
    "h": 35,
    "j": 36,
    "k": 37,
    "l": 38,
    "z": 44,
    "x": 45,
    "c": 46,
    "v": 47,
    "b": 48,
    "n": 49,
    "m": 50,
    "space": 57,
    "copilot": 193,
    "f23": 193,
}
_MODIFIER_ORDER = ("shift", "ctrl", "alt", "super")
_LABELS = {
    "shift": "Shift",
    "ctrl": "Ctrl",
    "alt": "Alt",
    "super": "Super",
    "space": "Space",
    "enter": "Enter",
    "tab": "Tab",
    "esc": "Esc",
    "backspace": "Backspace",
    "copilot": "Copilot",
    "f23": "Copilot",
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
        self._thread = threading.Thread(target=self._loop, name="wispo-hotkey", daemon=True)
        self._fds: list[int] = []
        self._lock = threading.Lock()
        self._reset_requested = False

    def start(self) -> None:
        with self._lock:
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
        with self._lock:
            for fd in self._fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds = []

    def reset(self) -> None:
        """Reset held state and request thread-safe rediscovery of keyboard FDs."""
        self.held.clear()
        with self._lock:
            self._reset_requested = True

    def _loop(self) -> None:
        import time

        try:
            while not self._stop.is_set():
                with self._lock:
                    if self._reset_requested:
                        self._reset_requested = False
                        for fd in self._fds:
                            try:
                                os.close(fd)
                            except OSError:
                                pass
                        self._fds = []
                        self.held.clear()

                if not self._fds:
                    self.held.clear()
                    new_fds = _open_keyboards(self.code)
                    with self._lock:
                        self._fds = new_fds
                    if not self._fds:
                        time.sleep(1.0)
                        continue

                try:
                    with self._lock:
                        fds_to_poll = list(self._fds)
                    if not fds_to_poll:
                        continue
                    readable, _, _ = select.select(fds_to_poll, [], [], 0.05)
                except (ValueError, OSError):
                    with self._lock:
                        dead = [fd for fd in self._fds if not _is_fd_valid(fd)]
                        for fd in dead:
                            self._fds.remove(fd)
                            try:
                                os.close(fd)
                            except OSError:
                                pass
                        if not self._fds:
                            self.held.clear()
                    continue

                dead_fds = []
                for fd in readable:
                    try:
                        data = os.read(fd, _EVENT.size * 64)
                        if not data:
                            dead_fds.append(fd)
                            continue
                    except OSError:
                        dead_fds.append(fd)
                        continue
                    for offset in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
                        _sec, _usec, event_type, code, value = _EVENT.unpack_from(data, offset)
                        if event_type != _EV_KEY or code != self.code:
                            continue
                        if value == 1:
                            self.held.set()
                        elif value == 0:
                            self.held.clear()

                if dead_fds:
                    with self._lock:
                        for fd in dead_fds:
                            if fd in self._fds:
                                self._fds.remove(fd)
                            try:
                                os.close(fd)
                            except OSError:
                                pass
                        if not self._fds:
                            self.held.clear()
        finally:
            with self._lock:
                for fd in self._fds:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                self._fds = []
                self.held.clear()


class HoldChord:
    """Hold is true only while every configured key is down."""

    def __init__(self, keys: tuple[str, ...] | list[str] = ("ctrl", "super")) -> None:
        self.keys = tuple(key.lower() for key in keys)
        if not self.keys:
            raise ValueError("shortcut needs at least one key")
        self.groups = tuple(codes_for(name) for name in self.keys)
        self.description = format_chord(self.keys)
        self.held = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="wispo-chord", daemon=True)
        self._fds: list[int] = []
        self._down: set[int] = set()
        self._lock = threading.Lock()
        self._reset_requested = False

    def start(self) -> None:
        codes: set[int] = set()
        for group in self.groups:
            codes.update(group)
        with self._lock:
            self._fds = _open_keyboards(codes)
            if not self._fds:
                raise RuntimeError(
                    "no keyboard exposing Ctrl and Super was readable. "
                    "The account needs to be in the input group."
                )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.held.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            for fd in self._fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds = []

    def reset(self) -> None:
        """Reset pressed state and request thread-safe rediscovery of keyboard FDs."""
        self.held.clear()
        with self._lock:
            self._down.clear()
            self._reset_requested = True

    def _loop(self) -> None:
        import time

        watched: set[int] = set()
        for group in self.groups:
            watched.update(group)
        try:
            while not self._stop.is_set():
                with self._lock:
                    if self._reset_requested:
                        self._reset_requested = False
                        for fd in self._fds:
                            try:
                                os.close(fd)
                            except OSError:
                                pass
                        self._fds = []
                        self._down.clear()
                        self.held.clear()

                if not self._fds:
                    with self._lock:
                        self._down.clear()
                        self.held.clear()
                    codes: set[int] = set()
                    for group in self.groups:
                        codes.update(group)
                    new_fds = _open_keyboards(codes)
                    with self._lock:
                        self._fds = new_fds
                    if not self._fds:
                        time.sleep(1.0)
                        continue

                try:
                    with self._lock:
                        fds_to_poll = list(self._fds)
                    if not fds_to_poll:
                        continue
                    readable, _, _ = select.select(fds_to_poll, [], [], 0.05)
                except (ValueError, OSError):
                    with self._lock:
                        dead = [fd for fd in self._fds if not _is_fd_valid(fd)]
                        for fd in dead:
                            self._fds.remove(fd)
                            try:
                                os.close(fd)
                            except OSError:
                                pass
                        if not self._fds:
                            self._down.clear()
                            self.held.clear()
                    continue

                dead_fds = []
                for fd in readable:
                    try:
                        data = os.read(fd, _EVENT.size * 64)
                        if not data:
                            dead_fds.append(fd)
                            continue
                    except OSError:
                        dead_fds.append(fd)
                        continue
                    for offset in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
                        _sec, _usec, event_type, code, value = _EVENT.unpack_from(data, offset)
                        if event_type != _EV_KEY or code not in watched or value == 2:
                            continue
                        with self._lock:
                            if value == 1:
                                self._down.add(code)
                            elif value == 0:
                                self._down.discard(code)
                            if chord_satisfied(self._down, self.groups):
                                self.held.set()
                            else:
                                self.held.clear()

                if dead_fds:
                    with self._lock:
                        for fd in dead_fds:
                            if fd in self._fds:
                                self._fds.remove(fd)
                            try:
                                os.close(fd)
                            except OSError:
                                pass
                        if not self._fds:
                            self._down.clear()
                            self.held.clear()
        finally:
            with self._lock:
                for fd in self._fds:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                self._fds = []
                self._down.clear()
                self.held.clear()


class OverviewKey:
    """Move GNOME's overview key aside for the life of this process.

    Mutter opens the overview when Super is released. That is separate from
    reading the key, so listening alone still opens the overview. The previous
    setting is put back on exit.
    """

    def __init__(self) -> None:
        self._previous: str | None = None

    def start(self) -> None:
        current = _gsettings_get("org.gnome.mutter", "overlay-key")
        if current == "":
            return
        self._previous = current
        subprocess.run(
            ["gsettings", "set", "org.gnome.mutter", "overlay-key", ""],
            check=True,
        )

    def stop(self) -> None:
        previous, self._previous = self._previous, None
        if not previous:
            return
        subprocess.run(
            ["gsettings", "set", "org.gnome.mutter", "overlay-key", previous],
            check=False,
        )


def chord_satisfied(down: set[int], groups: tuple[set[int], ...]) -> bool:
    return all(down & group for group in groups)


def codes_for(name: str) -> set[int]:
    key = name.lower()
    if key in _MODIFIER_GROUPS:
        return set(_MODIFIER_GROUPS[key])
    if key in _SINGLE_KEYS:
        return {_SINGLE_KEYS[key]}
    if key.startswith("key") and key[3:].isdigit():
        return {int(key[3:])}
    raise ValueError(f"unknown shortcut key {name!r}")


def format_chord(keys: tuple[str, ...] | list[str]) -> str:
    present = {key.lower() for key in keys}
    ordered = [key for key in _MODIFIER_ORDER if key in present]
    ordered.extend(key.lower() for key in keys if key.lower() not in _MODIFIER_ORDER)
    return "+".join(_LABELS.get(key, key.upper() if len(key) == 1 else key) for key in ordered)


def chord_from_codes(codes: set[int]) -> tuple[str, ...]:
    """Turn the keys that were held together into shortcut names."""
    # The physical Copilot key on modern laptops sends Shift (42) + Meta (125) + F23 (193).
    # Normalize this macro into the single dedicated 'copilot' key so synthetic modifiers
    # (especially Meta/Super) do not interfere with GNOME Shell's overlay key.
    if 193 in codes:
        return ("copilot",)

    names: list[str] = []
    seen: set[str] = set()
    for code in sorted(codes):
        if code <= 0 or code >= 256:
            continue
        name = _logical_name(code)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    ordered = [key for key in _MODIFIER_ORDER if key in seen]
    ordered.extend(name for name in names if name not in _MODIFIER_ORDER)
    return tuple(ordered)


def capture_chord(stop: threading.Event, timeout: float = 8.0) -> tuple[str, ...]:
    """Read the next press-and-release from the keyboard. Empty on timeout."""
    import time

    fds = _open_keyboards(CTRL_CODES | META_CODES | {KEY_CODES["space"], 193})
    if not fds:
        raise RuntimeError(
            "no keyboard was readable. The account needs to be in the input group."
        )
    down: set[int] = set()
    seen: set[int] = set()
    deadline = time.monotonic() + timeout
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            readable, _, _ = select.select(fds, [], [], 0.05)
            for fd in readable:
                try:
                    data = os.read(fd, _EVENT.size * 64)
                except OSError:
                    continue
                for offset in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
                    _sec, _usec, event_type, code, value = _EVENT.unpack_from(data, offset)
                    if event_type != _EV_KEY or value == 2 or code <= 0 or code >= 256:
                        continue
                    if value == 1:
                        down.add(code)
                        seen.add(code)
                    elif value == 0:
                        down.discard(code)
                        if seen and not down:
                            return chord_from_codes(seen)
        return ()
    finally:
        for fd in fds:
            os.close(fd)


def _logical_name(code: int) -> str:
    for name, group in _MODIFIER_GROUPS.items():
        if code in group:
            return name
    for name, single in _SINGLE_KEYS.items():
        if code == single:
            return name
    return f"key{code}"


def _gsettings_get(schema: str, key: str) -> str:
    raw = subprocess.check_output(["gsettings", "get", schema, key], text=True).strip()
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        return raw[1:-1]
    return raw


def _open_keyboards(codes: set[int] | int) -> list[int]:
    if isinstance(codes, int):
        codes = {codes}
    opened: list[int] = []
    for path in _keyboard_paths(codes):
        try:
            opened.append(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
        except OSError:
            continue
    return opened


def _keyboard_paths(codes: set[int]) -> list[str]:
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
        if not any(key_bits & (1 << code) for code in codes):
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


def _is_fd_valid(fd: int) -> bool:
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False
