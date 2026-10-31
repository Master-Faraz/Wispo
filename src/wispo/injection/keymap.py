"""Turn a finished transcript into US-layout key presses.

GNOME's remote-desktop portal accepts evdev keycodes, not a Unicode string.
This machine's layout is US, so each character becomes the key that produces
it, plus Shift when the character is on the shifted row. The clipboard is
not involved.
"""

from __future__ import annotations

KEY_LEFTSHIFT = 42
KEY_ENTER = 28
KEY_SPACE = 57

# evdev codes for the US QWERTY rows, in the same order as the characters.
_ROW_CODES = (
    41,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    43,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
)
_UNSHIFTED = r"`1234567890-=qwertyuiop[]\asdfghjkl;'zxcvbnm,./"
_SHIFTED = r'~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>?'


class UnsupportedCharacter(ValueError):
    def __init__(self, char: str) -> None:
        self.char = char
        shown = char if char.isprintable() else repr(char)
        super().__init__(
            f"cannot type {shown} with the US keyboard layout used for injection"
        )


def _layout() -> dict[str, tuple[int, bool]]:
    if not (len(_UNSHIFTED) == len(_SHIFTED) == len(_ROW_CODES)):
        raise RuntimeError("US keymap rows do not line up")
    table: dict[str, tuple[int, bool]] = {}
    for code, plain, shifted in zip(_ROW_CODES, _UNSHIFTED, _SHIFTED, strict=True):
        table[plain] = (code, False)
        table[shifted] = (code, True)
    return table


_US = _layout()

_CHAR_REPLACEMENTS = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
}


def key_events(text: str, ignore_unsupported: bool = False) -> list[tuple[int, int]]:
    """Press and release events as (evdev code, 1 press or 0 release)."""
    normalized = "".join(_CHAR_REPLACEMENTS.get(c, c) for c in text)
    events: list[tuple[int, int]] = []
    shift_down = False
    for char in normalized:
        if char == "\n":
            code, needs_shift = KEY_ENTER, False
        elif char == " ":
            code, needs_shift = KEY_SPACE, False
        else:
            found = _US.get(char)
            if found is None:
                if ignore_unsupported:
                    continue
                raise UnsupportedCharacter(char)
            code, needs_shift = found
        if needs_shift and not shift_down:
            events.append((KEY_LEFTSHIFT, 1))
            shift_down = True
        elif shift_down and not needs_shift:
            events.append((KEY_LEFTSHIFT, 0))
            shift_down = False
        events.append((code, 1))
        events.append((code, 0))
    if shift_down:
        events.append((KEY_LEFTSHIFT, 0))
    return events
