"""Type one sentence through the desktop portal, without the microphone.

    uv run python -m wispo.inject "hello from Wispo"

Allow the permission dialog if GNOME shows one, focus a text field, and the
sentence is typed there. The clipboard is not used.
"""

from __future__ import annotations

import sys
import time

from wispo.injection.portal import PortalError, PortalKeyboard


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    text = " ".join(args).strip() or "hello from Wispo"
    keyboard = PortalKeyboard()
    print("Opening the input session.")
    print("If GNOME asks for permission, allow keyboard control.")
    try:
        keyboard.open()
    except PortalError as exc:
        print(f"Input session failed: {exc}", file=sys.stderr)
        return 1
    print("Session ready. Focus a text field. Typing starts in 3 seconds.")
    try:
        time.sleep(3)
        keyboard.insert(text)
    except PortalError as exc:
        print(f"Typing failed: {exc}", file=sys.stderr)
        return 1
    finally:
        keyboard.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
