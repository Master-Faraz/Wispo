"""Launcher entry for the background process.

The path is this machine's project and virtualenv, written into the user's
local applications directory so Ubuntu's launcher can start WhisperFlow
without a terminal.
"""

from __future__ import annotations

from pathlib import Path

# GNOME matches a running window to this file name. The dock icon and the
# recording badge both use it.
DESKTOP_ID = "com.whisperflow.WhisperFlow.desktop"
APP_ID = "com.whisperflow.WhisperFlow"


def desktop_entry_text(python: str, root: str) -> str:
    return (
        "[Desktop Entry]\n"
        "Name=WhisperFlow\n"
        "Comment=Hold Ctrl+Super to transcribe into the focused window\n"
        f"Exec={python} -m whisperflow.service\n"
        f"Path={root}\n"
        "Icon=audio-input-microphone\n"
        "Terminal=false\n"
        "Type=Application\n"
        "Categories=Utility;\n"
        "StartupNotify=false\n"
        f"StartupWMClass={APP_ID}\n"
    )


def install_desktop_entry(python: str, root: str, directory: Path | None = None) -> Path:
    folder = directory or (Path.home() / ".local" / "share" / "applications")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / DESKTOP_ID
    path.write_text(desktop_entry_text(python, root), encoding="utf-8")
    previous = folder / "whisperflow.desktop"
    if previous != path and previous.is_file():
        previous.unlink()
    return path
