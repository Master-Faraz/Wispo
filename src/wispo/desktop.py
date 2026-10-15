"""Desktop entry and icon installation for the background service.

Installs a .desktop file and application icons under ~/.local/share/ so
the service can be launched from the system application launcher without
a terminal.
"""

from __future__ import annotations

from pathlib import Path

# GNOME matches a running window to this file name. The dock icon and the
# recording badge both use it.
DESKTOP_ID = "com.wispo.Wispo.desktop"
APP_ID = "com.wispo.Wispo"
ICON_SIZES = (512, 256, 128, 64, 48, 32, 16)


def find_logo_path(root: Path | str | None = None) -> Path | None:
    candidates: list[Path] = []
    if root:
        r = Path(root)
        candidates.extend([
            r / "src" / "wispo" / "assets" / "logo.png",
            r / "logo.png",
        ])
    here = Path(__file__).resolve().parent
    candidates.extend([
        here / "assets" / "logo.png",
        here.parent.parent / "logo.png",
    ])
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def install_icons(root: Path | str | None = None, icons_base: Path | None = None) -> list[Path]:
    logo = find_logo_path(root)
    if not logo or not logo.is_file():
        return []
    base = icons_base or (Path.home() / ".local" / "share" / "icons" / "hicolor")
    installed: list[Path] = []
    app_names = (APP_ID, "com.wispo.Settings")

    try:
        from PIL import Image

        with Image.open(logo) as im:
            for sz in ICON_SIZES:
                out_dir = base / f"{sz}x{sz}" / "apps"
                out_dir.mkdir(parents=True, exist_ok=True)
                resized = im.resize((sz, sz), Image.Resampling.LANCZOS)
                for name in app_names:
                    out_file = out_dir / f"{name}.png"
                    resized.save(out_file, format="PNG")
                    installed.append(out_file)
    except Exception:
        import shutil

        out_dir = base / "512x512" / "apps"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in app_names:
            out_file = out_dir / f"{name}.png"
            shutil.copy2(logo, out_file)
            installed.append(out_file)

    try:
        import subprocess

        subprocess.run(["gtk-update-icon-cache", "-q", "-t", str(base)], check=False, timeout=3)
    except Exception:
        pass

    return installed


def desktop_entry_text(python: str, root: str, icon_name: str = APP_ID, name: str = "Wispo") -> str:
    return (
        "[Desktop Entry]\n"
        f"Name={name}\n"
        "Comment=Hold Ctrl+Super to transcribe into the focused window\n"
        f"Exec={python} -m wispo.service\n"
        f"Path={root}\n"
        f"Icon={icon_name}\n"
        "Terminal=false\n"
        "Type=Application\n"
        "Categories=Utility;\n"
        "StartupNotify=false\n"
        f"StartupWMClass={APP_ID}\n"
    )


def install_desktop_entry(python: str, root: str, directory: Path | None = None) -> Path:
    folder = directory or (Path.home() / ".local" / "share" / "applications")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        install_icons(root)
    except Exception:
        pass
    path = folder / DESKTOP_ID
    entry_text = desktop_entry_text(python, root)
    path.write_text(entry_text, encoding="utf-8")

    # Clean up old desktop entry if named differently
    previous = folder / "wispo.desktop"
    if previous != path and previous.is_file():
        try:
            previous.unlink()
        except OSError:
            pass

    return path

