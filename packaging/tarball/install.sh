#!/usr/bin/env bash
set -e

# ==============================================================================
# Wispo Offline Linux x86_64 Installer
# ==============================================================================

if [ "$EUID" -ne 0 ]; then
    echo "Error: This installer must be run with root privileges." >&2
    echo "Usage: sudo ./install.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="/opt/wispo"

# Support running directly from archive root or packaging/tarball/
if [ -d "${SCRIPT_DIR}/src" ]; then
    PAYLOAD_DIR="${SCRIPT_DIR}"
elif [ -d "${SCRIPT_DIR}/../../src" ]; then
    PAYLOAD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    PAYLOAD_DIR="${SCRIPT_DIR}"
fi

if [ ! -f "${PAYLOAD_DIR}/runtime-python.env" ] || [ ! -f "${PAYLOAD_DIR}/wheels/SHA256SUMS" ]; then
    echo "Error: this archive is missing its offline Python runtime payload." >&2
    exit 1
fi
# shellcheck disable=SC1090
. "${PAYLOAD_DIR}/runtime-python.env"
ACTUAL_PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

if [ "${ACTUAL_PYTHON_VERSION}" != "${PYTHON_VERSION}" ]; then
    echo "Error: this Wispo archive requires Python ${PYTHON_VERSION}; found Python ${ACTUAL_PYTHON_VERSION}." >&2
    echo "Install the matching Ubuntu Python runtime, then run this installer again." >&2
    exit 1
fi

echo "==> Checking host GTK bindings..."
if ! python3 -c 'import gi' >/dev/null 2>&1; then
    echo "Error: PyGObject is missing. On Ubuntu, run:" >&2
    echo "  sudo apt install python3-gi gir1.2-gtk-4.0" >&2
    exit 1
fi
if ! python3 -m ensurepip --version >/dev/null 2>&1; then
    echo "Error: Python venv support is missing. On Ubuntu, run:" >&2
    echo "  sudo apt install python${PYTHON_VERSION}-venv" >&2
    exit 1
fi
if ! (cd "${PAYLOAD_DIR}/wheels" && sha256sum -c SHA256SUMS >/dev/null); then
    echo "Error: a bundled Python wheel failed its integrity check. Download the archive again." >&2
    exit 1
fi

echo "==> Installing Wispo to ${TARGET_DIR}..."

# Older Wispo archives installed this rule even though current releases use
# the desktop portal rather than /dev/uinput. Remove only our legacy file.
if [ -f /etc/udev/rules.d/99-wispo-uinput.rules ]; then
    rm -f /etc/udev/rules.d/99-wispo-uinput.rules
    if command -v udevadm >/dev/null 2>&1; then
        udevadm control --reload-rules || true
    fi
fi

# 1. Create directory structure
mkdir -p "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}/native/build"
mkdir -p "${TARGET_DIR}/native/lib"
mkdir -p /usr/share/applications
mkdir -p /usr/share/icons/hicolor

# 2. Copy application source and assets
echo "==> Copying application payload..."
cp -r "${PAYLOAD_DIR}/src" "${TARGET_DIR}/"
cp "${PAYLOAD_DIR}/wispo.yaml" "${TARGET_DIR}/" 2>/dev/null || true
cp "${PAYLOAD_DIR}/pyproject.toml" "${TARGET_DIR}/"
cp "${PAYLOAD_DIR}/README.md" "${TARGET_DIR}/" 2>/dev/null || true
cp "${PAYLOAD_DIR}/LICENSE" "${TARGET_DIR}/" 2>/dev/null || true
cp "${PAYLOAD_DIR}/requirements-runtime.txt" "${TARGET_DIR}/"
cp -a "${PAYLOAD_DIR}/wheels" "${TARGET_DIR}/"

# Copy the native wrapper and its bundled whisper.cpp/ggml libraries.
if [ -f "${PAYLOAD_DIR}/native/build/libwispo_asr.so" ]; then
    cp "${PAYLOAD_DIR}/native/build/libwispo_asr.so" "${TARGET_DIR}/native/build/"
elif [ -f "${PAYLOAD_DIR}/libwispo_asr.so" ]; then
    cp "${PAYLOAD_DIR}/libwispo_asr.so" "${TARGET_DIR}/native/build/"
else
    echo "Error: packaged native library is missing." >&2
    exit 1
fi
if [ -d "${PAYLOAD_DIR}/native/lib" ]; then
    cp -a "${PAYLOAD_DIR}/native/lib/." "${TARGET_DIR}/native/lib/"
else
    echo "Error: packaged whisper.cpp runtime libraries are missing." >&2
    exit 1
fi

# 3. Install the private Python environment from bundled wheels. No network
# request is made here; this is deliberately not a live pip/PyPI install.
echo "==> Installing bundled Python runtime..."
rm -rf "${TARGET_DIR}/venv"
python3 -m venv --system-site-packages "${TARGET_DIR}/venv"
if ! "${TARGET_DIR}/venv/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-index \
    --no-deps \
    --ignore-installed \
    --find-links "${TARGET_DIR}/wheels" \
    --requirement "${TARGET_DIR}/requirements-runtime.txt"; then
    echo "Error: could not install Wispo's bundled Python dependencies." >&2
    exit 1
fi

# 4. Install launcher wrapper
echo "==> Installing launcher into /usr/bin/wispo..."
if [ ! -f "${PAYLOAD_DIR}/packaging/bin/wispo" ]; then
    echo "Error: packaged launcher is missing." >&2
    exit 1
fi
install -m 755 "${PAYLOAD_DIR}/packaging/bin/wispo" /usr/bin/wispo

# 5. Install icons
echo "==> Generating application icons..."
python3 -c "
from pathlib import Path
from PIL import Image

logo = Path('${TARGET_DIR}/src/wispo/assets/logo.png')
if logo.is_file():
    with Image.open(logo) as im:
        for sz in (16, 32, 48, 64, 128, 256, 512):
            out_dir = Path(f'/usr/share/icons/hicolor/{sz}x{sz}/apps')
            out_dir.mkdir(parents=True, exist_ok=True)
            resized = im.resize((sz, sz), Image.Resampling.LANCZOS)
            resized.save(out_dir / 'com.wispo.Wispo.png')
            resized.save(out_dir / 'com.wispo.Settings.png')
" 2>/dev/null || true

# 6. Install Desktop Entry
echo "==> Registering desktop application..."
DESKTOP_SRC=""
if [ -f "${PAYLOAD_DIR}/packaging/com.wispo.Wispo.desktop" ]; then
    DESKTOP_SRC="${PAYLOAD_DIR}/packaging/com.wispo.Wispo.desktop"
elif [ -f "${PAYLOAD_DIR}/com.wispo.Wispo.desktop" ]; then
    DESKTOP_SRC="${PAYLOAD_DIR}/com.wispo.Wispo.desktop"
fi

if [ -n "${DESKTOP_SRC}" ]; then
    cp "${DESKTOP_SRC}" /usr/share/applications/com.wispo.Wispo.desktop
    chmod 644 /usr/share/applications/com.wispo.Wispo.desktop
fi

# 7. Refresh caches
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t /usr/share/icons/hicolor || true
fi

# 8. Install uninstaller
if [ -f "${PAYLOAD_DIR}/packaging/tarball/uninstall.sh" ]; then
    cp "${PAYLOAD_DIR}/packaging/tarball/uninstall.sh" "${TARGET_DIR}/uninstall.sh"
    chmod 755 "${TARGET_DIR}/uninstall.sh"
elif [ -f "${PAYLOAD_DIR}/uninstall.sh" ]; then
    cp "${PAYLOAD_DIR}/uninstall.sh" "${TARGET_DIR}/uninstall.sh"
    chmod 755 "${TARGET_DIR}/uninstall.sh"
fi

echo ""
echo "======================================================================"
echo "  Wispo installed successfully!"
echo "  Run 'wispo' in your terminal or launch Wispo from your applications."
echo "  To uninstall: sudo /opt/wispo/uninstall.sh"
echo "======================================================================"
