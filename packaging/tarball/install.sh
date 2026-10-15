#!/usr/bin/env bash
set -e

# ==============================================================================
# Wispo Universal Linux Installer
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

echo "==> Installing Wispo to ${TARGET_DIR}..."

# 1. Create directory structure
mkdir -p "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}/bin"
mkdir -p "${TARGET_DIR}/native/build"
mkdir -p /usr/share/applications
mkdir -p /usr/share/icons/hicolor
mkdir -p /etc/udev/rules.d

# 2. Copy application source and assets
echo "==> Copying application payload..."
cp -r "${PAYLOAD_DIR}/src" "${TARGET_DIR}/"
cp "${PAYLOAD_DIR}/wispo.yaml" "${TARGET_DIR}/" 2>/dev/null || true
cp "${PAYLOAD_DIR}/pyproject.toml" "${TARGET_DIR}/"
cp "${PAYLOAD_DIR}/README.md" "${TARGET_DIR}/" 2>/dev/null || true
cp "${PAYLOAD_DIR}/LICENSE" "${TARGET_DIR}/" 2>/dev/null || true

# Copy native SYCL / C++ library if built
if [ -f "${PAYLOAD_DIR}/native/build/libwispo_asr.so" ]; then
    cp "${PAYLOAD_DIR}/native/build/libwispo_asr.so" "${TARGET_DIR}/native/build/"
elif [ -f "${PAYLOAD_DIR}/libwispo_asr.so" ]; then
    cp "${PAYLOAD_DIR}/libwispo_asr.so" "${TARGET_DIR}/native/build/"
fi

# 3. Setup Python virtual environment
echo "==> Setting up isolated Python runtime..."
python3 -m venv "${TARGET_DIR}/venv"
"${TARGET_DIR}/venv/bin/pip" install --upgrade pip --quiet
"${TARGET_DIR}/venv/bin/pip" install "${TARGET_DIR}" --quiet

# 4. Install launcher wrapper
echo "==> Installing launcher into /usr/bin/wispo..."
cat << 'EOF' > /usr/bin/wispo
#!/usr/bin/env bash
exec /opt/wispo/venv/bin/wispo "$@"
EOF
chmod 755 /usr/bin/wispo

# 5. Install udev rules for /dev/uinput text injection
echo "==> Configuring uinput permissions..."
if [ -f "${PAYLOAD_DIR}/packaging/99-wispo-uinput.rules" ]; then
    cp "${PAYLOAD_DIR}/packaging/99-wispo-uinput.rules" /etc/udev/rules.d/99-wispo-uinput.rules
elif [ -f "${PAYLOAD_DIR}/99-wispo-uinput.rules" ]; then
    cp "${PAYLOAD_DIR}/99-wispo-uinput.rules" /etc/udev/rules.d/99-wispo-uinput.rules
fi

if command -v udevadm >/dev/null 2>&1; then
    udevadm control --reload-rules || true
    udevadm trigger /dev/uinput || true
fi

# 6. Install icons
echo "==> Generating application icons..."
"${TARGET_DIR}/venv/bin/python" -c "
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

# 7. Install Desktop Entry
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

# 8. Refresh caches
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t /usr/share/icons/hicolor || true
fi

# 9. Install uninstaller
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
