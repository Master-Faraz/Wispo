#!/usr/bin/env bash
set -e

# ==============================================================================
# Wispo Automated Multi-Format Packaging Pipeline
# Builds:
#   1. wispo-{VERSION}-linux-x86_64.tar.gz (Universal distribution bundle)
#   2. wispo_{VERSION}_amd64.deb (Debian / Ubuntu native package)
#   3. SHA256SUMS (Cryptographic verification checksums)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

# 1. Extract version from pyproject.toml
VERSION=$(grep -m 1 '^version = ' pyproject.toml | cut -d '"' -f 2)
if [ -z "${VERSION}" ]; then
    VERSION="0.1.0"
fi

echo "======================================================================"
echo "  Building Wispo Release Packages: v${VERSION}"
echo "======================================================================"

# 2. Check native shared library
if [ ! -f "native/build/libwispo_asr.so" ]; then
    echo "==> Native library libwispo_asr.so not found. Attempting build..."
    if [ -f "scripts/build_sycl.sh" ]; then
        bash scripts/build_sycl.sh || {
            echo "Warning: Native build failed. Continuing if fallback CPU mode is available." >&2
        }
    fi
fi

# 3. Create dist directory and temporary staging sandbox
DIST_DIR="${ROOT_DIR}/dist"
mkdir -p "${DIST_DIR}"

STAGING_BASE="/tmp/wispo-build-staging-$$"
mkdir -p "${STAGING_BASE}"

trap 'rm -rf "${STAGING_BASE}"' EXIT

# ==============================================================================
# BUILD 1: Universal Tarball (.tar.gz)
# ==============================================================================
echo ""
echo "==> [1/2] Assembling Universal Tarball (wispo-${VERSION}-linux-x86_64.tar.gz)..."

TAR_STAGE="${STAGING_BASE}/wispo"
mkdir -p "${TAR_STAGE}"
mkdir -p "${TAR_STAGE}/native/build"
mkdir -p "${TAR_STAGE}/packaging"

# Copy essential files
cp -r src "${TAR_STAGE}/"
cp pyproject.toml "${TAR_STAGE}/"
cp wispo.yaml "${TAR_STAGE}/" 2>/dev/null || true
cp README.md "${TAR_STAGE}/" 2>/dev/null || true
cp LICENSE "${TAR_STAGE}/" 2>/dev/null || true
cp -r packaging/* "${TAR_STAGE}/packaging/"

# Copy native library if built
if [ -f "native/build/libwispo_asr.so" ]; then
    cp "native/build/libwispo_asr.so" "${TAR_STAGE}/native/build/"
fi

# Put top-level install and uninstall scripts at root of tarball
cp packaging/tarball/install.sh "${TAR_STAGE}/install.sh"
cp packaging/tarball/uninstall.sh "${TAR_STAGE}/uninstall.sh"
chmod +x "${TAR_STAGE}/install.sh" "${TAR_STAGE}/uninstall.sh"

# Ensure models directory exists with placeholder
mkdir -p "${TAR_STAGE}/models"
touch "${TAR_STAGE}/models/.gitkeep"

# Purge any stray caches in staging
find "${TAR_STAGE}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${TAR_STAGE}" -name "*.pyc" -delete 2>/dev/null || true

# Compress tarball
TAR_FILE="${DIST_DIR}/wispo-${VERSION}-linux-x86_64.tar.gz"
tar -czvf "${TAR_FILE}" -C "${STAGING_BASE}" wispo > /dev/null

echo "    -> Created: ${TAR_FILE} ($(du -sh "${TAR_FILE}" | cut -f1))"

# ==============================================================================
# BUILD 2: Debian Package (.deb)
# ==============================================================================
echo ""
echo "==> [2/2] Assembling Debian Package (wispo_${VERSION}_amd64.deb)..."

DEB_STAGE="${STAGING_BASE}/deb"
mkdir -p "${DEB_STAGE}/DEBIAN"
mkdir -p "${DEB_STAGE}/opt/wispo"
mkdir -p "${DEB_STAGE}/opt/wispo/native/build"
mkdir -p "${DEB_STAGE}/usr/bin"
mkdir -p "${DEB_STAGE}/usr/share/applications"
mkdir -p "${DEB_STAGE}/etc/udev/rules.d"

# Copy payload to /opt/wispo
cp -r src "${DEB_STAGE}/opt/wispo/"
cp pyproject.toml "${DEB_STAGE}/opt/wispo/"
cp wispo.yaml "${DEB_STAGE}/opt/wispo/" 2>/dev/null || true
cp README.md "${DEB_STAGE}/opt/wispo/" 2>/dev/null || true
cp LICENSE "${DEB_STAGE}/opt/wispo/" 2>/dev/null || true

if [ -f "native/build/libwispo_asr.so" ]; then
    cp "native/build/libwispo_asr.so" "${DEB_STAGE}/opt/wispo/native/build/"
fi

# Copy wrapper to /usr/bin/wispo
cp packaging/bin/wispo "${DEB_STAGE}/usr/bin/wispo"
chmod 755 "${DEB_STAGE}/usr/bin/wispo"

# Copy desktop launcher
cp packaging/com.wispo.Wispo.desktop "${DEB_STAGE}/usr/share/applications/com.wispo.Wispo.desktop"
chmod 644 "${DEB_STAGE}/usr/share/applications/com.wispo.Wispo.desktop"

# Copy udev rules
cp packaging/99-wispo-uinput.rules "${DEB_STAGE}/etc/udev/rules.d/99-wispo-uinput.rules"
chmod 644 "${DEB_STAGE}/etc/udev/rules.d/99-wispo-uinput.rules"

# Copy Debian control and lifecycle scripts
sed "s/^Version: .*/Version: ${VERSION}/" packaging/deb/control > "${DEB_STAGE}/DEBIAN/control"
cp packaging/deb/postinst "${DEB_STAGE}/DEBIAN/postinst"
cp packaging/deb/prerm "${DEB_STAGE}/DEBIAN/prerm"
chmod 755 "${DEB_STAGE}/DEBIAN/postinst" "${DEB_STAGE}/DEBIAN/prerm"

# Pre-generate multi-resolution icons into package
python3 -c "
from pathlib import Path
from PIL import Image

logo = Path('src/wispo/assets/logo.png')
if logo.is_file():
    with Image.open(logo) as im:
        for sz in (16, 32, 48, 64, 128, 256, 512):
            out_dir = Path('${DEB_STAGE}/usr/share/icons/hicolor') / f'{sz}x{sz}/apps'
            out_dir.mkdir(parents=True, exist_ok=True)
            resized = im.resize((sz, sz), Image.Resampling.LANCZOS)
            resized.save(out_dir / 'com.wispo.Wispo.png')
            resized.save(out_dir / 'com.wispo.Settings.png')
" 2>/dev/null || true

# Purge any bytecode caches
find "${DEB_STAGE}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${DEB_STAGE}" -name "*.pyc" -delete 2>/dev/null || true

# Build debian archive
DEB_FILE="${DIST_DIR}/wispo_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "${DEB_STAGE}" "${DEB_FILE}" > /dev/null

echo "    -> Created: ${DEB_FILE} ($(du -sh "${DEB_FILE}" | cut -f1))"

# ==============================================================================
# BUILD 3: SHA256 Checksums
# ==============================================================================
echo ""
echo "==> Generating SHA256 checksums..."
cd "${DIST_DIR}"
sha256sum "$(basename "${TAR_FILE}")" "$(basename "${DEB_FILE}")" > SHA256SUMS
cat SHA256SUMS

echo ""
echo "======================================================================"
echo "  Build Complete! Artifacts ready in dist/:"
echo "    - ${TAR_FILE}"
echo "    - ${DEB_FILE}"
echo "    - ${DIST_DIR}/SHA256SUMS"
echo "======================================================================"
