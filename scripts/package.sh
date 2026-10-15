#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Wispo Automated Multi-Format Packaging Pipeline
# Builds:
#   1. wispo-{VERSION}-ubuntu-26.04-x86_64.tar.gz (offline x86_64 bundle)
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

# 2. Build and validate the native runtime. The wrapper alone is not enough:
# it dynamically links whisper.cpp and the ggml backend libraries.
if [ ! -f "native/build/libwispo_asr.so" ]; then
    echo "==> Native library libwispo_asr.so not found. Building it..."
    bash scripts/build_sycl.sh
fi

if [ ! -f "native/build/libwispo_asr.so" ]; then
    echo "Error: native/build/libwispo_asr.so was not produced." >&2
    exit 1
fi

WHISPER_LIBRARY="$(find third_party/whisper.cpp/build-sycl -type f -name 'libwhisper.so*' -print -quit)"
if [ -z "${WHISPER_LIBRARY}" ]; then
    echo "Error: whisper.cpp shared libraries were not found under build-sycl." >&2
    exit 1
fi
WHISPER_RUNTIME_DIR="$(dirname "${WHISPER_LIBRARY}")"
if ! compgen -G "${WHISPER_RUNTIME_DIR}/libggml*.so*" > /dev/null; then
    echo "Error: ggml shared libraries were not found in ${WHISPER_RUNTIME_DIR}." >&2
    exit 1
fi

copy_native_runtime() {
    local destination="$1"
    mkdir -p "${destination}/build" "${destination}/lib"
    cp -a native/build/libwispo_asr.so "${destination}/build/"
    cp -a "${WHISPER_RUNTIME_DIR}"/libwhisper.so* "${destination}/lib/"
    cp -a "${WHISPER_RUNTIME_DIR}"/libggml*.so* "${destination}/lib/"
}

build_tarball_wheelhouse() {
    local destination="$1"
    local requirements="packaging/tarball/requirements-runtime.txt"
    local python_version
    local python_abi

    python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    python_abi="$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
    mkdir -p "${destination}"

    # Resolve only the exact runtime versions declared above. This download is
    # a release-build concern; the end-user installer remains fully offline.
    UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/wispo-uv-cache}" \
        uv tool run --isolated --from pip pip download \
        --disable-pip-version-check \
        --only-binary=:all: \
        --platform manylinux_2_28_x86_64 \
        --implementation cp \
        --python-version "${python_version}" \
        --abi "${python_abi}" \
        --dest "${destination}" \
        --requirement "${requirements}" >&2

    if ! compgen -G "${destination}/*.whl" > /dev/null; then
        echo "Error: runtime wheelhouse is empty." >&2
        exit 1
    fi
    (
        cd "${destination}"
        sha256sum ./*.whl > SHA256SUMS
    )
    printf 'PYTHON_VERSION=%s\n' "${python_version}"
}

# 3. Create dist directory and temporary staging sandbox
DIST_DIR="${ROOT_DIR}/dist"
mkdir -p "${DIST_DIR}"

STAGING_BASE="/tmp/wispo-build-staging-$$"
mkdir -p "${STAGING_BASE}"

trap 'rm -rf "${STAGING_BASE}"' EXIT

# ==============================================================================
# BUILD 1: Ubuntu 26.04 x86_64 Tarball (.tar.gz)
# ==============================================================================
echo ""
echo "==> [1/2] Assembling Ubuntu 26.04 Tarball (wispo-${VERSION}-ubuntu-26.04-x86_64.tar.gz)..."

TAR_STAGE="${STAGING_BASE}/wispo"
mkdir -p "${TAR_STAGE}"
mkdir -p "${TAR_STAGE}/native"
mkdir -p "${TAR_STAGE}/packaging"

# Copy essential files
cp -r src "${TAR_STAGE}/"
cp pyproject.toml "${TAR_STAGE}/"
cp wispo.yaml "${TAR_STAGE}/" 2>/dev/null || true
cp README.md "${TAR_STAGE}/" 2>/dev/null || true
cp LICENSE "${TAR_STAGE}/" 2>/dev/null || true
cp -r packaging/* "${TAR_STAGE}/packaging/"

# Bundle locked wheels so tarball installation never downloads Python packages.
build_tarball_wheelhouse "${TAR_STAGE}/wheels" > "${TAR_STAGE}/runtime-python.env"
cp packaging/tarball/requirements-runtime.txt "${TAR_STAGE}/requirements-runtime.txt"

# Copy the wrapper and its whisper.cpp/ggml runtime closure. oneAPI runtime
# libraries remain an explicit host prerequisite and are checked at startup.
copy_native_runtime "${TAR_STAGE}/native"

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
TAR_FILE="${DIST_DIR}/wispo-${VERSION}-ubuntu-26.04-x86_64.tar.gz"
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
mkdir -p "${DEB_STAGE}/opt/wispo/native"
mkdir -p "${DEB_STAGE}/usr/bin"
mkdir -p "${DEB_STAGE}/usr/share/applications"

# Copy payload to /opt/wispo
cp -r src "${DEB_STAGE}/opt/wispo/"
cp pyproject.toml "${DEB_STAGE}/opt/wispo/"
cp wispo.yaml "${DEB_STAGE}/opt/wispo/" 2>/dev/null || true
cp README.md "${DEB_STAGE}/opt/wispo/" 2>/dev/null || true
cp LICENSE "${DEB_STAGE}/opt/wispo/" 2>/dev/null || true

copy_native_runtime "${DEB_STAGE}/opt/wispo/native"

# Copy wrapper to /usr/bin/wispo
cp packaging/bin/wispo "${DEB_STAGE}/usr/bin/wispo"
chmod 755 "${DEB_STAGE}/usr/bin/wispo"

# Copy desktop launcher
cp packaging/com.wispo.Wispo.desktop "${DEB_STAGE}/usr/share/applications/com.wispo.Wispo.desktop"
chmod 644 "${DEB_STAGE}/usr/share/applications/com.wispo.Wispo.desktop"

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
