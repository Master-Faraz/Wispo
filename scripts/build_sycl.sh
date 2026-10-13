#!/usr/bin/env bash
# Build whisper.cpp and the small C wrapper against the Intel Arc GPU.
#
# whisper.cpp's SYCL backend needs two things that are not the Python venv:
#   1. icpx, from intel-oneapi-compiler-dpcpp-cpp (already installed)
#   2. oneMKL, which the SYCL CMake file links unconditionally on Intel
#
# If CMake stops on MKLConfig.cmake, install the math library and rerun:
#   sudo apt install intel-oneapi-mkl-devel-2026.1 libze-dev
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/whisper.cpp"
BUILD="$SRC/build-sycl"
WRAPPER_OUT="$ROOT/native/build"

if [[ ! -d "$SRC" ]]; then
  echo "whisper.cpp source is missing at $SRC" >&2
  echo "Clone it with: git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git third_party/whisper.cpp" >&2
  exit 1
fi

# Intel's vars.sh unsets OCL_ICD_FILENAMES and then reads it. `set -u` above
# treats that as a crash, so the nounset check is off only while sourcing.
# A second source in the same shell exits 3, so skip it when oneAPI is already active.
set +u
if [[ -z "${ONEAPI_ROOT:-}" ]]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh >/dev/null
fi
# setvars sourced before oneMKL was installed will not have MKLROOT. Load it
# on its own so CMake can see MKLConfig.cmake.
if [[ -z "${MKLROOT:-}" && -f /opt/intel/oneapi/mkl/latest/env/vars.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/mkl/latest/env/vars.sh
fi
set -u
command -v icpx >/dev/null

MKL_CONFIG="$(find /opt/intel/oneapi/mkl -name MKLConfig.cmake -print -quit)"
if [[ -z "$MKL_CONFIG" ]]; then
  echo "oneMKL development files were not found." >&2
  echo "Install them with: sudo apt install intel-oneapi-mkl-devel-2026.1" >&2
  exit 1
fi

cmake -S "$SRC" -B "$BUILD" \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_DNN=OFF \
  -DGGML_SYCL_F16=OFF \
  -DMKL_DIR="$(dirname "$MKL_CONFIG")" \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_EXAMPLES=ON \
  -DBUILD_SHARED_LIBS=ON

cmake --build "$BUILD" --config Release -j "$(nproc)" --target whisper-cli

# The Python process loads libwispo_asr.so. That library calls
# whisper_full(), and ggml loads libggml-sycl.so from a directory we pass at
# runtime. RPATH lets the loader find the just-built shared libraries without
# a global ldconfig change.
mapfile -t LIBDIRS < <(find "$BUILD" -name 'libwhisper.so' -o -name 'libggml.so' -o -name 'libggml-sycl.so' -printf '%h\n' | sort -u)
if [[ ${#LIBDIRS[@]} -eq 0 ]]; then
  echo "Build finished but libwhisper.so was not found under $BUILD" >&2
  exit 1
fi

RPATH=""
for dir in "${LIBDIRS[@]}"; do
  RPATH+="${RPATH:+:}$dir"
done
SYCL_LIB_DIR="/opt/intel/oneapi/compiler/2026.1/lib"
if [[ -d "$SYCL_LIB_DIR" ]]; then
  RPATH+=":$SYCL_LIB_DIR"
fi

LINK_DIRS=()
for dir in "${LIBDIRS[@]}"; do
  LINK_DIRS+=("-L$dir")
done

mkdir -p "$WRAPPER_OUT"
icpx -fsycl -fPIC -shared \
  -I "$SRC/include" \
  -I "$SRC/ggml/include" \
  "$ROOT/native/wispo_asr.cpp" \
  -o "$WRAPPER_OUT/libwispo_asr.so" \
  "${LINK_DIRS[@]}" \
  -lwhisper \
  -Wl,-rpath,"$RPATH"

echo "SYCL whisper-cli: $BUILD/bin/whisper-cli"
echo "Python wrapper:   $WRAPPER_OUT/libwispo_asr.so"
