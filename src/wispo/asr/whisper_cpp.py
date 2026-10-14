"""whisper.cpp through the small native wrapper.

The wrapper keeps one whisper_context alive. start() loads the model.
transcribe_chunk() and finalize() only run inference. stop() ends a
recording from the caller's point of view and deliberately does not free
the model, because the next utterance should not pay the load cost again.

GPU vs CPU is a flag on load, plus GGML_SYCL_DEVICE. The same .so is used
for both. A CPU fallback does not require a second build.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

from wispo.config import project_root

_TEXT_CAP = 16384

# libggml-cpu.so is linked to Intel OpenMP (libiomp5.so). That file lives in
# the oneAPI compiler directory, which is not on the default library path.
# A parent library's RUNPATH is not searched for the libraries it loads, so
# the wrapper cannot point at libiomp5.so on behalf of libggml-cpu.so.
# LD_LIBRARY_PATH is only read when the process starts, so if it is missing
# we restart this same command once with the directory added.
def _intel_lib_dirs() -> list[str]:
    """Directories that hold libiomp5, oneMKL, and the Level Zero runtime.

    Versions differ by install, so this walks the oneAPI tree instead of
    assuming 2026.1. SYCL loads some of these by name at startup; if they
    are missing it reports "no device" even though the Arc GPU is present.
    """
    roots = (
        Path("/opt/intel/oneapi/compiler"),
        Path("/opt/intel/oneapi/mkl"),
        Path("/opt/intel/oneapi/tbb"),
        Path("/opt/intel/oneapi/tcm"),
        Path("/opt/intel/oneapi/umf"),
    )
    found: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*/lib", "*/lib/intel64/gcc4.8", "*/opt/compiler/lib"):
            found.extend(str(path) for path in root.glob(pattern) if path.is_dir())
    return found


def _ensure_intel_runtime() -> None:
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    missing = [path for path in _intel_lib_dirs() if path not in current]
    if not missing:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([*missing, *current])
    # orig_argv is the real command line. sys.argv drops the text of `python -c`,
    # so restarting from sys.argv would come back as `python -c` with no program.
    os.execv(sys.orig_argv[0], sys.orig_argv)


_ensure_intel_runtime()


class WhisperCppEngine:
    def __init__(
        self,
        model_path: Path,
        use_gpu: bool,
        gpu_device: int,
        threads: int,
        lib_path: Path | None = None,
        backend_dir: Path | None = None,
        language: str = "en",
    ) -> None:
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.threads = threads
        self.language = language or "en"
        def _default_lib() -> Path:
            return project_root() / "native" / "build" / "libwispo_asr.so"

        self.lib_path = lib_path or _default_lib()
        self.backend_dir = backend_dir
        self._lib: ctypes.CDLL | None = None
        self._loaded = False
        self._warmed = False
        self.last_audio_seconds = 0.0
        self.last_infer_seconds = 0.0

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def warmed(self) -> bool:
        return self._warmed

    def set_language(self, language: str) -> None:
        self.language = language or "en"
        if self._lib is not None and hasattr(self._lib, "wf_set_language"):
            self._lib.wf_set_language(self.language.encode("utf-8"))

    def start(self) -> None:
        if self._loaded:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"model not found: {self.model_path}\n"
                f"Please ensure the model file is downloaded into models/.\n"
                f"You can download it with: bash scripts/download_model.sh"
            )
        if not self.lib_path.is_file():
            raise FileNotFoundError(
                "libwispo_asr.so is not built yet.\n"
                "whisper.cpp's Intel SYCL backend also needs oneMKL:\n"
                "  sudo apt install intel-oneapi-mkl-devel-2026.1 libze-dev\n"
                "then:\n"
                "  bash scripts/build_sycl.sh"
            )
        backend = self.backend_dir or _find_backend_dir()
        self.backend_dir = backend
        if self.use_gpu:
            os.environ["GGML_SYCL_DEVICE"] = str(self.gpu_device)
        self._lib = ctypes.CDLL(str(self.lib_path))
        self._bind()
        if hasattr(self._lib, "wf_set_language"):
            self._lib.wf_set_language(self.language.encode("utf-8"))
        backend_arg = str(backend).encode() if backend is not None else None
        rc = self._lib.wf_load(
            str(self.model_path).encode(),
            1 if self.use_gpu else 0,
            self.gpu_device,
            backend_arg,
        )
        if rc != 0:
            raise RuntimeError(f"whisper failed to load {self.model_path}")
        self._loaded = True

    def system_info(self) -> str:
        self.start()
        assert self._lib is not None
        info = self._lib.wf_system_info()
        return info.decode("utf-8", errors="replace") if info else ""

    def transcribe_chunk(self, audio: np.ndarray) -> str:
        return self._transcribe(audio, single_segment=False)

    def finalize(self, audio: np.ndarray) -> str:
        # A finished phrase may contain more than one sentence. Allow whisper
        # to split segments here.
        return self._transcribe(audio, single_segment=False)

    def warmup(self) -> None:
        """Prime the inference pipeline by running a silent forward pass.

        On GPU/SYCL backends, the first inference JIT-compiles device kernels and
        allocates scratch buffers (taking 7–16s). Running a dummy pass during
        startup ensures that the user's first spoken sentence experiences full
        inference speed without latency or dropped audio.
        """
        if self._warmed:
            return
        self.start()
        dummy = np.zeros(8000, dtype=np.float32)
        self._transcribe(dummy, single_segment=False)
        self.last_audio_seconds = 0.0
        self.last_infer_seconds = 0.0
        self._warmed = True

    def stop(self) -> None:
        return None

    def close(self) -> None:
        if self._lib is not None and self._loaded:
            self._lib.wf_free()
        self._loaded = False
        self._warmed = False

    def _transcribe(self, audio: np.ndarray, single_segment: bool) -> str:
        self.start()
        assert self._lib is not None
        samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""
        out = ctypes.create_string_buffer(_TEXT_CAP)
        started = time.perf_counter()
        rc = self._lib.wf_transcribe(
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(samples.size),
            self.threads,
            1 if single_segment else 0,
            out,
            _TEXT_CAP,
        )
        self.last_infer_seconds = time.perf_counter() - started
        self.last_audio_seconds = samples.size / 16000.0
        if rc != 0:
            raise RuntimeError(f"whisper transcription failed (code {rc})")
        return out.value.decode("utf-8", errors="replace").strip()

    def _bind(self) -> None:
        assert self._lib is not None
        self._lib.wf_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        self._lib.wf_load.restype = ctypes.c_int
        self._lib.wf_system_info.argtypes = []
        self._lib.wf_system_info.restype = ctypes.c_char_p
        self._lib.wf_transcribe.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.wf_transcribe.restype = ctypes.c_int
        self._lib.wf_free.argtypes = []
        self._lib.wf_free.restype = None
        if hasattr(self._lib, "wf_set_language"):
            self._lib.wf_set_language.argtypes = [ctypes.c_char_p]
            self._lib.wf_set_language.restype = ctypes.c_int


def _find_backend_dir() -> Path | None:
    build = project_root() / "third_party" / "whisper.cpp" / "build-sycl"
    if not build.is_dir():
        return None
    matches = sorted(build.rglob("libggml-sycl.so"))
    if not matches:
        return None
    return matches[0].parent
