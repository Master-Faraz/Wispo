"""Choose SYCL on the Arc GPU, or the whisper.cpp CPU backend.

The Python process never contains Intel-specific math. It only asks two
questions: did `sycl-ls` list a Level Zero GPU, and did the user force a
device in the config? The compiled whisper library is what actually runs
on that device.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

_CACHE_FILE = Path.home() / ".local" / "state" / "wispo" / "device_cache.json"
_CACHE_TTL_S = 86400 * 7  # 7 days


@dataclass(frozen=True)
class DeviceInfo:
    # "sycl" or "cpu". This is the value passed to the whisper wrapper.
    backend: str
    # Human-readable name for the banner, for example "Intel(R) Arc(TM) Graphics".
    name: str
    use_gpu: bool

    @property
    def label(self) -> str:
        if self.backend == "sycl":
            return "Intel SYCL"
        return "CPU"


def parse_sycl_ls(text: str) -> str | None:
    """Return the GPU name from sycl-ls output, or None when no Level Zero GPU exists."""
    for line in text.splitlines():
        lowered = line.lower()
        if "level_zero:gpu" not in lowered and "[level_zero:gpu]" not in lowered:
            # sycl-ls prints the platform tag as "[level_zero:gpu]".
            if "level_zero" not in lowered or "gpu" not in lowered:
                continue
        # The device name sits between the runtime description and the version.
        # Example:
        # [level_zero:gpu][level_zero:0] Intel(R) ... Runtime ..., Intel(R) Arc(TM) Graphics 12.74.4 [1.14]
        if "," not in line:
            return "Intel Arc GPU"
        name = line.split(",")[-1].strip()
        # Drop the trailing driver version in square brackets.
        if " [" in name:
            name = name.split(" [", 1)[0].strip()
        # "Intel(R) Arc(TM) Graphics 12.74.4" -> keep the product, drop the build number
        # only when a version-like token is at the end.
        parts = name.split()
        if parts and parts[-1][:1].isdigit():
            name = " ".join(parts[:-1])
        return name or "Intel Arc GPU"
    return None


def _run_sycl_ls() -> str:
    if shutil.which("sycl-ls"):
        completed = subprocess.run(["sycl-ls"], check=False, capture_output=True, text=True)
        return completed.stdout + completed.stderr
    # icpx is installed, but a normal shell does not put it on PATH until
    # the oneAPI env script has been sourced.
    script = "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && sycl-ls"
    completed = subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True)
    return completed.stdout + completed.stderr


def _cached_gpu_name() -> str | None:
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            if time.time() - data.get("ts", 0) < _CACHE_TTL_S:
                return data.get("gpu_name")
    except Exception:
        pass
    gpu_name = parse_sycl_ls(_run_sycl_ls())
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({"gpu_name": gpu_name, "ts": time.time()}))
    except Exception:
        pass
    return gpu_name


def select_device(preference: str) -> DeviceInfo:
    preference = preference.lower()
    if preference not in {"auto", "sycl", "cpu"}:
        raise ValueError(f"device must be auto, sycl, or cpu (got {preference!r})")

    gpu_name = None
    if preference != "cpu":
        gpu_name = _cached_gpu_name()

    if preference == "cpu" or gpu_name is None:
        if preference == "sycl":
            raise RuntimeError(
                "device is set to sycl, but sycl-ls did not list a Level Zero GPU"
            )
        return DeviceInfo(backend="cpu", name="CPU", use_gpu=False)
    return DeviceInfo(backend="sycl", name=gpu_name, use_gpu=True)
