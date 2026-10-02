"""Measurements Stage 1 uses to decide if a model is fast enough.

CPU and process RAM come from the Python process itself. The Arc 130T is an
integrated GPU, so its memory is the same system RAM the process already
reports. GPU activity is estimated from the i915 RC6 counter: RC6 time
advances while the GPU is idle, so busy time is wall time minus RC6 time.
"""

from __future__ import annotations

import resource
from dataclasses import dataclass
from pathlib import Path

import psutil


def _rc6_path() -> Path | None:
    matches = sorted(Path("/sys/class/drm").glob("card*/gt/gt0/rc6_residency_ms"))
    return matches[0] if matches else None


def _read_int(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def gpu_frequency_mhz() -> int | None:
    for path in sorted(Path("/sys/class/drm").glob("card*/gt/gt0/rps_act_freq_mhz")):
        value = _read_int(path)
        if value is not None:
            return value
    return None


@dataclass
class GpuSample:
    rc6_ms: int | None
    frequency_mhz: int | None


def sample_gpu() -> GpuSample:
    return GpuSample(rc6_ms=_read_int(_rc6_path()), frequency_mhz=gpu_frequency_mhz())


def gpu_busy_percent(before: GpuSample, after: GpuSample, elapsed_ms: float) -> float | None:
    if before.rc6_ms is None or after.rc6_ms is None or elapsed_ms <= 0:
        return None
    idle_ms = max(0, after.rc6_ms - before.rc6_ms)
    busy = 1.0 - (idle_ms / elapsed_ms)
    return max(0.0, min(100.0, busy * 100.0))


def process_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def peak_rss_mb() -> float:
    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def ram_total_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)
