"""Load the Stage 1 settings.

The YAML file is the place that names the model, the device policy, and the
size of each audio window. The ASR code reads this object. It does not open
the YAML itself, so a test can build a Config without a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    device: str = "default"
    # VAD looks at short fixed frames. 30 ms is long enough for a stable RMS
    # reading and short enough that the start of a word is not delayed much.
    frame_ms: int = 30


@dataclass(frozen=True)
class AsrConfig:
    backend: str = "whisper_cpp"
    model: str = "tiny.en"
    model_path: str = "models/ggml-tiny.en.bin"
    # auto: SYCL when sycl-ls lists a Level Zero GPU, otherwise CPU.
    device: str = "auto"
    quantization: str = "fp16"
    threads: int = 4
    language: str = "en"
    gpu_device: int = 0


@dataclass(frozen=True)
class StreamingConfig:
    # How often a window is sent to whisper while the person is still talking.
    chunk_ms: int = 1000
    # Audio repeated from the previous window so a word cut by the boundary
    # is heard in both calls. The accumulator deletes the repeated words.
    overlap_ms: int = 250
    # How much audio one whisper call sees. Longer than chunk_ms, so each
    # call re-hears the recent overlap instead of a clipped fragment.
    length_ms: int = 5000


@dataclass(frozen=True)
class VadConfig:
    # Absolute RMS floor. Speech quieter than this is treated as silence
    # even if the room tone drifts.
    rms_threshold: float = 0.015
    # Speech must also be this many times louder than the learned noise floor.
    noise_multiplier: float = 4.0
    # Keep the utterance open briefly after the voice drops, so the end of
    # the last word is not cut off.
    hangover_ms: int = 700
    # Audio kept before the VAD flips to speech. The first consonant often
    # arrives in the frame that crosses the threshold.
    preroll_ms: int = 300
    highpass_hz: float = 100.0


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = AudioConfig()
    asr: AsrConfig = AsrConfig()
    streaming: StreamingConfig = StreamingConfig()
    vad: VadConfig = VadConfig()

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        rooted = project_root() / path
        return rooted


def _section(data: dict, name: str, cls):
    raw = data.get(name) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a mapping")
    known = {field for field in cls.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown {name} settings: {', '.join(sorted(unknown))}")
    return cls(**raw)


def load_config(path: Path | None = None) -> Config:
    if path is None:
        default = project_root() / "whisperflow.yaml"
        path = default if default.is_file() else None
    if path is None:
        return Config()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping")
    return Config(
        audio=_section(data, "audio", AudioConfig),
        asr=_section(data, "asr", AsrConfig),
        streaming=_section(data, "streaming", StreamingConfig),
        vad=_section(data, "vad", VadConfig),
    )
