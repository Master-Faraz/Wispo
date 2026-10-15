"""Load the Stage 1 settings.

The YAML file is the place that names the model, the device policy, and the
size of each audio window. The ASR code reads this object. It does not open
the YAML itself, so a test can build a Config without a file.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def runtime_data_root() -> Path | None:
    """Return the per-user data directory selected by a package launcher.

    Source checkouts deliberately keep their configuration and models beside the
    project. Installed packages set ``WISPO_DATA_DIR`` so a normal desktop user
    never has to write into the root-owned application directory.
    """
    raw = os.environ.get("WISPO_DATA_DIR", "").strip()
    return Path(raw).expanduser() if raw else None


def config_path(create: bool = False) -> Path:
    """Return the active settings file, optionally creating its user copy."""
    data_root = runtime_data_root()
    if data_root is None:
        return project_root() / "wispo.yaml"

    target = data_root / "wispo.yaml"
    if create and not target.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        template = project_root() / "wispo.yaml"
        if template.is_file():
            shutil.copyfile(template, target)
        else:
            target.write_text("{}\n", encoding="utf-8")
    return target


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
class DictionaryEntry:
    phrase: str
    replacement: str
    enabled: bool = True


@dataclass(frozen=True)
class HotkeyConfig:
    # Each name must be held at once. Left and right Ctrl both count as ctrl.
    keys: tuple[str, ...] = ("ctrl", "super")


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = AudioConfig()
    asr: AsrConfig = AsrConfig()
    streaming: StreamingConfig = StreamingConfig()
    vad: VadConfig = VadConfig()
    dictionary: tuple[DictionaryEntry, ...] = ()
    hotkey: HotkeyConfig = HotkeyConfig()

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        data_root = runtime_data_root()
        if data_root is not None:
            return data_root / path
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
        wispo_default = config_path()
        # A first packaged launch uses the root-owned template until the user
        # saves settings. Saving creates the writable per-user copy.
        if not wispo_default.is_file() and runtime_data_root() is not None:
            wispo_default = project_root() / "wispo.yaml"
        if wispo_default.is_file():
            path = wispo_default
        else:
            path = None
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
        dictionary=_dictionary(data),
        hotkey=_hotkey(data),
    )


def save_user_settings(
    path: Path,
    model: str,
    entries: tuple[DictionaryEntry, ...] | list[DictionaryEntry],
    hotkey: tuple[str, ...] | list[str] | None = None,
    audio_device: str | None = None,
    vad_settings: dict | None = None,
    threads: int | None = None,
    device: str | None = None,
    gpu_device: int | None = None,
    language: str | None = None,
    streaming_settings: dict | None = None,
) -> None:
    """Update the model, dictionary, hotkey, audio, and hardware settings in the YAML file.

    Other sections stay as they are. Comments in the file are not kept,
    because PyYAML rewrites the mapping. The running transcription process
    reads this file only at startup.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping")
    asr = data.get("asr") or {}
    if not isinstance(asr, dict):
        raise TypeError("asr must be a mapping")
    asr["model"] = model
    asr["model_path"] = f"models/ggml-{model}.bin"
    if threads is not None:
        asr["threads"] = int(threads)
    if device is not None:
        asr["device"] = str(device)
    if gpu_device is not None:
        asr["gpu_device"] = int(gpu_device)
    if language is not None:
        asr["language"] = str(language)
    data["asr"] = asr
    data["dictionary"] = [
        {"phrase": entry.phrase, "replacement": entry.replacement, "enabled": entry.enabled}
        for entry in entries
        if entry.phrase.strip()
    ]
    if hotkey is not None:
        data["hotkey"] = {"keys": [str(key) for key in hotkey]}
    if audio_device is not None:
        audio = data.get("audio") or {}
        if not isinstance(audio, dict):
            audio = {}
        audio["device"] = audio_device
        data["audio"] = audio
    if vad_settings is not None:
        vad = data.get("vad") or {}
        if not isinstance(vad, dict):
            vad = {}
        vad.update(vad_settings)
        data["vad"] = vad
    if streaming_settings is not None:
        streaming = data.get("streaming") or {}
        if not isinstance(streaming, dict):
            streaming = {}
        streaming.update(streaming_settings)
        data["streaming"] = streaming
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _hotkey(data: dict) -> HotkeyConfig:
    raw = data.get("hotkey") or {}
    if not raw:
        return HotkeyConfig()
    if not isinstance(raw, dict):
        raise TypeError("hotkey must be a mapping")
    unknown = set(raw) - {"keys"}
    if unknown:
        raise ValueError(f"unknown hotkey settings: {', '.join(sorted(unknown))}")
    keys = raw.get("keys") or ["ctrl", "super"]
    if isinstance(keys, str):
        keys = [part.strip() for part in keys.replace("+", " ").split() if part.strip()]
    if not isinstance(keys, list) or not keys:
        raise TypeError("hotkey.keys must be a list of key names")
    names = tuple(str(key).lower() for key in keys)
    from wispo.hotkey.linux import codes_for

    for name in names:
        codes_for(name)
    return HotkeyConfig(names)


def _dictionary(data: dict) -> tuple[DictionaryEntry, ...]:
    raw = data.get("dictionary") or []
    if not isinstance(raw, list):
        raise TypeError("dictionary must be a list")
    entries: list[DictionaryEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("dictionary entries must be mappings")
        unknown = set(item) - {"phrase", "replacement", "enabled"}
        if unknown:
            raise ValueError(f"unknown dictionary settings: {', '.join(sorted(unknown))}")
        if "phrase" not in item or "replacement" not in item:
            raise ValueError("dictionary entries need phrase and replacement")
        entries.append(
            DictionaryEntry(
                phrase=str(item["phrase"]),
                replacement=str(item["replacement"]),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return tuple(entries)
