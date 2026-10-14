"""Whisper model files the settings window can download.

The default stays base.en. A name is only usable after its ggml file is in
models/. OpenAI models come from the whisper.cpp Hugging Face repo. Distil
models come from the official distil-whisper ggml files.
"""

from __future__ import annotations

import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from wispo.config import project_root

_WHISPER_CPP = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


def _whisper_cpp(name: str) -> str:
    return f"{_WHISPER_CPP}/ggml-{name}.bin"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    size: str
    url: str


# Sizes are the verified published and on-disk file sizes, shown before download.
CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("tiny.en", "75 MB", _whisper_cpp("tiny.en")),
    ModelSpec("base.en", "142 MB", _whisper_cpp("base.en")),
    ModelSpec("small.en", "466 MB", _whisper_cpp("small.en")),
    ModelSpec("medium.en", "1.5 GB", _whisper_cpp("medium.en")),
    ModelSpec("large-v3-turbo-q5_0", "548 MB", _whisper_cpp("large-v3-turbo-q5_0")),
    ModelSpec("large-v3-turbo", "1.5 GB", _whisper_cpp("large-v3-turbo")),
    ModelSpec(
        "distil-small.en",
        "321 MB",
        "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-distil-small.en.bin",
    ),
    ModelSpec(
        "distil-medium.en",
        "758 MB",
        "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin",
    ),
    ModelSpec(
        "distil-large-v3",
        "1.5 GB",
        "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
    ),
    ModelSpec(
        "distil-large-v3.5",
        "1.45 GB",
        "https://huggingface.co/distil-whisper/distil-large-v3.5-ggml/resolve/main/ggml-model.bin",
    ),
)

DEFAULT_MODEL = "base.en"


def model_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in CATALOG)


def catalog_size(name: str) -> str:
    return _spec(name).size


def _spec(name: str) -> ModelSpec:
    for spec in CATALOG:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown model {name!r}")


def model_path(name: str, root: Path | None = None) -> Path:
    if name not in model_names():
        raise ValueError(f"unknown model {name!r}")
    return (root or project_root()) / "models" / f"ggml-{name}.bin"


def installed_models(root: Path | None = None) -> list[tuple[str, int]]:
    """Catalog models that already have a file, with the size in bytes."""
    root = root or project_root()
    found: list[tuple[str, int]] = []
    for name in model_names():
        path = model_path(name, root)
        if path.is_file():
            found.append((name, path.stat().st_size))
    return found


def download_model(
    name: str,
    root: Path | None = None,
    progress=None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Download one ggml file. progress(written, total) may be called often."""
    dest = model_path(name, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_name(dest.name + ".partial")
    request = urllib.request.Request(
        _spec(name).url,
        headers={"User-Agent": "wispo"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Download cancelled by user")
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if progress is not None:
                    progress(written, total)
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Download cancelled by user")
        if written == 0:
            raise RuntimeError(f"download of {name} was empty")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest
