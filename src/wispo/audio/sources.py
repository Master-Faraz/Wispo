"""Resolve the configured audio device to an ALSA name and PipeWire target.

When device is 'auto', uses wpctl to select a PipeWire digital microphone
source. Some desktops route the ALSA default device to a loopback or
Bluetooth sink that returns silence; auto selection avoids that by
explicitly targeting the built-in digital microphone when one is detected.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class MicTarget:
    # ALSA device passed to snd_pcm_open. "pipewire" plus a target records
    # one specific source. "default" is whatever the session calls default.
    alsa_device: str
    pipewire_target: str | None
    description: str


_SOURCE_LINE = re.compile(r"(\d+)\.\s+(.+?)(?:\s+\[vol:.*)?$")
_NODE_NAME = re.compile(r'node\.name\s*=\s*"([^"]+)"')


def parse_wpctl_sources(status: str) -> list[tuple[int, str]]:
    """Return (id, description) for sources in the Audio section only."""
    in_audio = False
    in_sources = False
    found: list[tuple[int, str]] = []
    for raw in status.splitlines():
        line = raw.strip()
        if line == "Audio":
            in_audio = True
            in_sources = False
            continue
        if in_audio and line == "Video":
            break
        if not in_audio:
            continue
        if "Sources:" in line:
            in_sources = True
            continue
        if in_sources and any(
            label in line for label in ("Filters:", "Sinks:", "Streams:", "Settings")
        ):
            in_sources = False
            continue
        if not in_sources:
            continue
        match = _SOURCE_LINE.search(line)
        if match is None:
            continue
        description = match.group(2).strip().lstrip("* ").strip()
        if "monitor" in description.lower():
            continue
        found.append((int(match.group(1)), description))
    return found


def choose_source(sources: list[tuple[int, str]]) -> tuple[int, str] | None:
    if not sources:
        return None
    for source_id, description in sources:
        if "digital microphone" in description.lower():
            return source_id, description
    for source_id, description in sources:
        if "microphone" in description.lower():
            return source_id, description
    return sources[0]


def _node_name(source_id: int) -> str | None:
    completed = subprocess.run(
        ["wpctl", "inspect", str(source_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    match = _NODE_NAME.search(completed.stdout)
    return match.group(1) if match else None


def resolve_mic(device: str) -> MicTarget:
    """Turn the config device into an ALSA name and an optional PipeWire target."""
    if device not in {"auto", "default", "pipewire"}:
        return MicTarget("pipewire", device, device)
    if device == "auto":
        completed = subprocess.run(
            ["wpctl", "status"],
            check=False,
            capture_output=True,
            text=True,
        )
        chosen = choose_source(parse_wpctl_sources(completed.stdout))
        if chosen is not None:
            source_id, description = chosen
            node = _node_name(source_id)
            if node:
                return MicTarget("pipewire", node, description)
    alsa = "pipewire" if device == "pipewire" else "default"
    return MicTarget(alsa, None, device)
