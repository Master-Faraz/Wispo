"""Dynamic decoder prompt builder for style priming and dictionary biasing."""

from __future__ import annotations

from typing import Sequence

from whisperflow.config import DictionaryEntry

DEFAULT_STYLE_ANCHOR = "Hello, world. Here is a clear transcription with proper punctuation and capitalization."


def build_initial_prompt(
    entries: Sequence[DictionaryEntry] | None = None,
    style_anchor: str = DEFAULT_STYLE_ANCHOR,
    include_style: bool = True,
    max_chars: int = 600,
) -> str:
    """Construct an initial_prompt string for whisper.cpp decoder.

    Whisper uses up to 224 tokens of initial_prompt to bias punctuation,
    casing, and custom domain vocabulary.
    """
    parts: list[str] = []
    if include_style and style_anchor.strip():
        parts.append(style_anchor.strip())

    if entries:
        valid_words = [
            e.replacement.strip()
            for e in entries
            if e.enabled and e.replacement.strip()
        ]
        if valid_words:
            seen: set[str] = set()
            unique_words = [w for w in valid_words if not (w.lower() in seen or seen.add(w.lower()))]
            parts.append(f"Vocabulary: {', '.join(unique_words)}.")

    full_prompt = " ".join(parts).strip()
    if len(full_prompt) > max_chars:
        full_prompt = full_prompt[:max_chars].rsplit(" ", 1)[0]
    return full_prompt
