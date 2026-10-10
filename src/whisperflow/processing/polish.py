"""Output text formatter and polish filter before desktop insertion."""

from __future__ import annotations

import re

_ARTIFACT_RE = re.compile(r"(\[.*?\]|\(.*?\))")
_MULTISPACE_RE = re.compile(r"[ \t]+")


def polish_transcript(
    text: str,
    auto_capitalize: bool = True,
    strip_artifacts: bool = True,
) -> str:
    """Clean and polish finished sentence for desktop insertion."""
    if not text:
        return ""
    result = text
    if strip_artifacts:
        result = _ARTIFACT_RE.sub("", result)
    result = _MULTISPACE_RE.sub(" ", result).strip()
    if not result:
        return ""
    if auto_capitalize and result:
        for i, char in enumerate(result):
            if char.isalpha():
                result = result[:i] + char.upper() + result[i + 1 :]
                break
    return result
