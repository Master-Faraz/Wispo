"""The ASR surface the rest of the app is allowed to call.

Stage 2 and stage 3 add a hotkey and an optional LLM. Those layers call
transcribe_chunk() and finalize(). They do not import whisper.cpp.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ASREngine(Protocol):
    def start(self) -> None:
        """Load the model once. Later recordings reuse it."""

    def transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe one window. The model stays loaded."""

    def finalize(self, audio: np.ndarray) -> str:
        """Transcribe a finished utterance. Still does not unload the model."""

    def stop(self) -> None:
        """End a recording. Must not unload the model."""
