"""The only method the rest of the app needs from a text processor."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        """Return the text to insert. Called once, on the finished sentence."""
