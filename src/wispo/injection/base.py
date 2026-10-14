"""The only method the rest of the app needs from an injector."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextInjector(ABC):
    def open(self) -> None:
        """Prepare the input channel once. insert() reuses it."""

    @abstractmethod
    def insert(self, text: str) -> None:
        """Type text into the currently focused application."""

    def close(self) -> None:
        """Release the input channel. Safe to call more than once."""
