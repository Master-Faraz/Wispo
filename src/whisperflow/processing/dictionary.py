"""Replace spoken phrases in the finished sentence.

Matching is case-insensitive and stops at word boundaries, so "git" does not
change "github". A longer phrase is preferred over a shorter one that starts
at the same word. Punctuation after a phrase is left in place. The work runs
once, after Whisper has finished, not on each partial line.
"""

from __future__ import annotations

import re

from whisperflow.config import DictionaryEntry
from whisperflow.processing.base import TextProcessor

_TOKEN = re.compile(r"\w+|\W+")
_WORD = re.compile(r"\w+\Z")


class DictionaryProcessor(TextProcessor):
    def __init__(self, entries: tuple[DictionaryEntry, ...] | list[DictionaryEntry]) -> None:
        phrases: list[tuple[tuple[str, ...], str]] = []
        for entry in entries:
            if not entry.enabled:
                continue
            words = tuple(entry.phrase.split())
            if not words or any(not _WORD.fullmatch(word) for word in words):
                continue
            phrases.append((tuple(word.casefold() for word in words), entry.replacement))
        phrases.sort(key=lambda item: len(item[0]), reverse=True)
        self._phrases = tuple(phrases)

    def process(self, text: str) -> str:
        if not text or not self._phrases:
            return text
        parts = _TOKEN.findall(text)
        word_at = [index for index, part in enumerate(parts) if _WORD.fullmatch(part)]
        pieces: list[str] = []
        cursor = 0
        word_index = 0
        while word_index < len(word_at):
            match = self._match_at(parts, word_at, word_index)
            if match is None:
                word_index += 1
                continue
            size, replacement = match
            start = word_at[word_index]
            end = word_at[word_index + size - 1]
            pieces.extend(parts[cursor:start])
            pieces.append(replacement)
            cursor = end + 1
            word_index += size
        pieces.extend(parts[cursor:])
        return "".join(pieces)

    def _match_at(self, parts: list[str], word_at: list[int], word_index: int):
        for words, replacement in self._phrases:
            size = len(words)
            if word_index + size > len(word_at):
                continue
            if not _words_match(parts, word_at, word_index, words):
                continue
            return size, replacement
        return None


def _words_match(parts: list[str], word_at: list[int], word_index: int, words: tuple[str, ...]) -> bool:
    for offset, expected in enumerate(words):
        part_index = word_at[word_index + offset]
        if parts[part_index].casefold() != expected:
            return False
        if offset == 0:
            continue
        previous = word_at[word_index + offset - 1]
        between = parts[previous + 1 : part_index]
        if not between or any(not piece.isspace() for piece in between):
            return False
    return True
