"""Turn overlapping window transcripts into one phrase.

Each whisper call hears audio that the previous call also heard. The new
text therefore repeats the end of the old text. This class keeps two views:

  committed  words that already agreed between the previous hypothesis and
             the new one
  partial    words the latest call added or revised

`text` is both of those joined. The CLI prints `text` on each partial line.
When the utterance ends, finish() folds the partial into the committed phrase.
"""

from __future__ import annotations


def _key(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalnum() or ch == "'")


def _words(text: str) -> list[str]:
    return [word for word in text.split() if word]


def _keys_match(k1: str, k2: str) -> bool:
    if k1 == k2:
        return True
    if len(k1) >= 4 and len(k2) >= 4:
        if k1.startswith(k2) or k2.startswith(k1):
            return True
        if abs(len(k1) - len(k2)) <= 1:
            diffs = sum(c1 != c2 for c1, c2 in zip(k1, k2)) + abs(len(k1) - len(k2))
            if diffs <= 1:
                return True
    return False


def _seq_match(seq1: list[str], seq2: list[str]) -> bool:
    if len(seq1) != len(seq2):
        return False
    return all(_keys_match(w1, w2) for w1, w2 in zip(seq1, seq2))


def merge_transcript(previous: str, incoming: str) -> str:
    prev = previous.strip()
    inc = incoming.strip()
    if not prev:
        return inc
    if not inc:
        return prev

    prev_words = _words(prev)
    inc_words = _words(inc)
    prev_keys = [_key(word) for word in prev_words]
    inc_keys = [_key(word) for word in inc_words]

    # The window grew but still starts with the same phrase ("hello" ->
    # "hello this is"). Keep the longer hypothesis instead of appending it.
    if inc_keys[: len(prev_keys)] == prev_keys:
        return inc

    # If the incoming text is completely contained in the previous text, keep previous
    prev_joined = " ".join(prev_keys)
    inc_joined = " ".join(inc_keys)
    if inc_joined in prev_joined:
        return prev

    max_k = min(len(prev_words), len(inc_words))
    overlap = 0
    # 1. Exact suffix-prefix match
    for k in range(max_k, 0, -1):
        if prev_keys[-k:] == inc_keys[:k]:
            overlap = k
            break
    if overlap > 0:
        return " ".join(prev_words + inc_words[overlap:])

    # 2. Fuzzy suffix-prefix match (handles slight acoustic token variation)
    fuzzy_overlap = 0
    for k in range(max_k, 0, -1):
        if _seq_match(prev_keys[-k:], inc_keys[:k]):
            fuzzy_overlap = k
            break
    if fuzzy_overlap > 0:
        return " ".join(prev_words[:-fuzzy_overlap] + inc_words)

    return f"{prev} {inc}"


class TranscriptAccumulator:
    def __init__(self) -> None:
        self.committed = ""
        self.partial = ""
        self.text = ""

    def merge(self, window_text: str) -> str:
        updated = merge_transcript(self.text, window_text)
        old_keys = [_key(word) for word in _words(self.text)]
        new_words = _words(updated)
        new_keys = [_key(word) for word in new_words]
        shared = 0
        limit = min(len(old_keys), len(new_keys))
        while shared < limit and old_keys[shared] == new_keys[shared] and old_keys[shared]:
            shared += 1
        if not self.text:
            self.committed = ""
            self.partial = updated
        else:
            self.committed = " ".join(new_words[:shared])
            self.partial = " ".join(new_words[shared:])
        self.text = updated
        return self.text

    def finish(self) -> str:
        phrase = self.text.strip()
        self.committed = phrase
        self.partial = ""
        self.text = phrase
        return phrase

    def reset(self) -> None:
        self.committed = ""
        self.partial = ""
        self.text = ""
