"""Test doubles: a fake transcriber and helpers to write transcripts by hand."""

from __future__ import annotations

from pathlib import Path

from video_cleaner.transcribe import Word

WORD = 0.3  # seconds per word
GAP = 0.05  # seconds between words of one utterance


def utterance(start: float, text: str) -> list[Word]:
    """The words of ``text`` spoken from ``start``: WORD seconds each, GAP apart."""
    words = []
    t = start
    for token in text.split():
        words.append(Word(token, round(t, 3), round(t + WORD, 3)))
        t += WORD + GAP
    return words


def transcript(*utterances: tuple[float, str]) -> list[Word]:
    """``transcript((0.0, "Hello there."), (2.5, "Next sentence."))``"""
    return [word for start, text in utterances for word in utterance(start, text)]


class FakeTranscriber:
    """Implements the ``Transcriber`` protocol with a canned transcript."""

    name = "fake"

    def __init__(self, words: list[Word]) -> None:
        self.words = list(words)
        self.calls: list[Path] = []

    def transcribe(self, media: Path) -> list[Word]:
        self.calls.append(media)
        return list(self.words)
