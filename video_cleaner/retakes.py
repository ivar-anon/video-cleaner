"""Repeated-take detection on a word-level transcript.

A *take* is one attempt at saying a sentence. When the speaker stops and says
the sentence again, every earlier attempt is cut and only the last one is kept.

1. Words are grouped into sentences at terminal punctuation (. ! ? …) or at
   pauses of at least ``sentence_gap`` seconds.
2. Inside a sentence, a restart ("I want to, I want to show you") is found by
   looking for the sentence's opening words occurring again later on.
3. Consecutive sentences are compared; the earlier one is a retake of the
   later one when both tests below pass:

   * similarity: the fuzzy character-level ratio between the earlier take and
     the later take, or the beginning of the later take, is at least
     ``similarity``;
   * end alignment: the earlier take's last word also appears near the start
     of the later take. An abandoned take stops somewhere inside the sentence
     that follows, whereas parallel sentences ("I use the editor." / "I use
     the terminal.") differ precisely at the end.

4. The cut runs from the start of the abandoned take to the start of the take
   that follows. Speech-recognition timestamps are often a few tenths of a
   second off, so when the audio's pauses are known, each cut boundary is
   placed inside the real pause between the two words instead.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import NamedTuple

from .timeline import Cut
from .transcribe import Word

FILLERS = frozenset({"ah", "eh", "er", "erm", "hm", "hmm", "mm", "uh", "uhm", "um", "umm"})
"""Hesitation sounds ignored when comparing takes."""

ANCHOR_WORDS = 2
"""Number of opening words that must repeat to flag a restart inside a sentence."""

PREFIX_SLACK = 2
"""Extra words of the later take considered when matching an abandoned take."""

SAME_WORD_RATIO = 0.8
"""Character ratio at which two words count as the same word ("tapes"/"takes")."""

SNAP_TOLERANCE = 0.3
"""How far (s) a pause may lie outside the gap between two word timestamps and still be used."""

_SENTENCE_END = (".", "!", "?", "…")
_CLOSING = "\"')]}\u00bb\u201d\u2019"  # quotes and brackets that may follow a period
_NON_WORD = re.compile(r"[^\w']+")

Span = tuple[float, float]


class _Settings(NamedTuple):
    similarity: float
    min_words: int
    padding: float
    pauses: Sequence[Span]


def detect_retakes(
    words: Sequence[Word],
    *,
    similarity: float = 0.85,
    min_words: int = 2,
    sentence_gap: float = 1.0,
    padding: float = 0.15,
    pauses: Sequence[Span] = (),
) -> list[Cut]:
    """Return one ``"retake"`` cut per abandoned take, sorted by time.

    Args:
        words: the transcript, in time order.
        similarity: minimum fuzzy similarity (0-1) between two takes.
        min_words: abandoned takes with fewer (non-filler) words are ignored.
        sentence_gap: a pause of at least this many seconds ends a sentence.
        padding: seconds of pause kept next to the surrounding speech (never
            more than half of the pause).
        pauses: quiet spans of the audio, used to place cut boundaries more
            precisely than the word timestamps allow. Optional.
    """
    settings = _Settings(similarity, min_words, padding, sorted(pauses))
    cuts: list[Cut] = []
    last_takes: list[tuple[int, int, int]] = []  # (sentence start, last take start, end)
    for lo, hi in _sentence_bounds(words, sentence_gap):
        if not _tokens(words[lo:hi]):
            continue  # only fillers ("Um."): not a take; removed if it sits inside a retake cut
        inner, take_lo = _restarts_within(words, lo, hi, settings)
        cuts.extend(inner)
        last_takes.append((lo, take_lo, hi))

    for (_, lo, hi), (next_lo, next_take_lo, next_hi) in zip(last_takes, last_takes[1:]):
        score = _retake_score(
            _tokens(words[lo:hi]), _tokens(words[next_take_lo:next_hi]), settings
        )
        if score is not None:
            cuts.append(_cut(words, lo, hi, next_lo, words[next_take_lo:next_hi], score, settings))
    return sorted(cuts, key=lambda c: c.start)


def split_sentences(words: Sequence[Word], max_gap: float = 1.0) -> list[list[Word]]:
    """Group words into sentences at terminal punctuation or pauses >= ``max_gap``."""
    return [list(words[lo:hi]) for lo, hi in _sentence_bounds(words, max_gap)]


def normalize(word: str) -> str:
    """Lower-case a word and strip punctuation; filler sounds become ``""``."""
    token = unicodedata.normalize("NFKC", word).lower().replace("\u2019", "'")
    token = _NON_WORD.sub("", token).strip("'")
    return "" if token in FILLERS else token


def take_similarity(earlier: Sequence[str], later: Sequence[str]) -> float:
    """Fuzzy similarity (0-1) of ``earlier`` to ``later`` or to its beginning.

    Both arguments are normalised tokens. The beginning of ``later`` is tried
    at several lengths around ``len(earlier)`` so an abandoned take matches
    the complete sentence that replaces it.
    """
    if not earlier or not later:
        return 0.0
    text = " ".join(earlier)
    best = _ratio(text, " ".join(later))
    for n in range(max(1, len(earlier) - 1), min(len(later), len(earlier) + PREFIX_SLACK) + 1):
        best = max(best, _ratio(text, " ".join(later[:n])))
    return best


def ends_align(earlier: Sequence[str], later: Sequence[str]) -> bool:
    """True if the last word of ``earlier`` also occurs near the start of ``later``."""
    if not earlier or not later:
        return False
    last = earlier[-1]
    return any(_same_word(last, word) for word in later[: len(earlier) + PREFIX_SLACK])


def _sentence_bounds(words: Sequence[Word], max_gap: float) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    lo = 0
    for i, word in enumerate(words):
        if i > lo and word.start - words[i - 1].end >= max_gap:
            bounds.append((lo, i))
            lo = i
        if word.text.rstrip(_CLOSING).endswith(_SENTENCE_END):
            bounds.append((lo, i + 1))
            lo = i + 1
    if lo < len(words):
        bounds.append((lo, len(words)))
    return bounds


def _restarts_within(
    words: Sequence[Word], lo: int, hi: int, settings: _Settings
) -> tuple[list[Cut], int]:
    """Cut takes abandoned inside ``words[lo:hi]``; also return where the last take starts."""
    content = [i for i in range(lo, hi) if normalize(words[i].text)]
    tokens = [normalize(words[i].text) for i in content]
    cuts: list[Cut] = []
    take_lo, k = lo, 0  # current take: raw index / position in `content`
    while len(tokens) - k >= ANCHOR_WORDS:
        anchor = tokens[k : k + ANCHOR_WORDS]
        for j in range(k + 1, len(tokens) - ANCHOR_WORDS + 1):
            if tokens[j : j + ANCHOR_WORDS] != anchor:
                continue
            score = _retake_score(tokens[k:j], tokens[j:], settings)
            if score is not None:
                restart = content[j]
                cuts.append(
                    _cut(words, take_lo, restart, restart, words[restart:hi], score, settings)
                )
                take_lo, k = restart, j
                break
        else:
            break
    return cuts, take_lo


def _retake_score(
    earlier: Sequence[str], later: Sequence[str], settings: _Settings
) -> float | None:
    """Similarity score if ``earlier`` looks like an abandoned take of ``later``, else None."""
    if len(earlier) < max(1, settings.min_words) or not ends_align(earlier, later):
        return None
    score = take_similarity(earlier, later)
    return score if score >= settings.similarity else None


def _cut(
    words: Sequence[Word],
    lo: int,
    hi: int,
    next_lo: int,
    matched: Sequence[Word],
    score: float,
    settings: _Settings,
) -> Cut:
    """Cut ``words[lo:hi]`` and the pauses around it, up to ``words[next_lo]``.

    Each boundary goes inside the pause between the neighbouring words and
    leaves up to ``padding`` seconds of it next to the speech that is kept.
    """
    first, last, following = words[lo], words[hi - 1], words[next_lo]
    previous_end = words[lo - 1].end if lo > 0 else 0.0
    pad = settings.padding

    start = first.start - min(pad, max(first.start - previous_end, 0.0) / 2)
    end = following.start - min(pad, max(following.start - last.end, 0.0) / 2)
    before = _pause_between(previous_end, first.start, settings.pauses)
    after = _pause_between(last.end, following.start, settings.pauses)
    if before is not None:
        start = before[0] + min(pad, (before[1] - before[0]) / 2)
    if after is not None:
        end = after[1] - min(pad, (after[1] - after[0]) / 2)

    return Cut(
        start,
        max(start, end),
        "retake",
        {
            "removed_text": _join(words[lo:hi]),
            "matched_text": _join(matched),
            "similarity": round(score, 3),
        },
    )


def _pause_between(t0: float, t1: float, pauses: Sequence[Span]) -> Span | None:
    """The longest pause overlapping the gap between two word timestamps (with tolerance)."""
    near = [p for p in pauses if p[1] > t0 - SNAP_TOLERANCE and p[0] < t1 + SNAP_TOLERANCE]
    return max(near, key=lambda p: p[1] - p[0], default=None)


def _tokens(words: Sequence[Word]) -> list[str]:
    return [t for t in (normalize(w.text) for w in words) if t]


def _same_word(earlier: str, later: str) -> bool:
    # An abandoned take may end on a clipped word ("sh-" for "show").
    return (
        earlier == later
        or (len(earlier) >= 2 and later.startswith(earlier))
        or _ratio(earlier, later) >= SAME_WORD_RATIO
    )


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _join(words: Sequence[Word]) -> str:
    return " ".join(w.text for w in words)
