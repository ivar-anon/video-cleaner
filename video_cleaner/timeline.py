"""Cuts, kept segments and the edit decision list (EDL).

All times are seconds on the source timeline. Everything in this module is
pure, so the editing logic can be tested without ffmpeg or a speech model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

EPS = 1e-6
"""Tolerance used when comparing times, in seconds."""

REASON_PRIORITY = {"retake": 0, "silence": 1}
"""When two cuts overlap, the reason with the lower number owns the overlap."""


@dataclass(frozen=True)
class Cut:
    """A span of the source to remove, and why."""

    start: float
    end: float
    reason: str
    info: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Segment:
    """A span of the source that is kept in the output."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class EditDecisionList:
    """What to cut, what to keep, and why.

    ``cuts`` are sorted and never overlap, so every removed second belongs to
    exactly one cut and ``removed_duration`` equals the sum of cut durations.
    """

    source_duration: float
    cuts: tuple[Cut, ...]
    keep: tuple[Segment, ...]

    @property
    def output_duration(self) -> float:
        return sum(segment.duration for segment in self.keep)

    @property
    def removed_duration(self) -> float:
        return self.source_duration - self.output_duration

    def summary(self) -> dict[str, dict[str, float]]:
        """Number of cuts and seconds removed, per reason."""
        stats: dict[str, dict[str, float]] = {}
        for cut in self.cuts:
            entry = stats.setdefault(cut.reason, {"count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += cut.duration
        return stats

    def to_dict(self, **metadata: Any) -> dict[str, Any]:
        """JSON-ready representation; times are rounded to milliseconds."""
        keep = []
        cursor = 0.0
        for segment in self.keep:
            keep.append(
                {
                    "start": _ms(segment.start),
                    "end": _ms(segment.end),
                    "output_start": _ms(cursor),
                    "output_end": _ms(cursor + segment.duration),
                }
            )
            cursor += segment.duration
        return {
            **metadata,
            "source_duration": _ms(self.source_duration),
            "output_duration": _ms(self.output_duration),
            "removed_duration": _ms(self.removed_duration),
            "summary": {
                reason: {"count": int(s["count"]), "seconds": _ms(s["seconds"])}
                for reason, s in self.summary().items()
            },
            "cuts": [
                {
                    "start": _ms(cut.start),
                    "end": _ms(cut.end),
                    "duration": _ms(cut.duration),
                    "reason": cut.reason,
                    **cut.info,
                }
                for cut in self.cuts
            ],
            "keep": keep,
        }


def build_edl(
    source_duration: float, cuts: Iterable[Cut], *, min_keep: float = 0.1
) -> EditDecisionList:
    """Resolve raw detections into a consistent edit decision list.

    Kept fragments shorter than ``min_keep`` seconds (slivers left between two
    nearby cuts) are folded into a neighbouring cut rather than rendered.
    """
    resolved = resolve_cuts(cuts, source_duration)
    resolved = _absorb_short_keeps(resolved, source_duration, min_keep)
    return EditDecisionList(
        source_duration, tuple(resolved), tuple(keep_segments(resolved, source_duration))
    )


def resolve_cuts(cuts: Iterable[Cut], duration: float) -> list[Cut]:
    """Clip cuts to ``[0, duration]``, remove overlaps, and sort them.

    Where cuts overlap, the cut whose reason has the higher priority (see
    ``REASON_PRIORITY``) keeps the overlapping time and the other one is
    trimmed, split or dropped. Between cuts of the same reason, the one that
    starts first wins.
    """

    def rank(cut: Cut) -> tuple[int, float]:
        return REASON_PRIORITY.get(cut.reason, len(REASON_PRIORITY)), cut.start

    accepted: list[Cut] = []
    for cut in sorted(cuts, key=rank):
        pieces = [(max(0.0, cut.start), min(duration, cut.end))]
        for other in accepted:
            pieces = [part for piece in pieces for part in _subtract(piece, other)]
        accepted.extend(replace(cut, start=s, end=e) for s, e in pieces if e - s > EPS)
    return sorted(accepted, key=lambda c: c.start)


def keep_segments(cuts: Iterable[Cut], duration: float) -> list[Segment]:
    """The parts of ``[0, duration]`` not covered by any cut, in order."""
    segments: list[Segment] = []
    cursor = 0.0
    for cut in sorted(cuts, key=lambda c: c.start):
        if cut.start - cursor > EPS:
            segments.append(Segment(cursor, cut.start))
        cursor = max(cursor, cut.end)
    if duration - cursor > EPS:
        segments.append(Segment(cursor, duration))
    return segments


def _subtract(piece: tuple[float, float], cut: Cut) -> list[tuple[float, float]]:
    start, end = piece
    if cut.end <= start or cut.start >= end:
        return [piece]
    return [p for p in ((start, cut.start), (cut.end, end)) if p[1] - p[0] > EPS]


def _absorb_short_keeps(cuts: list[Cut], duration: float, min_keep: float) -> list[Cut]:
    """Extend cuts over kept gaps shorter than ``min_keep`` (``cuts`` sorted, disjoint)."""
    if not cuts or min_keep <= 0:
        return cuts
    out: list[Cut] = []
    cursor = 0.0
    for cut in cuts:
        gap = cut.start - cursor
        if EPS < gap < min_keep:
            if out:
                out[-1] = replace(out[-1], end=cut.start)
            else:
                cut = replace(cut, start=cursor)
        out.append(cut)
        cursor = cut.end
    if EPS < duration - cursor < min_keep:
        out[-1] = replace(out[-1], end=duration)
    return out


def _ms(seconds: float) -> float:
    return round(seconds, 3)
