"""Silence detection with ffmpeg's ``silencedetect`` filter, and silence cuts."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from . import ffmpeg
from .timeline import EPS, Cut

_NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_START = re.compile(r"silence_start:\s*" + _NUMBER)
_END = re.compile(r"silence_end:\s*" + _NUMBER)


def detect_silences(
    media: Path, *, threshold_db: float, min_duration: float, duration: float
) -> list[tuple[float, float]]:
    """Run ``silencedetect`` on the first audio stream; return ``(start, end)`` pairs."""
    log = ffmpeg.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-i", str(media),
            "-map", "0:a:0",
            "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
            "-f", "null", "-",
        ]
    )
    return parse_silencedetect(log, duration)


def parse_silencedetect(log: str, duration: float) -> list[tuple[float, float]]:
    """Extract silent spans from ``silencedetect`` log output.

    A silence that is still open when the log ends lasts until ``duration``.
    Times are clamped to ``[0, duration]``.
    """
    silences: list[tuple[float, float]] = []
    start = None
    for line in log.splitlines():
        if (match := _START.search(line)) is not None:
            start = min(max(0.0, float(match.group(1))), duration)
        elif (match := _END.search(line)) is not None and start is not None:
            silences.append((start, min(max(start, float(match.group(1))), duration)))
            start = None
    if start is not None and duration - start > EPS:
        silences.append((start, duration))
    return silences


def silence_cuts(
    silences: Iterable[tuple[float, float]],
    duration: float,
    *,
    min_silence: float = 0.6,
    padding: float = 0.15,
) -> list[Cut]:
    """Turn silent spans into cuts that leave ``padding`` seconds next to speech.

    Silences shorter than ``min_silence`` are kept. Leading and trailing
    silence is trimmed down to ``padding`` on the speech side only.
    """
    cuts = []
    for start, end in silences:
        if end - start < min_silence:
            continue
        cut_start = 0.0 if start <= EPS else start + padding
        cut_end = duration if end >= duration - EPS else end - padding
        if cut_end - cut_start > EPS:
            cuts.append(
                Cut(cut_start, cut_end, "silence", {"silence": [round(start, 3), round(end, 3)]})
            )
    return cuts
