"""Thin wrappers around the ffmpeg and ffprobe command-line tools."""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .timeline import Segment

log = logging.getLogger(__name__)

FADE = 0.01
"""Seconds of audio fade at each edit point, to avoid clicks."""

AUDIO_BLOCK = 64
"""Audio is cut in blocks of this many samples (1.3 ms at 48 kHz)."""

_INLINE_GRAPH_LIMIT = 100_000
"""Longer filter graphs are passed through a file (Linux caps one argument at 128 KiB)."""


class FFmpegError(RuntimeError):
    """ffmpeg or ffprobe is missing or failed."""


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_video: bool
    has_audio: bool
    sample_rate: int = 48_000
    """Of the first audio stream."""


def ensure_available() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FFmpegError(f"{' and '.join(missing)} not found on PATH; please install ffmpeg")


def run(cmd: Sequence[str], *, capture: bool = True) -> str:
    """Run a command and return its stderr (where ffmpeg writes its log)."""
    log.debug("$ %s", shlex.join(cmd))
    proc = subprocess.run(
        list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE if capture else None, text=True
    )
    if proc.returncode != 0:
        detail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise FFmpegError(f"{cmd[0]} exited with status {proc.returncode}\n{detail}".rstrip())
    return proc.stderr or ""


def probe(path: Path) -> MediaInfo:
    """Duration and stream types of a media file (cover art is not video)."""
    log.debug("probing %s", path)
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-of", "json",
            "-show_entries",
            "format=duration:stream=codec_type,sample_rate:stream_disposition=attached_pic",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe could not read {path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    try:
        duration = float(data["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise FFmpegError(f"ffprobe reported no duration for {path}") from exc
    streams = data.get("streams", [])
    has_video = any(
        s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        for s in streams
    )
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    sample_rate = int(audio[0].get("sample_rate") or 48_000) if audio else 48_000
    return MediaInfo(duration, has_video, bool(audio), sample_rate)


def build_filter_graph(
    segments: Sequence[Segment], *, has_video: bool, sample_rate: int = 48_000
) -> str:
    """A single-pass ffmpeg filter graph that keeps only ``segments``.

    The input is streamed once, so memory use does not grow with the length
    of the video or the number of cuts:

    * audio: split into blocks of ``AUDIO_BLOCK`` samples, ``aselect`` keeps
      the blocks inside the segments, ``volume`` applies a ``FADE`` at each
      edit point, and ``asetpts`` makes the kept samples contiguous;
    * video: ``select`` keeps the frames inside the segments, and ``setpts``
      moves each one back by the time removed before it, which preserves the
      original frame timing (variable frame rate included).

    Segment boundaries are first snapped to the audio block grid, so audio
    and video lose exactly the same time and cannot drift apart however many
    cuts there are. A block is kept when its centre lies inside a segment.

    "Which segment contains t?" is written as a balanced tree of ``if()``
    over the segment start times, so each frame costs O(log n) comparisons.
    """
    block = AUDIO_BLOCK / sample_rate
    snapped = [
        Segment(round(s.start / block) * block, round(s.end / block) * block) for s in segments
    ]
    snapped = [s for s in snapped if s.duration > 0]
    if not snapped:
        raise ValueError("nothing to render: no segments to keep")
    removed_before = []
    output_time = 0.0
    for seg in snapped:
        removed_before.append(seg.start - output_time)
        output_time += seg.duration

    # A block starting at t is kept when from[i] <= t < until[i], i.e. its centre is inside.
    from_ = [f"{s.start - block / 2:.7f}" for s in snapped]
    until = [f"{s.end - block / 2:.7f}" for s in snapped]
    starts = [f"{s.start:.7f}" for s in snapped]
    ends = [f"{s.end:.7f}" for s in snapped]
    keep = _lookup("t", from_, lambda i: f"gte(t,{from_[i]})*lt(t,{until[i]})")
    fade = _lookup(
        "t", from_, lambda i: f"min(1,min((t-{starts[i]})/{FADE},({ends[i]}-t)/{FADE}))"
    )
    chains = []
    if has_video:
        shift = _lookup("T", from_, lambda i: f"{removed_before[i]:.7f}")
        chains.append(f"[0:v:0]select='{keep}',setpts='PTS-({shift})/TB'[outv]")
    chains.append(
        f"[0:a:0]asetnsamples=n={AUDIO_BLOCK}:p=0,aselect='{keep}',"
        f"volume='{fade}':eval=frame,asetpts=N/SR/TB[outa]"
    )
    return ";\n".join(chains)


def _lookup(var: str, starts: Sequence[str], leaf: Callable[[int], str]) -> str:
    """Expression equal to ``leaf(i)`` for the last segment ``i`` starting at or before ``var``."""

    def tree(lo: int, hi: int) -> str:
        if hi - lo == 1:
            return leaf(lo)
        mid = (lo + hi) // 2
        return f"if(lt({var},{starts[mid]}),{tree(lo, mid)},{tree(mid, hi)})"

    return tree(0, len(starts))


def render(
    src: Path,
    dst: Path,
    segments: Sequence[Segment],
    *,
    has_video: bool,
    sample_rate: int = 48_000,
    crf: int = 20,
    preset: str = "veryfast",
    quiet: bool = False,
) -> None:
    """Render the kept ``segments`` of ``src`` into ``dst``.

    Video is re-encoded with libx264 and audio with AAC, so every cut is
    frame-accurate. Audio-only inputs get ffmpeg's default codec for the
    output container.
    """
    graph = build_filter_graph(segments, has_video=has_video, sample_rate=sample_rate)
    with tempfile.TemporaryDirectory(prefix="video-cleaner-") as tmp:
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(src)]
        if len(graph) < _INLINE_GRAPH_LIMIT:
            cmd += ["-filter_complex", graph]
        else:
            script = Path(tmp, "filter_graph.txt")
            script.write_text(graph, encoding="utf-8")
            version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
            major = ffmpeg_major_version(version)
            # ffmpeg 7 replaced -filter_complex_script with the generic "-/option file".
            legacy = major is not None and major < 7
            cmd += ["-filter_complex_script" if legacy else "-/filter_complex", str(script)]
        if has_video:
            cmd += ["-map", "[outv]", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-map", "[outa]", "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-map", "[outa]"]
        if dst.suffix.lower() in {".mp4", ".m4v", ".m4a", ".mov"}:
            cmd += ["-movflags", "+faststart"]
        cmd += ["-loglevel", "error"] + ([] if quiet else ["-stats"]) + [str(dst)]
        run(cmd, capture=quiet)


def ffmpeg_major_version(version_output: str) -> int | None:
    """Major version from ``ffmpeg -version`` output; None for git builds."""
    match = re.search(r"ffmpeg version n?(\d+)\.", version_output)
    return int(match.group(1)) if match else None
