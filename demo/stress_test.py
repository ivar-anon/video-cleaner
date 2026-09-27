#!/usr/bin/env python3
"""Stress test: render a long video with many cuts; check memory and A/V sync.

In the synthetic clip a tone and a white frame switch on together for 0.8 s
every 1.6 s, so every pause becomes a cut. The clip is cleaned with
``--no-retakes``, then the script reports the wall time, the peak memory of
the child processes (ffmpeg included) and the offset between video and audio
onsets at the start, middle and end of the source and of the result.

Usage (Linux or macOS; needs ffmpeg):
    python3 demo/stress_test.py --minutes 20
"""

from __future__ import annotations

import argparse
import bisect
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

PERIOD, ON = 1.6, 0.8


def make_clip(path: Path, seconds: float) -> None:
    gate = f"lt(mod(t,{PERIOD}),{ON})"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"aevalsrc='0.4*sin(2*PI*300*t)*{gate}':s=48000:d={seconds}",
            "-f", "lavfi", "-i", f"color=c=black:s=320x180:r=30:d={seconds},"
            f"drawbox=w=iw:h=ih:color=white:t=fill:enable='{gate}'",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )


def onsets(path: Path) -> tuple[list[float], list[float]]:
    """Times at which the picture turns white and the tone starts."""
    stats = subprocess.run(
        ["ffprobe", "-v", "error", "-f", "lavfi", f"movie={path},signalstats",
         "-show_entries", "frame=pts_time:frame_tags=lavfi.signalstats.YAVG", "-of", "csv=p=0"],
        capture_output=True, text=True, check=True,
    ).stdout
    frames = [(float(t), float(y)) for t, y, *_ in (row.split(",") for row in stats.split())]
    video = [t for (t, y), (_, prev) in zip(frames[1:], frames) if y > 128 >= prev]
    if frames and frames[0][1] > 128:
        video.insert(0, frames[0][0])
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a",
         "-af", "silencedetect=noise=-30dB:d=0.05", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    audio = [float(t) for t in re.findall(r"silence_end: ([\d.]+)", log)]
    return video, audio


def sync_report(path: Path) -> str:
    video, audio = onsets(path)
    offsets = []
    for a in audio:
        i = bisect.bisect_left(video, a)
        nearest = min(video[max(0, i - 1) : i + 1], key=lambda v: abs(v - a))
        if abs(nearest - a) < PERIOD / 4:  # ignore an onset cut off at the very end
            offsets.append(nearest - a)
    n = len(offsets)
    parts = {
        "start": offsets[:50],
        "middle": offsets[n // 2 - 25 : n // 2 + 25],
        "end": offsets[-50:],
    }
    return ", ".join(f"{k} {1000 * sum(v) / len(v):+.1f} ms" for k, v in parts.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--out", type=Path, default=Path("demo/output/stress"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src, dst = args.out / "stress.mp4", args.out / "stress.clean.mp4"

    print(f"building a {args.minutes:g}-minute clip ...")
    make_clip(src, args.minutes * 60)
    started = time.monotonic()
    subprocess.run(
        [sys.executable, "-m", "video_cleaner", str(src), "-o", str(dst), "--no-retakes", "-q"],
        check=True,
    )
    elapsed = time.monotonic() - started
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_mb = peak / 1024 / (1024 if sys.platform == "darwin" else 1)  # bytes on macOS

    cuts = len(re.findall(r'"reason"', dst.with_suffix(".edl.json").read_text()))
    print(f"cleaned in {elapsed:.1f} s, {cuts} cuts, child processes peaked at {peak_mb:.0f} MB")
    print(f"video minus audio onset, source: {sync_report(src)}")
    print(f"video minus audio onset, result: {sync_report(dst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
