#!/usr/bin/env python3
"""Build the synthetic talking-head clip used by the README demo.

The speech comes from espeak-ng; the picture is a plain ffmpeg background with
the current caption and the source time burnt in, so jumps are visible after
cleaning. The script has long pauses and one abandoned take.

Usage:
    python3 demo/make_clip.py demo/output/talking_head.mp4

Next to the clip it writes ``<clip>.transcript.json``: a word-level transcript
derived from the script (line times are exact, word times inside a line are
interpolated). It lets the demo run without a Whisper model:

    python3 -m video_cleaner demo/output/talking_head.mp4 \\
        --transcript demo/output/talking_head.transcript.json

Requires espeak-ng and ffmpeg (built with libfreetype/fontconfig for drawtext).
"""

from __future__ import annotations

import argparse
import array
import json
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

RATE = 22050  # espeak-ng's native sample rate
VOICE = "en-us"
WORDS_PER_MINUTE = 150
NOISE_RMS = 40  # about -58 dBFS of room noise, so pauses are not digital silence
VOICED = 300  # |sample| above this counts as speech when trimming espeak's own silence
MARGIN = 0.02  # seconds kept around the speech of each synthesized line
SEED = 7

# (pause before the line in seconds, spoken text)
SCRIPT = [
    (1.5, "Hi, and welcome to this short demo."),
    (1.8, "Today I want to show you"),  # abandoned take
    (1.2, "Today I want to show you how this tool cleans up a recording."),
    (2.2, "The first step finds the long pauses."),
    (0.4, "The second step finds the repeated takes."),
    (2.5, "Everything else is left exactly as it was."),
    (1.6, "Thanks for watching."),
]
TAIL = 2.0  # seconds of silence after the last line


def synthesize(text: str, wav_path: Path) -> array.array:
    subprocess.run(
        ["espeak-ng", "-v", VOICE, "-s", str(WORDS_PER_MINUTE), "-w", str(wav_path), text],
        check=True,
    )
    with wave.open(str(wav_path)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise SystemExit(f"unexpected espeak-ng output format in {wav_path}")
        samples = array.array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder == "big":
        samples.byteswap()
    voiced = [i for i, s in enumerate(samples) if abs(s) > VOICED]
    if not voiced:
        return samples
    margin = int(MARGIN * RATE)
    return samples[max(0, voiced[0] - margin) : voiced[-1] + margin]


def silence(seconds: float) -> array.array:
    return array.array("h", [0]) * round(seconds * RATE)


def build_audio(tmp: Path) -> tuple[array.array, list[dict]]:
    audio = array.array("h")
    lines = []
    for i, (pause, text) in enumerate(SCRIPT):
        audio.extend(silence(pause))
        start = len(audio) / RATE
        audio.extend(synthesize(text, tmp / f"line{i}.wav"))
        lines.append({"start": round(start, 3), "end": round(len(audio) / RATE, 3), "text": text})
    audio.extend(silence(TAIL))
    rng = random.Random(SEED)
    for i, sample in enumerate(audio):
        audio[i] = max(-32768, min(32767, sample + round(rng.gauss(0, NOISE_RMS))))
    return audio, lines


def script_words(lines: list[dict]) -> list[dict]:
    """Spread each line's words over its duration, proportionally to their length."""
    words = []
    for line in lines:
        tokens = line["text"].split()
        weights = [len(t) + 1 for t in tokens]
        span = line["end"] - line["start"]
        t = line["start"]
        for token, weight in zip(tokens, weights):
            step = span * weight / sum(weights)
            words.append({"text": token, "start": round(t, 3), "end": round(t + step, 3)})
            t += step
    return words


def render_video(wav: Path, out: Path, lines: list[dict], duration: float, tmp: Path) -> None:
    filters = [
        "drawtext=text='video-cleaner demo (synthetic clip)':x=40:y=36"
        ":fontsize=26:fontcolor=white@0.55",
        "drawtext=text='source time %{pts\\:hms}':x=40:y=h-64:fontsize=28:fontcolor=white@0.85",
    ]
    for i, line in enumerate(lines):
        caption = tmp / f"caption{i}.txt"
        caption.write_text(line["text"], encoding="utf-8")
        filters.append(
            f"drawtext=textfile='{caption}':x=(w-text_w)/2:y=(h-text_h)/2:fontsize=27"
            f":fontcolor=white:enable='between(t,{line['start']:.3f},{line['end']:.3f})'"
        )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=0x1f2937:s=960x540:r=30:d={duration:.3f}",
            "-i", str(wav),
            "-filter_complex", f"[0:v]{','.join(filters)}[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
            "-movflags", "+faststart", str(out),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, help="clip to create (e.g. demo/output/clip.mp4)")
    args = parser.parse_args()
    for tool in ("espeak-ng", "ffmpeg"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is required (e.g. `apt install {tool}`)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="video-cleaner-demo-") as tmp_dir:
        tmp = Path(tmp_dir)
        audio, lines = build_audio(tmp)
        wav = tmp / "speech.wav"
        with wave.open(str(wav), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            if sys.byteorder == "big":
                audio.byteswap()
            out.writeframes(audio.tobytes())
        duration = len(audio) / RATE
        render_video(wav, args.output, lines, duration, tmp)

    transcript = args.output.with_suffix(".transcript.json")
    transcript.write_text(
        json.dumps(
            {
                "source": args.output.name,
                "transcriber": "script (line times exact, word times interpolated)",
                "lines": lines,
                "words": script_words(lines),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({duration:.2f} s) and {transcript}")
    for line in lines:
        print(f"  {line['start']:6.2f}-{line['end']:6.2f}  {line['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
