"""Command-line interface: ``video-cleaner INPUT [-o OUTPUT] [options]``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .ffmpeg import FFmpegError
from .pipeline import Options, clean
from .transcribe import FasterWhisperTranscriber, JsonTranscriber, Transcriber

log = logging.getLogger("video_cleaner")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-cleaner",
        description="Remove long silences and repeated takes from a talking-head video.",
    )
    parser.add_argument("input", type=Path, help="video (or audio) file to clean")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="rendered file; the EDL, report and transcript are written next to it "
        "(default: <input>.clean.mp4, or .m4a for audio)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write the EDL and report but do not render the video",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    silence = parser.add_argument_group("silence removal")
    silence.add_argument(
        "--silence-threshold", type=float, default=-35.0, metavar="DB",
        help="audio quieter than this counts as silence (default: %(default)s)",
    )
    silence.add_argument(
        "--min-silence", type=float, default=0.6, metavar="SEC",
        help="remove silences at least this long (default: %(default)s)",
    )
    silence.add_argument(
        "--padding", type=float, default=0.15, metavar="SEC",
        help="pause kept next to speech at every cut (default: %(default)s)",
    )
    silence.add_argument("--no-silence", action="store_true", help="keep all silences")

    takes = parser.add_argument_group("repeated takes")
    takes.add_argument(
        "--no-retakes", action="store_true",
        help="skip transcription and keep repeated takes",
    )
    takes.add_argument(
        "--transcript", type=Path, metavar="JSON",
        help="use this word-level transcript instead of running Whisper "
        "(for example the .words.json of a previous run)",
    )
    takes.add_argument(
        "--model", default="base",
        help="faster-whisper model: tiny, base, small, ... or a local model directory "
        "(default: %(default)s)",
    )
    takes.add_argument("--language", help="spoken language, e.g. en (default: auto-detect)")
    takes.add_argument(
        "--device", default="cpu", help="faster-whisper device, cpu or cuda (default: %(default)s)"
    )
    takes.add_argument(
        "--similarity", type=float, default=0.85, metavar="0-1",
        help="minimum fuzzy similarity between two takes of a sentence (default: %(default)s)",
    )
    takes.add_argument(
        "--min-take-words", type=int, default=2, metavar="N",
        help="ignore abandoned takes shorter than N words (default: %(default)s)",
    )
    takes.add_argument(
        "--sentence-gap", type=float, default=1.0, metavar="SEC",
        help="a pause this long between words also ends a sentence (default: %(default)s)",
    )

    encoding = parser.add_argument_group("encoding (video is re-encoded with x264 + AAC)")
    encoding.add_argument(
        "--crf", type=int, default=20, help="x264 quality, lower is better (default: %(default)s)"
    )
    encoding.add_argument(
        "--preset", default="veryfast", help="x264 speed preset (default: %(default)s)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.quiet)

    problems = _validate(args)
    if problems:
        parser.error("; ".join(problems))

    options = Options(
        silence_threshold_db=args.silence_threshold,
        min_silence=args.min_silence,
        padding=args.padding,
        similarity=args.similarity,
        min_take_words=args.min_take_words,
        sentence_gap=args.sentence_gap,
        remove_silences=not args.no_silence,
        remove_retakes=not args.no_retakes,
    )
    transcriber: Transcriber | None = None
    if not args.no_retakes:
        transcriber = (
            JsonTranscriber(args.transcript)
            if args.transcript
            else FasterWhisperTranscriber(args.model, device=args.device, language=args.language)
        )
    try:
        result = clean(
            args.input,
            args.output,
            transcriber=transcriber,
            options=options,
            render=not args.dry_run,
            crf=args.crf,
            preset=args.preset,
            quiet=args.quiet,
        )
    except (FFmpegError, OSError, RuntimeError, ValueError) as exc:
        log.error("error: %s", exc)
        return 1
    if not args.quiet:
        sys.stdout.write(result.report)
    return 0


def _validate(args: argparse.Namespace) -> list[str]:
    problems = []
    if args.min_silence <= 0:
        problems.append("--min-silence must be positive")
    if args.padding < 0:
        problems.append("--padding cannot be negative")
    if not 0 < args.similarity <= 1:
        problems.append("--similarity must be in (0, 1]")
    if args.min_take_words < 1:
        problems.append("--min-take-words must be at least 1")
    if args.no_silence and args.no_retakes:
        problems.append("--no-silence and --no-retakes together leave nothing to do")
    if args.transcript and args.no_retakes:
        problems.append("--transcript has no effect with --no-retakes")
    return problems


def _configure_logging(quiet: bool) -> None:
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.propagate = False
    log.setLevel(logging.WARNING if quiet else logging.INFO)
