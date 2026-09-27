"""The cleaning pipeline: detect, decide, write the EDL and report, render."""

from __future__ import annotations

import bisect
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__, ffmpeg
from .report import format_seconds, render_report
from .retakes import detect_retakes
from .silence import detect_silences, silence_cuts
from .timeline import EditDecisionList, build_edl
from .transcribe import JsonTranscriber, Transcriber, Word, save_words

log = logging.getLogger(__name__)

MIN_PAUSE = 0.15
"""Shortest quiet span detected. Spans of at least ``Options.min_silence`` are
removed; all of them help place retake cuts inside real pauses."""


@dataclass(frozen=True)
class Options:
    """Analysis settings; each one maps to a command-line flag."""

    silence_threshold_db: float = -35.0
    min_silence: float = 0.6
    padding: float = 0.15
    similarity: float = 0.85
    min_take_words: int = 2
    sentence_gap: float = 1.0
    remove_silences: bool = True
    remove_retakes: bool = True


DEFAULT_OPTIONS = Options()


@dataclass(frozen=True)
class Result:
    """What :func:`clean` produced. ``output`` is None on a dry run, ``words_path``
    when no transcript was saved."""

    edl: EditDecisionList
    words: Sequence[Word]
    report: str
    edl_path: Path
    report_path: Path
    words_path: Path | None
    output: Path | None


def plan_edit(
    duration: float,
    *,
    silences: Iterable[tuple[float, float]] = (),
    words: Sequence[Word] = (),
    options: Options = DEFAULT_OPTIONS,
) -> EditDecisionList:
    """Decide what to cut, given the quiet spans of the audio and a transcript. No I/O.

    Quiet spans at least ``options.min_silence`` long are cut; all of them are
    used to place the boundaries of retake cuts.
    """
    silences = sorted(silences)
    long_silences = [(a, b) for a, b in silences if b - a >= options.min_silence]
    cuts = []
    if options.remove_silences:
        cuts += silence_cuts(
            long_silences, duration, min_silence=options.min_silence, padding=options.padding
        )
    if options.remove_retakes:
        heard = drop_words_in_silence(words, long_silences)
        if len(heard) < len(words):
            log.info(
                "  ignoring %d transcribed words inside silences (likely hallucinated)",
                len(words) - len(heard),
            )
        cuts += detect_retakes(
            heard,
            similarity=options.similarity,
            min_words=options.min_take_words,
            sentence_gap=options.sentence_gap,
            padding=options.padding,
            pauses=silences,
        )
    return build_edl(duration, cuts)


def drop_words_in_silence(
    words: Sequence[Word], silences: Sequence[tuple[float, float]]
) -> list[Word]:
    """Remove words whose whole time span lies inside one of ``silences`` (sorted, disjoint).

    Nobody is speaking there, so such words are speech-recognition
    hallucinations ("Thanks for watching!" in a silent ending is a classic).
    They would otherwise look like takes and could cause real speech to be cut.
    """
    starts = [start for start, _ in silences]

    def inside(word: Word) -> bool:
        i = bisect.bisect_right(starts, word.start) - 1
        return i >= 0 and word.end <= silences[i][1]

    return [word for word in words if not inside(word)]


def clean(
    src: Path,
    output: Path | None = None,
    *,
    transcriber: Transcriber | None,
    options: Options = DEFAULT_OPTIONS,
    render: bool = True,
    crf: int = 20,
    preset: str = "veryfast",
    quiet: bool = False,
) -> Result:
    """Analyse ``src``, write ``<output>.edl.json`` and ``<output>.report.txt``, render ``output``.

    ``output`` defaults to ``<src>.clean.mp4`` (``.m4a`` for audio-only input).
    A transcript produced by ``transcriber`` is saved to ``<output>.words.json``
    so a later run can reuse it through
    :class:`~video_cleaner.transcribe.JsonTranscriber`.
    """
    src = Path(src)
    if not src.is_file():
        raise ValueError(f"input file not found: {src}")
    ffmpeg.ensure_available()
    info = ffmpeg.probe(src)
    if not info.has_audio:
        raise ValueError(f"{src} has no audio stream to analyse")
    if output is None:
        output = src.with_name(f"{src.stem}.clean{'.mp4' if info.has_video else '.m4a'}")
    output = Path(output)
    if output.resolve() == src.resolve():
        raise ValueError("the output path must differ from the input path")
    output.parent.mkdir(parents=True, exist_ok=True)
    log.info("Input: %s (%s)", src, format_seconds(info.duration))

    log.info("Detecting pauses below %g dB ...", options.silence_threshold_db)
    silences = detect_silences(
        src,
        threshold_db=options.silence_threshold_db,
        min_duration=min(MIN_PAUSE, options.min_silence),
        duration=info.duration,
    )
    long_ones = sum(1 for start, end in silences if end - start >= options.min_silence)
    log.info("  %d pauses, %d of them at least %g s", len(silences), long_ones, options.min_silence)

    words: list[Word] = []
    words_path = None
    name = None
    if options.remove_retakes:
        if transcriber is None:
            raise ValueError("retake detection needs a transcriber")
        name = getattr(transcriber, "name", type(transcriber).__name__)
        log.info("Transcribing with %s ...", name)
        started = time.monotonic()
        words = transcriber.transcribe(src)
        log.info("  %d words in %.1f s", len(words), time.monotonic() - started)
        if not isinstance(transcriber, JsonTranscriber):
            words_path = output.with_suffix(".words.json")
            save_words(
                words_path,
                words,
                source=src.name,
                transcriber=name,
                model=getattr(transcriber, "model", None),
                language=getattr(transcriber, "detected_language", None),
            )

    edl = plan_edit(info.duration, silences=silences, words=words, options=options)
    if not edl.keep:
        raise ValueError(
            "nothing would be left after cutting; is the audio silent? "
            "Try a lower --silence-threshold."
        )

    edl_path = output.with_suffix(".edl.json")
    edl_data = edl.to_dict(
        format="video-cleaner-edl/1",
        tool=f"video-cleaner {__version__}",
        source=str(src),
        output=str(output) if render else None,
        transcriber=name,
        settings=asdict(options),
    )
    edl_path.write_text(json.dumps(edl_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render_report(edl, source=str(src), output=str(output) if render else None)
    report_path = output.with_suffix(".report.txt")
    report_path.write_text(report, encoding="utf-8")
    log.info("Wrote %s and %s", edl_path, report_path)

    if render:
        log.info("Rendering %d segments to %s ...", len(edl.keep), output)
        ffmpeg.render(
            src,
            output,
            edl.keep,
            has_video=info.has_video,
            sample_rate=info.sample_rate,
            crf=crf,
            preset=preset,
            quiet=quiet,
        )
        log.info("Done: %s (%s)", output, format_seconds(ffmpeg.probe(output).duration))

    return Result(
        edl=edl,
        words=words,
        report=report,
        edl_path=edl_path,
        report_path=report_path,
        words_path=words_path,
        output=output if render else None,
    )
