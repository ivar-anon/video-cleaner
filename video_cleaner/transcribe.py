"""Word-level transcription behind a small, pluggable interface.

The rest of the tool only needs a list of :class:`Word` objects, so any speech
recogniser can be plugged in by implementing :class:`Transcriber`. Two
implementations ship with the package:

* :class:`FasterWhisperTranscriber` runs faster-whisper locally (CPU by default).
* :class:`JsonTranscriber` reads words from a JSON file: the transcript saved
  by a previous run, the output of another ASR system, or a test fixture.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Union, runtime_checkable

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Word:
    """A transcribed word and its time span in the source, in seconds."""

    text: str
    start: float
    end: float
    probability: float = 1.0


@runtime_checkable
class Transcriber(Protocol):
    """Turns a media file into a time-ordered list of words.

    Implementations may also expose a ``name`` attribute, which is recorded in
    the saved transcript.
    """

    def transcribe(self, media: Path) -> list[Word]:
        ...


class FasterWhisperTranscriber:
    """Word-level transcription with faster-whisper, imported lazily.

    ``model`` is a model size ("tiny", "base", "small", ...) that is downloaded
    from the Hugging Face Hub on first use, or a path to a converted model
    directory for offline use. ``vad`` runs faster-whisper's voice activity
    filter first, which stops Whisper from inventing text in long silences.
    """

    name = "faster-whisper"

    def __init__(
        self,
        model: str = "base",
        *,
        device: str = "cpu",
        compute_type: str | None = None,
        language: str | None = None,
        vad: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type or ("int8" if device == "cpu" else "float16")
        self.language = language
        self.vad = vad
        self.detected_language: str | None = None

    def transcribe(self, media: Path) -> list[Word]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "faster-whisper is not installed: run `pip install faster-whisper`, "
                "or use --transcript FILE / --no-retakes"
            ) from exc

        try:
            model = WhisperModel(self.model, device=self.device, compute_type=self.compute_type)
        except Exception as exc:  # download, cache or device problems surface in many types
            raise RuntimeError(
                f"could not load the Whisper model {self.model!r} ({exc}). If it cannot be "
                "downloaded, pass a local model directory with --model, or use "
                "--transcript FILE / --no-retakes"
            ) from exc
        segments, info = model.transcribe(
            str(media),
            language=self.language,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=self.vad,
        )
        # `segments` is a generator: decoding happens while it is consumed.
        words = [
            Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
            for segment in segments
            for w in segment.words or ()
            if w.word.strip()
        ]
        self.detected_language = info.language
        return words


class JsonTranscriber:
    """Reads words from a JSON transcript instead of running speech recognition."""

    name = "json"

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)

    def transcribe(self, media: Path | None = None) -> list[Word]:
        return load_words(self.path)


def words_from_json(data: Any) -> list[Word]:
    """Parse ``{"words": [...]}`` or a bare list of word objects.

    Each item needs ``text`` (or ``word``), ``start`` and ``end`` in seconds;
    ``probability`` is optional. Items with empty text are skipped.
    """
    items = data.get("words") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError('transcript must be a list of words or {"words": [...]}')
    words = []
    for index, item in enumerate(items):
        try:
            text = str(item.get("text", item.get("word", ""))).strip()
            if text:
                words.append(
                    Word(
                        text,
                        float(item["start"]),
                        float(item["end"]),
                        float(item.get("probability", 1.0)),
                    )
                )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid word #{index} in transcript: {item!r}") from exc
    return sorted(words, key=lambda w: (w.start, w.end))


def load_words(path: PathLike) -> list[Word]:
    with open(path, encoding="utf-8") as fh:
        return words_from_json(json.load(fh))


def save_words(path: PathLike, words: Iterable[Word], **metadata: Any) -> None:
    """Write words (plus optional metadata) in the format :func:`load_words` reads."""
    payload = {
        **metadata,
        "words": [
            {
                "text": w.text,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "probability": round(w.probability, 3),
            }
            for w in words
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")
