import json

import pytest

from fakes import FakeTranscriber
from video_cleaner.transcribe import (
    FasterWhisperTranscriber,
    JsonTranscriber,
    Transcriber,
    Word,
    load_words,
    save_words,
    words_from_json,
)


def test_accepts_a_bare_list_or_an_object_and_the_word_alias():
    items = [
        {"word": " Hello", "start": 0.0, "end": 0.4},
        {"text": "world.", "start": 0.5, "end": 0.9},
    ]
    expected = [Word("Hello", 0.0, 0.4), Word("world.", 0.5, 0.9)]
    assert words_from_json(items) == expected
    assert words_from_json({"language": "en", "words": items}) == expected


def test_sorts_words_and_skips_empty_ones():
    items = [
        {"text": "b", "start": 1, "end": 2},
        {"text": " ", "start": 0, "end": 1},
        {"text": "a", "start": 0, "end": 1},
    ]
    assert [w.text for w in words_from_json(items)] == ["a", "b"]


def test_invalid_items_are_reported():
    with pytest.raises(ValueError, match="word #0"):
        words_from_json([{"text": "no times"}])
    with pytest.raises(ValueError):
        words_from_json({"segments": []})


def test_save_and_load_round_trip(tmp_path):
    words = [Word("Hi,", 0.1234, 0.5, 0.91), Word("there.", 0.6, 1.0)]
    path = tmp_path / "t.words.json"
    save_words(path, words, model="base")
    assert json.loads(path.read_text())["model"] == "base"
    assert load_words(path) == [Word("Hi,", 0.123, 0.5, 0.91), Word("there.", 0.6, 1.0)]
    assert JsonTranscriber(path).transcribe() == load_words(path)


def test_every_transcriber_satisfies_the_protocol():
    # Creating a FasterWhisperTranscriber does not import faster-whisper or load a model.
    for transcriber in [
        FasterWhisperTranscriber("tiny"),
        JsonTranscriber("x.json"),
        FakeTranscriber([]),
    ]:
        assert isinstance(transcriber, Transcriber)
