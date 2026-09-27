import pytest

from fakes import transcript
from video_cleaner.retakes import (
    detect_retakes,
    ends_align,
    normalize,
    split_sentences,
    take_similarity,
)


def removed(cuts):
    return [cut.info["removed_text"] for cut in cuts]


class TestNormalize:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize("Hello,") == "hello"
        assert normalize("\u201cDon\u2019t\u201d") == "don't"
        assert normalize("sh-") == "sh"

    def test_fillers_become_empty(self):
        assert normalize("Um,") == ""
        assert normalize("uh...") == ""


class TestSplitSentences:
    def test_terminal_punctuation_ends_a_sentence(self):
        words = transcript((0, 'Hello there. How are you? "Fine!" Thanks'))
        sentences = [[w.text for w in s] for s in split_sentences(words)]
        assert sentences == [["Hello", "there."], ["How", "are", "you?"], ['"Fine!"'], ["Thanks"]]

    def test_long_pause_ends_a_sentence(self):
        words = transcript((0, "so today I"), (3, "want to"))
        assert len(split_sentences(words, max_gap=1.0)) == 2

    def test_short_pause_does_not(self):
        words = transcript((0, "so today I"), (1.3, "want to"))  # 0.3 s pause
        assert len(split_sentences(words, max_gap=1.0)) == 1


class TestScores:
    def test_similarity_of_identical_and_prefix_takes(self):
        assert take_similarity(["a", "b"], ["a", "b"]) == 1.0
        assert take_similarity("today i want to".split(), "today i want to show you".split()) == 1.0

    def test_similarity_of_unrelated_sentences_is_low(self):
        assert take_similarity("hi and welcome".split(), "today i want to show you".split()) < 0.6
        assert take_similarity([], ["a"]) == 0.0

    def test_ends_align(self):
        assert ends_align("today i want to".split(), "today i want to show you".split())
        assert ends_align("today i want to sh".split(), "today i want to show you".split())
        assert not ends_align("i use the editor".split(), "i use the terminal too".split())


class TestDetectRetakes:
    def test_abandoned_take_is_cut_and_the_last_take_kept(self):
        words = transcript(
            (0.0, "Hi there."),
            (2.0, "Today I want to show you"),
            (5.0, "Today I want to show you how this works."),
        )
        [cut] = detect_retakes(words)
        assert cut.reason == "retake"
        assert cut.info["removed_text"] == "Today I want to show you"
        assert cut.info["matched_text"] == "Today I want to show you how this works."
        assert cut.info["similarity"] == 1.0
        # From the abandoned take up to `padding` before the take that is kept.
        assert cut.start == pytest.approx(2.0 - 0.15)
        assert cut.end == pytest.approx(5.0 - 0.15)

    def test_only_the_last_of_several_takes_survives(self):
        words = transcript(
            (0, "So in this video"),
            (2, "So in this video we"),
            (4, "So in this video we will edit."),
        )
        cuts = detect_retakes(words)
        assert removed(cuts) == ["So in this video", "So in this video we"]
        assert cuts[0].end == pytest.approx(cuts[1].start)  # nothing left in between

    def test_a_repeat_with_a_correction_keeps_the_correction(self):
        words = transcript((0, "The price is ten dollars."), (3, "The price is twenty dollars."))
        assert removed(detect_retakes(words)) == ["The price is ten dollars."]

    @pytest.mark.parametrize(
        "first, second",
        [
            ("I want to show you the editor.", "I want to show you the terminal too."),
            ("The first step finds the long pauses.", "The second step finds the repeated takes."),
            ("We need three things.", "We need four things."),
            ("It is fast.", "It is simple."),
            ("Click on the button.", "Click on the file menu."),
        ],
    )
    def test_similar_but_different_sentences_are_kept(self, first, second):
        assert detect_retakes(transcript((0, first), (4, second))) == []

    def test_restart_inside_one_sentence(self):
        # Whisper often writes a comma, not a period, after an abandoned take.
        words = transcript((0, "Today I want to, today I want to show you how it works."))
        [cut] = detect_retakes(words)
        assert cut.info["removed_text"] == "Today I want to,"
        assert cut.info["matched_text"] == "today I want to show you how it works."

    def test_parallel_clauses_inside_a_sentence_are_kept(self):
        assert detect_retakes(transcript((0, "The more you practice, the more you learn."))) == []

    def test_restart_then_abandoned_sentence(self):
        words = transcript(
            (0, "Today I, today I want to."),
            (3, "Today I want to show you."),
        )
        cuts = detect_retakes(words)
        assert removed(cuts) == ["Today I,", "today I want to."]
        assert cuts[0].end == pytest.approx(cuts[1].start)

    def test_min_words(self):
        words = transcript((0, "Yes."), (2, "Yes, exactly."))
        assert detect_retakes(words) == []
        assert removed(detect_retakes(words, min_words=1)) == ["Yes."]

    def test_fillers_are_ignored_when_comparing(self):
        words = transcript((0, "Today I want to, um."), (3, "Today I want to show you."))
        assert removed(detect_retakes(words)) == ["Today I want to, um."]

    def test_a_filler_sentence_between_takes_is_cut_too(self):
        words = transcript((0, "Today I want to."), (2, "Uh."), (3, "Today I want to show you."))
        [cut] = detect_retakes(words)
        assert cut.start < 2.0 and cut.end == pytest.approx(3.0 - 0.15)

    def test_clipped_last_word(self):
        words = transcript((0, "Today I want to sh-"), (3, "Today I want to show you how."))
        assert removed(detect_retakes(words)) == ["Today I want to sh-"]

    def test_similarity_threshold(self):
        words = transcript(
            (0, "Today I'm going to show you."), (3, "Today I want to show you how it works.")
        )
        assert detect_retakes(words) == []
        assert len(detect_retakes(words, similarity=0.8)) == 1

    def test_padding_is_at_most_half_the_pause(self):
        words = transcript((0, "Today I want to"), (1.5, "Today I want to go."))
        [cut] = detect_retakes(words)  # the take ends at 1.35 s: a 0.15 s pause
        assert cut.end == pytest.approx(1.5 - 0.075)

    def test_boundaries_snap_to_pauses_in_the_audio(self):
        # The timestamps say the kept take starts at 5.0 s; the audio is quiet until 5.4 s.
        words = transcript((0, "Hi there."), (2.0, "Today I want to"), (5.0, "Today I want to go."))
        [cut] = detect_retakes(words, pauses=[(0.65, 2.1), (3.35, 5.4)])
        assert cut.start == pytest.approx(0.65 + 0.15)
        assert cut.end == pytest.approx(5.4 - 0.15)

    def test_pauses_far_from_the_words_are_ignored(self):
        words = transcript((0, "Hi there."), (2.0, "Today I want to"), (5.0, "Today I want to go."))
        [cut] = detect_retakes(words, pauses=[(10.0, 12.0)])
        assert (cut.start, cut.end) == (pytest.approx(1.85), pytest.approx(4.85))

    def test_empty_transcript(self):
        assert detect_retakes([]) == []
