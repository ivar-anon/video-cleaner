import pytest

from video_cleaner.silence import parse_silencedetect, silence_cuts

LOG = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'talk.mp4':
  Duration: 00:00:30.57, start: 0.000000, bitrate: 272 kb/s
[silencedetect @ 0x55d2] silence_start: 0
[silencedetect @ 0x55d2] silence_end: 1.516 | silence_duration: 1.516
size=N/A time=00:00:05.00 bitrate=N/A speed= 800x
[silencedetect @ 0x55d2] silence_start: 4.07
[silencedetect @ 0x55d2] silence_end: 5.87 | silence_duration: 1.8
[silencedetect @ 0x55d2] silence_start: 28.54
"""


class TestParseSilencedetect:
    def test_pairs_starts_and_ends(self):
        assert parse_silencedetect(LOG, duration=30.567) == [
            (0.0, 1.516),
            (4.07, 5.87),
            (28.54, 30.567),  # still silent at the end of the file
        ]

    def test_clamps_to_the_media(self):
        log = "silence_start: -0.00133\nsilence_end: 2.5e-01 | silence_duration: 0.25\n"
        log += "silence_start: 9.5\nsilence_end: 10.02 | silence_duration: 0.52\n"
        assert parse_silencedetect(log, duration=10.0) == [(0.0, 0.25), (9.5, 10.0)]

    def test_no_silence(self):
        assert parse_silencedetect("size=N/A time=00:00:05.00\n", duration=5.0) == []


class TestSilenceCuts:
    def test_padding_is_kept_on_both_sides(self):
        [cut] = silence_cuts([(4.0, 6.0)], 10.0, padding=0.15)
        assert cut.reason == "silence"
        assert (cut.start, cut.end) == (pytest.approx(4.15), pytest.approx(5.85))

    def test_leading_and_trailing_silence_keep_padding_on_the_speech_side_only(self):
        cuts = silence_cuts([(0.0, 1.5), (8.0, 10.0)], 10.0, padding=0.15)
        assert [(c.start, c.end) for c in cuts] == [
            (0.0, pytest.approx(1.35)),
            (pytest.approx(8.15), 10.0),
        ]

    def test_short_silences_are_kept(self):
        assert silence_cuts([(4.0, 4.5)], 10.0, min_silence=0.6) == []

    def test_no_cut_when_padding_covers_the_whole_silence(self):
        assert silence_cuts([(4.0, 4.7)], 10.0, min_silence=0.6, padding=0.4) == []
