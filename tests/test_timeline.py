import json

import pytest

from video_cleaner.timeline import Cut, Segment, build_edl, keep_segments, resolve_cuts


def spans(cuts):
    return [(round(c.start, 6), round(c.end, 6), c.reason) for c in cuts]


class TestResolveCuts:
    def test_retake_wins_where_it_overlaps_a_silence(self):
        cuts = resolve_cuts([Cut(4, 6, "silence"), Cut(5, 9, "retake")], 10)
        assert spans(cuts) == [(4, 5, "silence"), (5, 9, "retake")]

    def test_silence_inside_a_retake_disappears(self):
        cuts = resolve_cuts([Cut(5, 6, "silence"), Cut(4, 9, "retake")], 10)
        assert spans(cuts) == [(4, 9, "retake")]

    def test_silence_around_a_retake_is_split(self):
        cuts = resolve_cuts([Cut(2, 8, "silence"), Cut(4, 5, "retake")], 10)
        assert spans(cuts) == [(2, 4, "silence"), (4, 5, "retake"), (5, 8, "silence")]

    def test_overlaps_of_the_same_reason_are_not_double_counted(self):
        cuts = resolve_cuts([Cut(1, 3, "silence"), Cut(2, 4, "silence")], 10)
        assert spans(cuts) == [(1, 3, "silence"), (3, 4, "silence")]

    def test_cuts_are_clipped_to_the_media(self):
        cuts = [Cut(-1, 2, "silence"), Cut(9, 12, "silence"), Cut(11, 12, "silence")]
        cuts = resolve_cuts(cuts, 10)
        assert spans(cuts) == [(0, 2, "silence"), (9, 10, "silence")]


def test_keep_segments_are_the_complement_of_the_cuts():
    cuts = [Cut(0, 1, "silence"), Cut(3, 4, "retake"), Cut(4, 5, "silence")]
    assert keep_segments(cuts, 10) == [Segment(1, 3), Segment(5, 10)]
    assert keep_segments([], 10) == [Segment(0, 10)]


class TestBuildEdl:
    def test_slivers_between_cuts_are_absorbed(self):
        edl = build_edl(10, [Cut(1, 3, "silence"), Cut(3.05, 5, "retake")], min_keep=0.1)
        assert edl.keep == (Segment(0, 1), Segment(5, 10))
        assert spans(edl.cuts) == [(1, 3.05, "silence"), (3.05, 5, "retake")]

    def test_slivers_at_both_ends_are_absorbed(self):
        edl = build_edl(10, [Cut(0.05, 2, "silence"), Cut(8, 9.95, "silence")], min_keep=0.1)
        assert edl.keep == (Segment(2, 8),)

    def test_durations_add_up(self):
        edl = build_edl(
            30, [Cut(0, 1.4, "silence"), Cut(4, 8.5, "retake"), Cut(8, 9, "silence")]
        )
        assert edl.output_duration == pytest.approx(30 - 1.4 - 5.0)
        assert edl.removed_duration == pytest.approx(sum(c.duration for c in edl.cuts))
        assert edl.summary() == {
            "silence": {"count": 2, "seconds": pytest.approx(1.9)},
            "retake": {"count": 1, "seconds": pytest.approx(4.5)},
        }

    def test_everything_can_be_cut(self):
        assert build_edl(5, [Cut(0, 5, "silence")]).keep == ()

    def test_to_dict_maps_kept_segments_onto_the_output_timeline(self):
        edl = build_edl(10, [Cut(1, 2, "silence"), Cut(4, 7, "retake", {"removed_text": "x"})])
        data = json.loads(json.dumps(edl.to_dict(source="in.mp4")))
        assert data["source"] == "in.mp4"
        assert data["output_duration"] == 6.0
        assert data["cuts"][1] == {
            "start": 4.0, "end": 7.0, "duration": 3.0, "reason": "retake", "removed_text": "x"
        }
        assert [(k["output_start"], k["output_end"]) for k in data["keep"]] == [
            (0.0, 1.0), (1.0, 3.0), (3.0, 6.0)
        ]
