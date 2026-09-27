import re

import pytest

from video_cleaner.ffmpeg import build_filter_graph, ffmpeg_major_version
from video_cleaner.timeline import Segment


def evaluate(expression: str, **variables: float) -> float:
    """Evaluate the small subset of ffmpeg's expression language the graph uses."""
    functions = {
        "if_": lambda cond, then, other: then if cond else other,
        "lt": lambda a, b: float(a < b),
        "gte": lambda a, b: float(a >= b),
        "min": min,
    }
    return eval(expression.replace("if(", "if_("), {"__builtins__": {}}, {**functions, **variables})


def test_graph_structure():
    graph = build_filter_graph([Segment(1, 2.5), Segment(4, 6)], has_video=True)
    video, audio = graph.split(";\n")
    assert video.startswith("[0:v:0]select='if(lt(t,3.9993333),")
    assert video.endswith("setpts='PTS-(if(lt(T,3.9993333),1.0000000,2.5000000))/TB'[outv]")
    assert audio.startswith("[0:a:0]asetnsamples=n=64:p=0,aselect='if(lt(t,3.9993333),")
    assert audio.endswith("':eval=frame,asetpts=N/SR/TB[outa]")


def test_boundaries_snap_to_the_audio_block_grid():
    # 64 samples at 48 kHz = 1.333 ms: 1.0004 s snaps to 1.0 s.
    graph = build_filter_graph([Segment(1.0004, 2.0)], has_video=True, sample_rate=48_000)
    assert "setpts='PTS-(1.0000000)/TB'" in graph
    # A segment shorter than half a block disappears.
    with pytest.raises(ValueError):
        build_filter_graph([Segment(1.0, 1.0003)], has_video=True, sample_rate=48_000)


def test_expressions_keep_fade_and_shift_the_right_times():
    segments = [Segment(2 * i + 0.5, 2 * i + 1.5) for i in range(37)]  # 1 s kept every 2 s
    graph = build_filter_graph(segments, has_video=True)
    keep = re.search(r"select='([^']+)'", graph).group(1)
    shift = re.search(r"setpts='PTS-\(([^']+)\)/TB'", graph).group(1)
    fade = re.search(r"volume='([^']+)'", graph).group(1)

    for i, seg in enumerate(segments):
        assert evaluate(keep, t=seg.start) == 1
        assert evaluate(keep, t=seg.end - 0.001) == 1
        assert evaluate(keep, t=seg.end) == 0
        assert evaluate(keep, t=seg.start - 0.25) == 0
        # Removed time before segment i: 0.5 s of lead-in plus 1 s per earlier gap.
        assert evaluate(shift, T=seg.start + 0.3) == pytest.approx(0.5 + i)
        assert evaluate(fade, t=seg.start + 0.005) == pytest.approx(0.5)
        assert evaluate(fade, t=seg.start + 0.5) == 1


def test_graph_for_audio_only_input():
    graph = build_filter_graph([Segment(0, 3)], has_video=False)
    assert "[0:v" not in graph
    assert graph.startswith("[0:a:0]asetnsamples") and graph.endswith("[outa]")


def test_graph_needs_something_to_keep():
    with pytest.raises(ValueError):
        build_filter_graph([], has_video=True)


@pytest.mark.parametrize(
    "banner, major",
    [
        ("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023", 6),
        ("ffmpeg version n7.1 Copyright (c) 2000-2024", 7),
        ("ffmpeg version N-118000-g1234abcd Copyright", None),
    ],
)
def test_ffmpeg_major_version(banner, major):
    assert ffmpeg_major_version(banner) == major
