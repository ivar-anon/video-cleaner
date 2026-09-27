from video_cleaner.report import format_seconds, render_report
from video_cleaner.timeline import Cut, build_edl


def test_format_seconds():
    assert format_seconds(0) == "0:00.00"
    assert format_seconds(83.456) == "1:23.46"
    assert format_seconds(59.999) == "1:00.00"
    assert format_seconds(3723) == "1:02:03.00"


def test_report_shows_durations_cuts_and_removed_takes():
    edl = build_edl(
        30.0,
        [
            Cut(0.0, 1.5, "silence"),
            Cut(10.0, 12.0, "silence"),
            Cut(4.0, 8.5, "retake", {
                "removed_text": "Today I want to",
                "matched_text": "Today I want to go.",
                "similarity": 1.0,
            }),
        ],
    )
    report = render_report(edl, source="in.mp4", output="out.mp4")
    assert "original duration  0:30.00  (30.00 s)" in report
    assert "final duration     0:22.00  (22.00 s)" in report
    assert "removed            8.00 s  (26.7 %)" in report
    assert "silence            2 cuts     3.50 s" in report
    assert "retake             1 cut      4.50 s" in report
    assert '0:04.00-0:08.50  "Today I want to"' in report
    assert 'restarted as "Today I want to go." (similarity 1.00)' in report
    assert "(not rendered)" in render_report(edl, source="in.mp4", output=None)
