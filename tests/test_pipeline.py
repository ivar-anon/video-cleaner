"""The whole pipeline with a fake transcript, first without and then with ffmpeg."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from fakes import FakeTranscriber, transcript
from video_cleaner.cli import main
from video_cleaner.ffmpeg import probe
from video_cleaner.pipeline import Options, clean, drop_words_in_silence, plan_edit
from video_cleaner.retakes import detect_retakes
from video_cleaner.transcribe import Word, load_words, save_words

# An 8 s clip: "speech" (a tone) at 1-2, 3.5-4.5 and 5.5-7 s, silence elsewhere.
DURATION = 8.0
BURSTS = [(1.0, 2.0), (3.5, 4.5), (5.5, 7.0)]
SILENCES = [(0.0, 1.0), (2.0, 3.5), (4.5, 5.5), (7.0, 8.0)]
WORDS = transcript((1.0, "Hello there."), (3.5, "Today I want"), (5.5, "Today I want more."))

# Expected: trim the leading and trailing silence, and cut the abandoned take
# "Today I want" together with the pauses around it (0.15 s padding kept).
EXPECTED_CUTS = [(0.0, 0.85, "silence"), (2.15, 5.35, "retake"), (7.15, 8.0, "silence")]
EXPECTED_OUTPUT = 1.3 + 1.8

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def spans(edl):
    return [(round(c.start, 3), round(c.end, 3), c.reason) for c in edl.cuts]


def test_plan_edit_with_a_fake_transcriber():
    words = FakeTranscriber(WORDS).transcribe(Path("talk.mp4"))
    edl = plan_edit(DURATION, silences=SILENCES, words=words)
    assert spans(edl) == EXPECTED_CUTS
    assert edl.output_duration == pytest.approx(EXPECTED_OUTPUT)
    assert edl.cuts[1].info["removed_text"] == "Today I want"


def test_plan_edit_can_skip_either_step():
    silences_only = plan_edit(
        DURATION, silences=SILENCES, words=WORDS, options=Options(remove_retakes=False)
    )
    assert [c.reason for c in silences_only.cuts] == ["silence"] * 4

    retakes_only = plan_edit(
        DURATION, silences=SILENCES, words=WORDS, options=Options(remove_silences=False)
    )
    assert spans(retakes_only) == [(2.15, 5.35, "retake")]


def test_drop_words_in_silence():
    words = [Word("a", 0.5, 0.9), Word("b", 1.2, 1.4), Word("c", 1.9, 2.3), Word("d", 3.0, 3.0)]
    kept = drop_words_in_silence(words, [(1.0, 2.0), (2.9, 3.5)])
    assert [w.text for w in kept] == ["a", "c"]  # "c" only overlaps the silence


def test_words_hallucinated_in_a_silence_do_not_cause_cuts():
    # Real last sentence, then text a speech model invented inside the final silence.
    real = transcript((5.5, "Thanks for watching."))
    invented = [Word(w, 7.9, 7.9) for w in "Thanks for watching, see you next time.".split()]
    assert len(detect_retakes(real + invented)) == 1  # would cut the real sentence

    edl = plan_edit(DURATION, silences=SILENCES, words=real + invented)
    assert [c.reason for c in edl.cuts] == ["silence"] * 4


def make_clip(path: Path) -> Path:
    gate = "+".join(f"between(t,{a},{b})" for a, b in BURSTS)
    audio = f"aevalsrc='0.5*sin(2*PI*440*t)*({gate})':s=16000:d={DURATION}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", audio]
    if path.suffix != ".wav":
        cmd += [
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=25:d={DURATION}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le",
        ]
    subprocess.run([*cmd, str(path)], check=True)
    return path


@needs_ffmpeg
def test_clean_renders_the_video_described_by_the_edl(tmp_path):
    clip = make_clip(tmp_path / "talk.mkv")
    fake = FakeTranscriber(WORDS)

    result = clean(clip, tmp_path / "out" / "talk.clean.mp4", transcriber=fake, quiet=True)

    assert fake.calls == [clip]
    assert spans(result.edl) == EXPECTED_CUTS
    info = probe(result.output)
    assert info.has_video and info.has_audio
    assert info.duration == pytest.approx(EXPECTED_OUTPUT, abs=0.1)
    edl = json.loads(result.edl_path.read_text())
    assert edl["summary"] == {
        "silence": {"count": 2, "seconds": 1.7},
        "retake": {"count": 1, "seconds": 3.2},
    }
    assert "restarted as" in result.report_path.read_text()
    assert load_words(result.words_path) == WORDS


@needs_ffmpeg
def test_cli_dry_run_with_a_json_transcript(tmp_path, capsys):
    clip = make_clip(tmp_path / "talk.mkv")
    save_words(tmp_path / "words.json", WORDS)
    out = tmp_path / "talk.clean.mp4"

    args = [str(clip), "-o", str(out), "--transcript", str(tmp_path / "words.json"), "--dry-run"]
    code = main(args)

    assert code == 0
    assert not out.exists()
    assert json.loads(out.with_suffix(".edl.json").read_text())["output"] is None
    assert '"Today I want"' in capsys.readouterr().out


@needs_ffmpeg
def test_cli_audio_only_input_silences_only(tmp_path):
    clip = make_clip(tmp_path / "talk.wav")

    assert main([str(clip), "--no-retakes", "-q"]) == 0

    out = tmp_path / "talk.clean.m4a"
    assert probe(out).duration == pytest.approx(DURATION - 0.85 - 1.2 - 0.7 - 0.85, abs=0.1)


def test_cli_rejects_contradictory_flags(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["talk.mp4", "--no-silence", "--no-retakes"])
    assert exit_info.value.code == 2
    assert "nothing to do" in capsys.readouterr().err


def test_cli_reports_a_missing_input(tmp_path):
    assert main([str(tmp_path / "missing.mp4"), "--no-retakes", "-q"]) == 1
