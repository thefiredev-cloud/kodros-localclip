"""Trim edit, per-clip captions/framing, batch export and the one-shot make, on generated media.

No models, no services: transcription is replaced by synthetic word timings and the
LLM is off, so selection takes the deterministic heuristic path.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from localclip import pipeline
from localclip.crop import CropPlan, crop_box, plan_crop
from localclip.pipeline import ClipSpec, Options, snap_to_words, validate_spec
from localclip.render import probe, video_size
from localclip.transcribe import Transcript
from localclip.words import Word

from media import make_video, sentence_words

needs_ffmpeg = pytest.mark.skipif(not all(shutil.which(x) for x in ("ffmpeg", "ffprobe")), reason="ffmpeg missing")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("media") / "talk.mp4")


@pytest.fixture
def fake_asr(monkeypatch, tmp_path):
    """Synthetic transcript instead of whisper; data home in tmp."""
    monkeypatch.setattr(pipeline, "DATA_HOME", tmp_path / "home")
    monkeypatch.setattr(
        pipeline, "_load_or_transcribe",
        lambda *a, **k: Transcript(sentence_words(), "synthetic", "test words", "en", 0.0),
    )


# -- pure logic -------------------------------------------------------------

def test_snap_moves_cuts_out_of_words():
    words = [Word(1.0, 1.5, "one"), Word(1.6, 2.0, "two"), Word(2.5, 3.0, "three")]
    s, e = snap_to_words(words, 1.2, 2.7)
    assert s == pytest.approx(0.95)  # just before "one"
    assert e == pytest.approx(3.0 + pipeline.SNAP_AFTER)  # just after "three"


def test_snap_leaves_cuts_in_silence_alone():
    words = [Word(1.0, 1.5, "one"), Word(2.5, 3.0, "two")]
    assert snap_to_words(words, 2.0, 2.2) == (2.0, 2.2)


def test_validate_clamps_and_rejects():
    with pytest.raises(ValueError, match="at most"):
        validate_spec(ClipSpec(-3, 400), 1000.0)
    s = validate_spec(ClipSpec(-3, 10), 100.0)
    assert (s.start, s.end) == (0.0, 10.0)
    with pytest.raises(ValueError, match="at least"):
        validate_spec(ClipSpec(5, 5.4), 100.0)
    with pytest.raises(ValueError, match="at least"):
        validate_spec(ClipSpec(99.5, 140), 100.0)  # clamps to 0.5 s
    with pytest.raises(ValueError, match="crop"):
        validate_spec(ClipSpec(0, 5, crop="zoom"), 100.0)
    with pytest.raises(ValueError, match="crop_x"):
        validate_spec(ClipSpec(0, 5, crop_x=1.5), 100.0)


def test_clipspec_dict_roundtrip_ignores_unknown_keys():
    d = ClipSpec(1, 9, title="t", sentences=(2, 3), crop_x=0.4).to_dict()
    d["id"] = 7  # the editor adds ids; ClipSpec must ignore them
    back = ClipSpec.from_dict(d)
    assert back.sentences == (2, 3) and back.crop_x == 0.4 and back.title == "t"


def test_manual_crop_needs_no_video_and_clamps():
    cw, _ = crop_box(1920, 1080, (9, 16))
    left = plan_crop("unused.mp4", 0, 5, 1920, 1080, (9, 16), manual_x=0.0)
    right = plan_crop("unused.mp4", 0, 5, 1920, 1080, (9, 16), manual_x=1.0)
    mid = plan_crop("unused.mp4", 0, 5, 1920, 1080, (9, 16), manual_x=0.5)
    assert (left.mode, left.keyframes) == ("manual", [(0.0, 0)])
    assert right.keyframes == [(0.0, 1920 - cw)]
    assert abs(mid.keyframes[0][1] - (1920 - cw) // 2) <= 1


def test_center_mode_skips_face_sampling():
    plan = plan_crop("unused.mp4", 0, 5, 1920, 1080, (9, 16), use_faces=True, mode="center")
    assert plan.mode == "center"
    with pytest.raises(ValueError):
        plan_crop("unused.mp4", 0, 5, 1920, 1080, (9, 16), mode="zoom")


def test_expand_inputs_sorts_dir_and_rejects_missing(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"x")
    (tmp_path / "a.MOV").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert [Path(p).name for p in pipeline.expand_inputs([str(tmp_path)])] == ["a.MOV", "b.mp4"]
    with pytest.raises(FileNotFoundError):
        pipeline.expand_inputs([str(tmp_path / "missing.mp4")])


# -- real renders -----------------------------------------------------------

@needs_ffmpeg
def test_export_trim_without_captions(video, tmp_path):
    words = sentence_words()
    spec = ClipSpec(start=2.0, end=8.0, title="trimmed", captions=False, crop="center", edited=True)
    clips = pipeline.export_clips(str(video), words, [spec], Options(use_faces=False), tmp_path / "out", log=lambda s: None)
    c = clips[0]
    info = probe(c["file"])
    assert video_size(info) == (1080, 1920)
    assert {s["codec_name"] for s in info["streams"]} == {"h264", "aac"}
    assert float(info["format"]["duration"]) == pytest.approx(6.0, abs=0.2)
    assert c["captions"] == {"enabled": False, "events": 0, "words": c["captions"]["words"], "ass": None}
    assert not list((tmp_path / "out" / "work").glob("*.ass"))
    assert c["crop"]["mode"] == "center" and c["edited"] is True
    assert "alpha" in c["transcript"] and "theta" in c["transcript"]  # recomputed from the trimmed window
    assert (tmp_path / "out" / "sheet.png").stat().st_size > 5000


@needs_ffmpeg
def test_export_manual_crop_changes_the_picture(video, tmp_path):
    import subprocess

    import numpy as np

    def mid_frame(path):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", path, "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(640, 360, 3).astype(int)

    specs = [ClipSpec(1, 4, captions=False, crop_x=0.0), ClipSpec(1, 4, captions=False, crop_x=1.0)]
    clips = pipeline.export_clips(str(video), sentence_words(), specs, Options(use_faces=False, long_side=640), tmp_path / "o", log=lambda s: None)
    assert [c["crop"]["mode"] for c in clips] == ["manual", "manual"]
    left, right = (mid_frame(c["file"]) for c in clips)
    assert np.abs(left - right).mean() > 5  # different horizontal windows of the source


@needs_ffmpeg
def test_make_end_to_end_heuristic(video, fake_asr, tmp_path):
    out = tmp_path / "run"
    m = pipeline.make(str(video), Options(count=2, min_dur=6, max_dur=12, use_llm=False, use_faces=False,
                                          long_side=640, out_dir=str(out)), log=lambda s: None)
    assert len(m["clips"]) == 2
    assert m["selection"]["engine"] == "heuristic"
    for c in m["clips"]:
        assert Path(c["file"]).exists()
        assert c["selected_by"] == "heuristic" and c["edited"] is False
        assert video_size(probe(c["file"])) == (360, 640)
    assert Path(m["output"]["contact_sheet"]).stat().st_size > 5000
    assert json.loads((out / "manifest.json").read_text())["clips"][0]["captions"]["enabled"] is True


@needs_ffmpeg
def test_analyze_then_export_with_trim(video, fake_asr, tmp_path):
    opts = Options(count=2, min_dur=6, max_dur=12, use_llm=False, use_faces=False, long_side=640)
    an = pipeline.analyze(str(video), opts, tmp_path / "w", log=lambda s: None)
    assert an.selection.shortlist and len(an.specs) == 2
    first = an.specs[0]
    first.end = first.start + 3.0  # user trims the pick to 3 s
    first.edited = True
    clips = pipeline.export_clips(str(video), an.transcript.words, [first], opts, tmp_path / "x", duration=an.duration,
                                  log=lambda s: None)
    assert clips[0]["duration"] == pytest.approx(3.0, abs=0.01)
    assert clips[0]["output_duration"] == pytest.approx(3.0, abs=0.2)


@needs_ffmpeg
def test_batch_continues_after_a_bad_file(video, fake_asr, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    second = tmp_path / "second.mp4"
    shutil.copy(video, second)
    results = pipeline.batch_make(
        [str(video), str(bad), str(second)],
        Options(count=1, min_dur=6, max_dur=12, use_llm=False, use_faces=False, long_side=640, out_dir=str(tmp_path / "batch")),
        log=lambda s: None,
    )
    assert [r["ok"] for r in results] == [True, False, True]
    assert results[1]["error"]
    assert all(Path(c).exists() for r in results if r["ok"] for c in r["clips"])
    assert (tmp_path / "batch" / "talk").is_dir() and (tmp_path / "batch" / "second").is_dir()
