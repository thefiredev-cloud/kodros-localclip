"""Decode tiny generated clips to verify caption paths, without ASR, faces or models."""

import shutil
import subprocess

import numpy as np
import pytest

from localclip.captions import CaptionStyle, build_ass, caption_events
from localclip.crop import CropPlan
from localclip.render import measure_loudness, probe, render_clip, video_size
from localclip.words import Word


@pytest.mark.skipif(not all(shutil.which(x) for x in ("ffmpeg", "ffprobe")), reason="ffmpeg/ffprobe missing")
@pytest.mark.parametrize("filename", ["captions.ass", "it's.ass", "colon:comma,[bracket];back\\slash.ass", " leading.ass "])
def test_render_decodes_captions_from_special_paths(tmp_path, filename):
    src = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:r=30:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src),
    ], check=True)
    style = CaptionStyle(size_ratio=0.2, outline_px=2, shadow_px=0)
    ass = tmp_path / filename
    ass.write_text(build_ass(caption_events([Word(0, 0.9, "TEST")], 1, style), 320, 240, style))
    out = tmp_path / "clip.mp4"
    enc, note = render_clip(
        str(src), 0, 1, CropPlan("full-frame", 320, 240, 320, 240, 0), 320, 240,
        str(ass), measure_loudness(str(src), 0, 1), str(out), encoders=("libx264",),
    )
    assert (enc, note) == ("libx264", "")
    info = probe(str(out))
    assert video_size(info) == (320, 240)
    assert {s["codec_name"] for s in info["streams"]} == {"h264", "aac"}
    assert float(info["format"]["duration"]) == pytest.approx(1, abs=0.1)
    frame = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(out), "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], capture_output=True, check=True).stdout
    pixels = np.frombuffer(frame, np.uint8).reshape(240, 320, 3)
    assert np.count_nonzero(pixels.max(axis=2) > 100) > 100  # visible captions on black
    pcm = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(out), "-vn", "-f", "s16le", "pipe:1",
    ], capture_output=True, check=True).stdout
    assert np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.int32)).max() > 100
