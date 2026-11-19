"""Tiny generated media and word timings shared by the pipeline and server tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

from localclip.words import Word

SENTENCE = "alpha beta gamma delta epsilon zeta theta iota."


def make_video(path: Path, seconds: int = 32, size: str = "640x360") -> Path:
    """Moving test pattern plus a sine tone: H.264/AAC, so browsers (and the editor) can play it."""
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=48000:duration={seconds}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return path


def sentence_words(count: int = 9, wps: float = 2.6, gap: float = 0.3) -> list[Word]:
    """``count`` eight-word sentences, evenly paced; each ends on a period."""
    words: list[Word] = []
    t, slot = 0.5, 1.0 / wps
    for _ in range(count):
        for tok in SENTENCE.split():
            words.append(Word(round(t, 3), round(t + slot * 0.85, 3), tok))
            t += slot
        t += gap
    return words
