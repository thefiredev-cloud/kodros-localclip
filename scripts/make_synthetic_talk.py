#!/usr/bin/env python3
"""Build a synthetic 16:9 talking-head test video: Kokoro TTS narration over a
still portrait that drifts across the frame (so face tracking has work to do).

Needs kokoro_onnx and onnxruntime, plus the Kokoro model files. Point KOKORO_DIR at the folder
that holds kokoro-v1.0.onnx and voices-v1.0.bin (default ~/.local/share/kokoro):

    python scripts/make_synthetic_talk.py \
        fixtures/synthetic-talk.txt fixtures/astronaut-public-domain.png OUT.mp4

The output is synthetic. It is a pipeline test input, not real footage.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

HOME = Path.home()
KOKORO_DIR = Path(os.environ.get("KOKORO_DIR", str(HOME / ".local/share/kokoro")))
MODEL = KOKORO_DIR / "kokoro-v1.0.onnx"
VOICES = KOKORO_DIR / "voices-v1.0.bin"
RATE = 24000


def synth(paragraphs: list[str], voice: str = "am_michael", speed: float = 1.0) -> np.ndarray:
    from kokoro_onnx import Kokoro

    k = Kokoro(str(MODEL), str(VOICES))
    gap = np.zeros(int(RATE * 0.6), dtype=np.float32)
    out: list[np.ndarray] = []
    for p in paragraphs:
        samples, rate = k.create(p, voice=voice, speed=speed, lang="en-us")
        if rate != RATE:
            raise RuntimeError(f"unexpected rate {rate}")
        out += [np.asarray(samples, dtype=np.float32), gap]
    return np.concatenate(out)


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    script, portrait, dest = map(Path, sys.argv[1:])
    paragraphs = [p.strip() for p in script.read_text().split("\n\n") if p.strip()]
    pcm = synth(paragraphs)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "voice.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
        dur = len(pcm) / RATE
        # Portrait 760 px tall drifts left-right on a dark background.
        graph = (
            f"color=c=0x1b2230:s=1920x1080:r=30:d={dur:.2f}[bg];"
            "[0:v]scale=-2:760,format=rgba[p];"
            "[bg][p]overlay=x='120+(W-w-240)*(0.5+0.5*sin(2*PI*t/40))':y=(H-h)/2,"
            "format=yuv420p[v]"
        )
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-i", str(portrait),
            "-i", str(wav),
            "-filter_complex", graph,
            "-map", "[v]", "-map", "1:a",
            "-t", f"{dur:.2f}",
            "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            str(dest),
        ]
        subprocess.run(cmd, check=True)
    print(f"wrote {dest} ({dur:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
