"""Word-timestamp transcription.

Order: the local Parakeet service (127.0.0.1:8798) is probed first and used
only if it returns per-word timestamps. Today it returns plain text, so the
fallback, faster-whisper on the GPU, does the work. The chosen engine and the
reason are recorded in the manifest.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import subprocess
import time
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .words import Word

PARAKEET_URL = os.environ.get("LOCALCLIP_PARAKEET_URL", "http://127.0.0.1:8798/v1/audio/transcriptions")
PARAKEET_TOKEN = Path(os.environ.get("LOCALCLIP_PARAKEET_TOKEN_FILE", str(Path.home() / ".config/localclip/stt-token")))


@dataclass
class Transcript:
    words: list[Word]
    engine: str
    note: str
    language: str | None
    seconds: float

    def to_json(self) -> dict:
        return {
            "engine": self.engine,
            "note": self.note,
            "language": self.language,
            "seconds": round(self.seconds, 2),
            "words": [w.to_json() for w in self.words],
        }


def extract_audio(src: str, dest: str, seconds: float | None = None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dest]
    subprocess.run(cmd, check=True)


def _multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
        + data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _parakeet_request(wav: str, timeout: float) -> dict:
    token = PARAKEET_TOKEN.read_text().strip()
    body, ctype = _multipart(
        {"response_format": "verbose_json", "timestamp_granularities[]": "word"},
        "file",
        "audio.wav",
        Path(wav).read_bytes(),
    )
    req = urllib.request.Request(
        PARAKEET_URL, data=body, headers={"Content-Type": ctype, "Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _parakeet_words(data: dict) -> list[Word] | None:
    items = data.get("words")
    if not isinstance(items, list) or not items:
        return None
    return [Word(float(w["start"]), float(w["end"]), str(w["word"]).strip()) for w in items]


def probe_parakeet(src: str, workdir: str) -> tuple[bool, str]:
    """Send 8 s of audio; report whether word timestamps come back."""
    sample = os.path.join(workdir, "parakeet-probe.wav")
    try:
        extract_audio(src, sample, seconds=8)
        data = _parakeet_request(sample, timeout=30)
    except Exception as e:
        return False, f"parakeet probe failed ({type(e).__name__})"
    finally:
        if os.path.exists(sample):
            os.unlink(sample)
    if _parakeet_words(data) is None:
        return False, f"parakeet returned no word timestamps (keys: {sorted(data)})"
    return True, "parakeet returned word timestamps"


def _preload_cuda_libs() -> None:
    """ctranslate2 needs cuBLAS 12 and cuDNN 9; load the pip wheels' copies globally."""
    import nvidia  # namespace package from nvidia-cublas-cu12 / nvidia-cudnn-cu12

    roots = list(getattr(nvidia, "__path__", []))
    order = ["cublas/lib/libcublasLt.so.*", "cublas/lib/libcublas.so.*", "cudnn/lib/libcudnn*.so.*"]
    for pattern in order:
        for root in roots:
            for lib in sorted(glob.glob(os.path.join(root, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def whisper_words(wav: str, model_name: str = "small", device: str = "cuda") -> tuple[list[Word], str, str]:
    from faster_whisper import WhisperModel

    note = ""
    if device == "cuda":
        try:
            _preload_cuda_libs()
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
            engine = f"faster-whisper {model_name} (cuda float16)"
        except Exception as e:
            note = f"CUDA load failed ({type(e).__name__}); "
            device = "cpu"
    if device == "cpu":
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        engine = f"faster-whisper {model_name} (cpu int8)"
    # Pass samples, not a path: faster-whisper's PyAV decode path breaks on some av releases.
    with wave.open(wav, "rb") as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    segments, info = model.transcribe(audio, word_timestamps=True, vad_filter=True, beam_size=5)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(round(float(w.start), 3), round(float(w.end), 3), text))
    return words, engine, note + f"language {info.language} p={info.language_probability:.2f}"


def transcribe(src: str, workdir: str, engine: str = "auto", whisper_model: str = "small", gpu: bool = True) -> Transcript:
    t0 = time.monotonic()
    notes: list[str] = []
    wav = os.path.join(workdir, "audio16k.wav")
    extract_audio(src, wav)
    try:
        if engine in ("auto", "parakeet"):
            ok, why = probe_parakeet(src, workdir)
            notes.append(why)
            if ok:
                words = _parakeet_words(_parakeet_request(wav, timeout=600)) or []
                return Transcript(words, "parakeet-tdt-0.6b-v2 (127.0.0.1:8798)", "; ".join(notes), "en", time.monotonic() - t0)
            if engine == "parakeet":
                raise RuntimeError("parakeet requested but it does not return word timestamps: " + why)
        words, eng, wnote = whisper_words(wav, whisper_model, "cuda" if gpu else "cpu")
        notes.append(wnote)
        lang = wnote.split("language ")[-1].split(" ")[0] if "language " in wnote else None
        return Transcript(words, eng, "; ".join(notes), lang, time.monotonic() - t0)
    finally:
        if os.path.exists(wav):
            os.unlink(wav)
