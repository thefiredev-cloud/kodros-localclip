"""Pipeline: analyze (transcribe, pick moments) then export (trim, reframe, caption, encode).

``make`` is analyze plus export of the picked clips. The web UI and the ``analyze`` /
``export`` CLI commands use the two halves separately, so a person can adjust each
clip's start/end, captions and framing between them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .captions import CaptionStyle, build_ass, caption_events, relative_words
from .crop import CROP_MODES, out_size, plan_crop
from .gpu import GPUMonitor, gpu_reserved_running
from .llm import DEFAULT_MODEL, make_cluster_llm
from .render import has_audio, measure_loudness, probe, render_clip, video_size
from .select import Selection, select_clips
from .sheet import contact_sheet
from .transcribe import Transcript, transcribe
from .words import Word, split_sentences, words_from_json

DATA_HOME = Path(os.environ.get("LOCALCLIP_HOME", str(Path.home() / ".local/share/localclip")))

# Pads around a picked moment so the first and last word are not clipped mid-phoneme.
PAD_BEFORE = 0.15
PAD_AFTER = 0.35
MIN_CLIP_S = 1.0
MAX_CLIP_S = 180.0
SNAP_BEFORE = 0.05
SNAP_AFTER = 0.12


@dataclass
class Options:
    count: int = 3
    aspect: tuple[int, int] = (9, 16)
    min_dur: float = 20.0
    max_dur: float = 60.0
    out_dir: str | None = None
    end_card: str | None = None
    end_card_sec: float = 2.5
    use_llm: bool = True
    llm_timeout: float = 240.0
    use_faces: bool = True
    transcribe_engine: str = "auto"
    whisper_model: str = "small"
    upper: bool = True
    long_side: int = 1920  # output pixels on the long edge; 1920 gives 1080x1920 for 9:16


@dataclass
class ClipSpec:
    """One clip to export. ``start``/``end`` are final source seconds (no extra pad)."""

    start: float
    end: float
    title: str = ""
    why: str = ""
    score: float | None = None
    selected_by: str = "manual"
    text: str | None = None
    sentences: tuple[int, int] | None = None
    captions: bool = True
    crop: str = "auto"  # auto | face | center
    crop_x: float | None = None  # 0..1 manual crop center, overrides ``crop``
    snap: bool = False  # nudge start/end off the middle of a word
    edited: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "ClipSpec":
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        if kw.get("sentences") is not None:
            kw["sentences"] = tuple(kw["sentences"])
        return cls(**kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sentences"] = list(self.sentences) if self.sentences else None
        return d


@dataclass
class Analysis:
    src: str
    duration: float
    size: tuple[int, int]
    transcript: Transcript
    selection: Selection
    specs: list[ClipSpec]
    stages: dict[str, float]
    gpu_reserved_gate: bool
    gpu: dict
    options: Options


def _private_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    os.chmod(p, 0o700)
    return p


def _cache_key(src: str, opts: Options) -> str:
    st = os.stat(src)
    raw = f"{os.path.abspath(src)}|{st.st_size}|{int(st.st_mtime)}|{opts.transcribe_engine}|{opts.whisper_model}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _load_or_transcribe(src: str, workdir: Path, opts: Options, gpu_ok: bool, log) -> Transcript:
    cache = _private_dir(DATA_HOME / "cache") / f"{_cache_key(src, opts)}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        log(f"transcript: cache hit {cache.name}")
        return Transcript(words_from_json(d["words"]), d["engine"], d["note"] + "; from cache", d.get("language"), 0.0)
    t = transcribe(src, str(workdir), opts.transcribe_engine, opts.whisper_model, gpu=gpu_ok)
    cache.write_text(json.dumps(t.to_json()))
    os.chmod(cache, 0o600)
    return t


def snap_to_words(words: list[Word], start: float, end: float) -> tuple[float, float]:
    """Move a cut that lands inside a word to just before (start) or after (end) it."""
    for w in words:
        if w.start < start < w.end:
            start = max(0.0, w.start - SNAP_BEFORE)
            break
    for w in words:
        if w.start < end < w.end:
            end = w.end + SNAP_AFTER
            break
    return start, end


def validate_spec(spec: ClipSpec, duration: float) -> ClipSpec:
    """Clamp to the source and reject clips that are inverted, too short or too long."""
    if spec.crop not in CROP_MODES:
        raise ValueError(f"crop must be one of {CROP_MODES}, got {spec.crop!r}")
    if spec.crop_x is not None and not 0.0 <= spec.crop_x <= 1.0:
        raise ValueError("crop_x must be between 0 and 1")
    start, end = max(0.0, float(spec.start)), min(duration, float(spec.end))
    if end - start < MIN_CLIP_S:
        raise ValueError(f"clip must be at least {MIN_CLIP_S:.0f}s (got {end - start:.2f}s after clamping to 0-{duration:.1f}s)")
    if end - start > MAX_CLIP_S:
        raise ValueError(f"clip must be at most {MAX_CLIP_S:.0f}s (got {end - start:.1f}s)")
    spec.start, spec.end = start, end
    return spec


def clip_text(words: list[Word], start: float, end: float) -> str:
    return " ".join(w.text.strip() for w in words if w.start >= start - 0.05 and w.end <= end + 0.05).strip()


def analyze(src: str, opts: Options, work: Path, log=print) -> Analysis:
    """Transcribe the source and pick moments. Writes only scratch files under ``work``."""
    src = os.path.abspath(src)
    info = probe(src)
    if not has_audio(info):
        raise SystemExit("input has no audio stream; nothing to transcribe")
    size = video_size(info)
    duration = float(info["format"]["duration"])
    work = _private_dir(work)
    reserved = gpu_reserved_running()
    gpu_ok = not reserved
    if reserved:
        log("a GPU-reserved process is running: GPU work disabled (CPU whisper, libx264)")
    stages: dict[str, float] = {}
    with GPUMonitor() as mon:
        t0 = time.monotonic()
        tr = _load_or_transcribe(src, work, opts, gpu_ok, log)
        stages["transcribe_s"] = round(time.monotonic() - t0, 2)
        log(f"transcript: {len(tr.words)} words via {tr.engine} in {stages['transcribe_s']}s")
        sentences = split_sentences(tr.words)

        t0 = time.monotonic()
        llm = make_cluster_llm(timeout=opts.llm_timeout) if opts.use_llm else None
        sel = select_clips(tr.words, sentences, opts.count, llm=llm, min_dur=opts.min_dur, max_dur=opts.max_dur)
        stages["select_s"] = round(time.monotonic() - t0, 2)
        log(f"selection: {sel.engine} ({sel.note}) in {stages['select_s']}s")
    specs = [
        ClipSpec(
            start=max(0.0, p.start - PAD_BEFORE), end=min(duration, p.end + PAD_AFTER), title=p.title,
            why=p.reason, score=p.score, selected_by=p.selector, text=p.text,
            sentences=(p.first_sentence, p.last_sentence),
        )
        for p in sel.picks
    ]
    return Analysis(src, duration, size, tr, sel, specs, stages, reserved, mon.summary(), opts)


def export_clips(
    src: str, words: list[Word], specs: list[ClipSpec], opts: Options, out_dir: Path,
    duration: float | None = None, log=print, sheet: bool = True,
) -> list[dict]:
    """Render ``specs`` to ``out_dir/clipNN.mp4``. Returns the manifest entry of each clip."""
    src = os.path.abspath(src)
    info = probe(src)
    src_w, src_h = video_size(info)
    duration = duration or float(info["format"]["duration"])
    out_dir = _private_dir(out_dir)
    work = _private_dir(out_dir / "work")
    gpu_ok = not gpu_reserved_running()
    encoders = ("h264_nvenc", "libx264") if gpu_ok else ("libx264",)
    out_w, out_h = out_size(opts.aspect, opts.long_side)
    style = CaptionStyle(upper=opts.upper)
    clips: list[dict] = []
    for n, spec in enumerate(specs, 1):
        t0 = time.monotonic()
        if spec.snap:
            spec.start, spec.end = snap_to_words(words, spec.start, spec.end)
        validate_spec(spec, duration)
        start, end = spec.start, spec.end
        dur = end - start
        crop_mode = spec.crop if spec.crop != "auto" or opts.use_faces else "center"
        plan = plan_crop(src, start, dur, src_w, src_h, opts.aspect, use_faces=opts.use_faces,
                         mode=crop_mode, manual_x=spec.crop_x)
        rel = relative_words(words, start, end)
        events = caption_events(rel, dur, style) if spec.captions else []
        ass_path = None
        if spec.captions:
            ass_path = work / f"clip{n:02d}.ass"
            ass_path.write_text(build_ass(events, out_w, out_h, style))
        loud = measure_loudness(src, start, dur)
        out_path = out_dir / f"clip{n:02d}.mp4"
        enc, enc_note = render_clip(
            src, start, dur, plan, out_w, out_h, str(ass_path) if ass_path else None, loud, str(out_path),
            end_card=opts.end_card, end_card_sec=opts.end_card_sec, encoders=encoders,
        )
        secs = round(time.monotonic() - t0, 2)
        log(f"clip {n}: {start:.1f}-{end:.1f}s crop={plan.mode} enc={enc} {secs}s -> {out_path.name}")
        clips.append({
            "file": str(out_path),
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(dur, 3),
            "output_duration": round(float(probe(str(out_path))["format"]["duration"]), 3),
            "title": spec.title or clip_text(words, start, end)[:60],
            "why": spec.why,
            "score": spec.score,
            "selected_by": spec.selected_by,
            "edited": spec.edited,
            "sentences": list(spec.sentences) if spec.sentences else None,
            "transcript": spec.text if spec.text is not None and not spec.edited else clip_text(words, start, end),
            "crop": {"mode": plan.mode, "box": [plan.w, plan.h], "keyframes": len(plan.keyframes),
                     "face_ratio": plan.face_ratio},
            "captions": {"enabled": spec.captions, "events": len(events), "words": len(rel),
                         "ass": str(ass_path) if ass_path else None},
            "loudness_in": {k: loud[k] for k in ("input_i", "input_tp", "input_lra")},
            "loudness_target": {"I": -14.0, "TP": -1.5, "LRA": 11.0},
            "encoder": enc,
            "encoder_note": enc_note.strip() or None,
            "end_card": opts.end_card,
            "render_s": secs,
        })
    if sheet and clips:
        contact_sheet([c["file"] for c in clips], str(out_dir / "sheet.png"), labels=[Path(c["file"]).name for c in clips])
    return clips


def make(src: str, opts: Options, log=print) -> dict:
    t_start = time.monotonic()
    src = os.path.abspath(src)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _private_dir(Path(opts.out_dir) if opts.out_dir else DATA_HOME / "runs" / f"{Path(src).stem}-{stamp}")
    an = analyze(src, opts, out_dir / "work", log=log)
    clips = export_clips(src, an.transcript.words, an.specs, opts, out_dir, duration=an.duration, log=log)
    stages = dict(an.stages)
    stages["render_s"] = round(sum(c["render_s"] for c in clips), 2)
    tr, sel = an.transcript, an.selection
    manifest = {
        "tool": f"localclip {__version__}",
        "created_utc": stamp,
        "input": {"path": src, "duration": round(an.duration, 3), "size": list(an.size)},
        "output": {"aspect": f"{opts.aspect[0]}:{opts.aspect[1]}", "size": list(out_size(opts.aspect, opts.long_side)),
                   "dir": str(out_dir), "contact_sheet": str(out_dir / "sheet.png")},
        "transcription": {"engine": tr.engine, "note": tr.note, "language": tr.language, "words": len(tr.words)},
        "selection": {"engine": sel.engine, "note": sel.note,
                      "llm_model": DEFAULT_MODEL if opts.use_llm else None},
        "gpu_reserved_gate": an.gpu_reserved_gate,
        "clips": clips,
        "runtime": {**stages, "total_s": round(time.monotonic() - t_start, 2)},
        "gpu": an.gpu,
        "options": {k: v for k, v in asdict(opts).items()},
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    os.chmod(mpath, 0o600)
    log(f"manifest: {mpath}")
    return manifest


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpg", ".mpeg", ".ts"}


def expand_inputs(paths: list[str]) -> list[str]:
    """Files stay; directories expand to their video files (non-recursive, sorted)."""
    out: list[str] = []
    for p in paths:
        pp = Path(p).expanduser()
        if pp.is_dir():
            out += sorted(str(f) for f in pp.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
        elif pp.is_file():
            out.append(str(pp))
        else:
            raise FileNotFoundError(p)
    return out


def batch_make(paths: list[str], opts: Options, log=print) -> list[dict]:
    """Run ``make`` over many inputs. One failure is recorded and the batch goes on.

    With ``opts.out_dir`` set, each input gets ``out_dir/<stem>/``; otherwise the
    default run directory per input.
    """
    results: list[dict] = []
    files = expand_inputs(paths)
    if not files:
        raise ValueError("no video files found in the given inputs")
    for i, f in enumerate(files, 1):
        log(f"[{i}/{len(files)}] {f}")
        o = Options(**{**asdict(opts), "out_dir": str(Path(opts.out_dir) / Path(f).stem) if opts.out_dir else None})
        o.aspect = tuple(opts.aspect)
        try:
            m = make(f, o, log=log)
            results.append({"input": f, "ok": True, "manifest": str(Path(m["output"]["dir"]) / "manifest.json"),
                            "clips": [c["file"] for c in m["clips"]]})
        except (Exception, SystemExit) as e:  # noqa: BLE001 - report and continue with the next file
            msg = "not a readable video file (ffprobe failed)" if isinstance(e, subprocess.CalledProcessError) else str(e)
            log(f"  failed: {msg}")
            results.append({"input": f, "ok": False, "error": msg})
    return results


__all__ = ["Analysis", "ClipSpec", "Options", "analyze", "batch_make", "export_clips", "make",
           "snap_to_words", "validate_spec"]
