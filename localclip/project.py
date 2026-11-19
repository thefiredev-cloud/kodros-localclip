"""On-disk projects for the editor: source, transcript, picked clips and exports.

Layout under ``$LOCALCLIP_HOME/projects/<id>/`` (mode 0700, never in git)::

    project.json     metadata, selection result, candidate moments, clips, exports
    transcript.json  word timings
    source.<ext>     uploaded video (or ``source_path`` in project.json for CLI imports)
    preview.mp4      browser-playable proxy, only when the source is not H.264/AAC
    exports/<id>/    clipNN.mp4, manifest.json, sheet.png
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import pipeline
from .render import has_audio, probe
from .words import Word, words_from_json

ID_RE = re.compile(r"^p[0-9a-f]{10}$")
EXPORT_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z(-[0-9]+)?$")
FILE_RE = re.compile(r"^(clip[0-9]{2}\.mp4|sheet\.png|manifest\.json)$")


def _private(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    os.chmod(p, 0o700)
    return p


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def safe_name(name: str) -> str:
    base = os.path.basename(name or "video.mp4")
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .") or "video.mp4"
    return base[:120]


class Store:
    def __init__(self, root: Path | None = None):
        self.root = _private((root or pipeline.DATA_HOME) / "projects")

    def dir(self, pid: str) -> Path:
        if not ID_RE.match(pid):
            raise KeyError(pid)
        d = self.root / pid
        if not d.is_dir():
            raise KeyError(pid)
        return d

    def create(self, name: str, source: str | None = None) -> str:
        pid = "p" + secrets.token_hex(5)
        d = _private(self.root / pid)
        meta = {
            "id": pid, "name": safe_name(name), "source": source, "status": "new", "error": None,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "clips": [], "candidates": [], "exports": [],
        }
        _write_json(d / "project.json", meta)
        return pid

    def upload_path(self, pid: str, filename: str) -> Path:
        ext = Path(safe_name(filename)).suffix.lower()
        if ext not in pipeline.VIDEO_EXTS:
            raise ValueError(f"unsupported file type {ext or '(none)'}; use one of {sorted(pipeline.VIDEO_EXTS)}")
        return self.dir(pid) / f"source{ext}"

    def load(self, pid: str) -> dict:
        return json.loads((self.dir(pid) / "project.json").read_text())

    def save(self, meta: dict) -> None:
        _write_json(self.dir(meta["id"]) / "project.json", meta)

    def update(self, pid: str, **fields) -> dict:
        meta = self.load(pid)
        meta.update(fields)
        self.save(meta)
        return meta

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if ID_RE.match(d.name) and (d / "project.json").exists():
                m = json.loads((d / "project.json").read_text())
                out.append({k: m.get(k) for k in ("id", "name", "status", "created", "duration")}
                           | {"clips": len(m.get("clips", [])), "exports": len(m.get("exports", []))})
        out.sort(key=lambda m: m["created"] or "", reverse=True)
        return out

    def words(self, pid: str) -> list[Word]:
        p = self.dir(pid) / "transcript.json"
        return words_from_json(json.loads(p.read_text())["words"]) if p.exists() else []

    def source(self, pid: str) -> Path:
        meta = self.load(pid)
        if meta.get("source"):
            return Path(meta["source"])
        found = sorted(self.dir(pid).glob("source.*"))
        if not found:
            raise FileNotFoundError("project has no source video")
        return found[0]

    def playable(self, pid: str) -> Path:
        pv = self.dir(pid) / "preview.mp4"
        return pv if pv.exists() else self.source(pid)

    def export_file(self, pid: str, export_id: str, name: str) -> Path:
        if not EXPORT_RE.match(export_id) or not FILE_RE.match(name):
            raise KeyError(name)
        p = self.dir(pid) / "exports" / export_id / name
        if not p.is_file():
            raise KeyError(name)
        return p


def import_path(store: Store, path: str) -> str:
    """Register a local video as a project without copying it."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(path)
    probe(str(p))  # raises CalledProcessError for non-media
    return store.create(p.name, source=str(p))


def browser_safe(info: dict) -> bool:
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    return bool(v and a and v.get("codec_name") == "h264" and v.get("pix_fmt") == "yuv420p"
                and a.get("codec_name") in ("aac", "mp3"))


def ensure_preview(store: Store, pid: str, info: dict, log=print) -> str | None:
    """Make a 540p H.264 proxy when the browser could not play the source. Returns its name or None."""
    if browser_safe(info):
        return None
    dest = store.dir(pid) / "preview.mp4"
    src = str(store.source(pid))
    for enc in (["-c:v", "h264_nvenc", "-preset", "p3", "-cq", "30"], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"]):
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vf", "scale=-2:540,format=yuv420p",
             *enc, "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dest)],
            capture_output=True, text=True)
        if r.returncode == 0:
            os.chmod(dest, 0o600)
            log("preview: made a 540p H.264 proxy (source is not browser-playable)")
            return dest.name
    raise RuntimeError("could not make a preview proxy: " + r.stderr.strip()[-300:])


def analyze_project(store: Store, pid: str, opts: pipeline.Options, log=print) -> dict:
    """Probe, transcribe and pick moments for a project. Saves everything the editor needs."""
    store.update(pid, status="analyzing", error=None)
    try:
        src = str(store.source(pid))
        info = probe(src)
        if not has_audio(info):
            raise ValueError("input has no audio stream; nothing to transcribe")
        preview = ensure_preview(store, pid, info, log)
        an = pipeline.analyze(src, opts, store.dir(pid) / "work", log=log)
        _write_json(store.dir(pid) / "transcript.json", an.transcript.to_json())
        sel = an.selection
        meta = store.load(pid)
        meta.update({
            "status": "ready", "error": None,
            "duration": round(an.duration, 3), "size": list(an.size), "preview": preview,
            "transcription": {"engine": an.transcript.engine, "note": an.transcript.note,
                              "language": an.transcript.language, "words": len(an.transcript.words)},
            "selection": {"engine": sel.engine, "note": sel.note},
            "candidates": [
                {"start": round(c.start, 3), "end": round(c.end, 3), "score": round(c.score, 1),
                 "text": c.text, "reasons": c.reasons}
                for c in sel.shortlist
            ],
            "clips": [dict(s.to_dict(), id=i + 1) for i, s in enumerate(an.specs)],
            "options": asdict(opts), "runtime": an.stages,
        })
        store.save(meta)
        return meta
    except BaseException as e:  # SystemExit from pipeline.analyze included
        store.update(pid, status="error", error=str(e) or type(e).__name__)
        raise


def export_project(store: Store, pid: str, clips: list[dict], opts: pipeline.Options, log=print) -> dict:
    """Export the given clip dicts (see ``ClipSpec``) from a project. Returns the export record."""
    meta = store.load(pid)
    if meta.get("status") != "ready":
        raise ValueError("project is not analyzed yet")
    if not clips:
        raise ValueError("no clips selected")
    specs = [pipeline.ClipSpec.from_dict(c) for c in clips]
    words = store.words(pid)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_id, n = stamp, 1
    while (store.dir(pid) / "exports" / export_id).exists():
        n += 1
        export_id = f"{stamp}-{n}"
    out_dir = store.dir(pid) / "exports" / export_id
    t0 = time.monotonic()
    done = pipeline.export_clips(str(store.source(pid)), words, specs, opts, out_dir,
                                 duration=meta["duration"], log=log)
    record = {
        "id": export_id, "created": stamp, "aspect": f"{opts.aspect[0]}:{opts.aspect[1]}",
        "seconds": round(time.monotonic() - t0, 2), "sheet": "sheet.png" if done else None,
        "clips": [{"name": Path(c["file"]).name, "start": c["start"], "end": c["end"],
                   "duration": c["duration"], "title": c["title"], "crop": c["crop"]["mode"],
                   "captions": c["captions"]["enabled"], "encoder": c["encoder"]} for c in done],
    }
    _write_json(out_dir / "manifest.json", {"project": pid, "options": asdict(opts), "clips": done})
    meta = store.load(pid)
    meta["exports"].append(record)
    store.save(meta)
    return record


__all__ = ["Store", "analyze_project", "export_project", "import_path"]
