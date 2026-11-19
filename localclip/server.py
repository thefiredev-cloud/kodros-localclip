"""Local web editor: upload a video, review auto-picked clips, trim, export.

Binds 127.0.0.1 only, rejects foreign Host and Origin headers, serves its own static
files (no CDN, no third-party requests). One worker thread runs analyze/export jobs
one at a time, because both use the GPU and ffmpeg heavily.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import shutil
import threading
import time
import traceback
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import pipeline, project
from .crop import CROP_MODES
from .render import has_audio, probe

DEFAULT_PORT = 8871
WEB_DIR = Path(__file__).parent / "web"
STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
MAX_JSON = 1 << 20
MAX_UPLOAD = int(float(os.environ.get("LOCALCLIP_MAX_UPLOAD_GB", "20")) * (1 << 30))
ASPECTS = {"9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "16:9": (16, 9)}
ANALYZE_KEYS = {"count": int, "min_dur": float, "max_dur": float, "use_llm": bool, "use_faces": bool,
                "transcribe_engine": str, "whisper_model": str, "long_side": int}
NEW_SELECTORS = {"manual", "candidate"}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def parse_aspect(value) -> tuple[int, int]:
    if value not in ASPECTS:
        raise ApiError(400, f"aspect must be one of {sorted(ASPECTS)}")
    return ASPECTS[value]


def _num(d: dict, key: str, required: bool = False):
    v = d.get(key)
    if v is None:
        if required:
            raise ApiError(400, f"clip needs {key}")
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ApiError(400, f"{key} must be a number")
    return float(v)


def clips_from_json(items, stored: list[dict], duration: float) -> list[pipeline.ClipSpec]:
    """Turn the editor's clip list into ClipSpecs. Known ids inherit the auto-pick metadata."""
    if not isinstance(items, list) or not items:
        raise ApiError(400, "clips must be a non-empty list")
    if len(items) > 50:
        raise ApiError(400, "at most 50 clips per export")
    by_id = {c["id"]: c for c in stored}
    specs = []
    for it in items:
        if not isinstance(it, dict):
            raise ApiError(400, "each clip must be an object")
        start, end = _num(it, "start", True), _num(it, "end", True)
        base = by_id.get(it.get("id"))
        crop = it.get("crop", "auto")
        if crop not in CROP_MODES:
            raise ApiError(400, f"crop must be one of {list(CROP_MODES)}")
        crop_x = _num(it, "crop_x")
        if crop_x is not None and not 0 <= crop_x <= 1:
            raise ApiError(400, "crop_x must be between 0 and 1")
        title = it.get("title", "")
        if not isinstance(title, str):
            raise ApiError(400, "title must be text")
        spec = pipeline.ClipSpec(
            start=start, end=end, title=title[:120], captions=bool(it.get("captions", True)),
            crop=crop, crop_x=crop_x, snap=bool(it.get("snap", False)),
        )
        if base:
            spec.why, spec.score, spec.selected_by = base["why"], base["score"], base["selected_by"]
            spec.text = base.get("text")
            spec.sentences = tuple(base["sentences"]) if base.get("sentences") else None
            spec.edited = abs(start - base["start"]) > 0.01 or abs(end - base["end"]) > 0.01
        else:
            sel = it.get("selected_by", "manual")
            spec.selected_by = sel if sel in NEW_SELECTORS else "manual"
            spec.why = str(it.get("why", ""))[:400]
            spec.edited = True
        try:
            pipeline.validate_spec(spec, duration)
        except ValueError as e:
            raise ApiError(400, f"clip {len(specs) + 1}: {e}") from None
        specs.append(spec)
    return specs


def _is_type(v, typ) -> bool:
    if typ is bool:
        return isinstance(v, bool)
    if typ is float:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if typ is int:
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, typ)


def options_from_json(body: dict, base: dict | None = None) -> pipeline.Options:
    """Whitelisted analyze/export options layered over ``base`` (stored options)."""
    merged = dict(base or {})
    for k, typ in ANALYZE_KEYS.items():
        if k in body:
            if not _is_type(body[k], typ):
                raise ApiError(400, f"{k} must be {typ.__name__}")
            merged[k] = typ(body[k])
    if "upper" in body:
        merged["upper"] = bool(body["upper"])
    if "aspect" in body:
        merged["aspect"] = parse_aspect(body["aspect"])
    merged.pop("out_dir", None)
    merged.pop("end_card", None)
    merged["aspect"] = tuple(merged.get("aspect", (9, 16)))
    opts = pipeline.Options(**merged)
    if not 1 <= opts.count <= 20:
        raise ApiError(400, "count must be 1-20")
    if not 0 < opts.min_dur < opts.max_dur <= pipeline.MAX_CLIP_S:
        raise ApiError(400, f"need 0 < min < max <= {pipeline.MAX_CLIP_S:.0f}")
    if opts.transcribe_engine not in ("auto", "parakeet", "whisper"):
        raise ApiError(400, "transcribe_engine must be auto, parakeet or whisper")
    if opts.whisper_model not in ("tiny", "base", "small", "medium", "large-v3"):
        raise ApiError(400, "whisper_model must be tiny, base, small, medium or large-v3")
    if not 320 <= opts.long_side <= 3840:
        raise ApiError(400, "long_side must be 320-3840")
    return opts


class Jobs:
    """Serial job runner: analyze and export jobs share one GPU and a few CPU cores."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.n = 0
        threading.Thread(target=self._run, daemon=True, name="localclip-jobs").start()

    def submit(self, kind: str, pid: str, fn) -> dict:
        with self.lock:
            for j in self.jobs.values():
                if j["project"] == pid and j["status"] in ("queued", "running"):
                    raise ApiError(409, f"project already has a {j['kind']} job in progress")
            self.n += 1
            job = {"id": f"j{self.n}", "kind": kind, "project": pid, "status": "queued", "log": [],
                   "result": None, "error": None, "created": time.time()}
            self.jobs[job["id"]] = job
        self.q.put((job, fn))
        return job

    def _run(self) -> None:
        while True:
            job, fn = self.q.get()
            job["status"] = "running"
            try:
                job["result"] = fn(lambda s: job["log"].append(str(s)))
                job["status"] = "done"
            except BaseException as e:  # noqa: BLE001 - surfaced to the UI; SystemExit included
                job["error"] = str(e) or type(e).__name__
                job["log"].append(traceback.format_exc(limit=3))
                job["status"] = "error"
            finally:
                job["finished"] = time.time()

    def get(self, jid: str) -> dict:
        j = self.jobs.get(jid)
        if not j:
            raise ApiError(404, "no such job")
        return {**j, "log": j["log"][-200:]}

    def active_for(self, pid: str) -> dict | None:
        for j in self.jobs.values():
            if j["project"] == pid and j["status"] in ("queued", "running"):
                return self.get(j["id"])
        return None


def send_file(h: BaseHTTPRequestHandler, path: Path, ctype: str | None = None, head: bool = False,
              download: str | None = None) -> None:
    """Serve a file with HTTP Range support, which browsers need for video seeking."""
    size = path.stat().st_size
    start, end, status = 0, size - 1, HTTPStatus.OK
    rng = h.headers.get("Range")
    if rng:
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        if not m or (not m.group(1) and not m.group(2)):
            h.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:
            start = max(0, size - int(m.group(2)))
        if start > end or start >= size:
            h.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            h.send_header("Content-Range", f"bytes */{size}")
            h.send_header("Content-Length", "0")
            h.end_headers()
            return
        status = HTTPStatus.PARTIAL_CONTENT
    length = end - start + 1
    h.send_response(status)
    h.send_header("Content-Type", ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    h.send_header("Accept-Ranges", "bytes")
    h.send_header("Content-Length", str(length))
    if status == HTTPStatus.PARTIAL_CONTENT:
        h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    if download:
        h.send_header("Content-Disposition", f'attachment; filename="{download}"')
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    if head:
        return
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while left:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                break
            try:
                h.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            left -= len(chunk)


def make_handler(store: project.Store, jobs: Jobs, port: int):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in hosts}

    class Handler(BaseHTTPRequestHandler):
        server_version = "localclip"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet; the CLI prints the URL once
            pass

        # -- plumbing ---------------------------------------------------------
        def _guard(self) -> bool:
            if self.headers.get("Host", "") not in hosts:
                self._fail(403, "foreign Host header")
                return False
            origin = self.headers.get("Origin")
            if self.command not in ("GET", "HEAD") and origin not in origins:
                self._fail(403, "foreign or missing Origin")
                return False
            return True

        def _json(self, status: int, data) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_JSON:
                raise ApiError(413, "body too large")
            raw = self.rfile.read(n) if n else b"{}"
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                raise ApiError(400, "body is not valid JSON") from None
            if not isinstance(data, dict):
                raise ApiError(400, "body must be a JSON object")
            return data

        def _fail(self, status: int, message: str) -> None:
            # A failed POST may leave its body unread; do not reuse the connection.
            if self.command not in ("GET", "HEAD"):
                self.close_connection = True
                self.send_response(status)
                body = json.dumps({"error": message}).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(status, {"error": message})

        def _dispatch(self, fn) -> None:
            try:
                if self._guard():
                    fn()
            except ApiError as e:
                self._fail(e.status, e.message)
            except KeyError:
                self._fail(404, "not found")
            except FileNotFoundError as e:
                self._fail(404, str(e))
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as e:  # noqa: BLE001
                self._fail(500, f"{type(e).__name__}: {e}")

        def do_GET(self):
            self._dispatch(lambda: self._get(False))

        def do_HEAD(self):
            self._dispatch(lambda: self._get(True))

        def do_POST(self):
            self._dispatch(self._post)

        # -- GET --------------------------------------------------------------
        def _get(self, head: bool) -> None:
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            if u.path in STATIC:
                f = WEB_DIR / STATIC[u.path]
                return send_file(self, f, mimetypes.guess_type(f.name)[0] or "text/plain", head)
            if parts[:1] != ["api"]:
                raise KeyError(u.path)
            if parts == ["api", "projects"]:
                return self._json(200, {"projects": store.list()})
            if len(parts) >= 3 and parts[1] == "projects":
                pid = parts[2]
                rest = parts[3:]
                if not rest:
                    meta = store.load(pid)
                    meta["job"] = jobs.active_for(pid)
                    return self._json(200, meta)
                if rest == ["transcript"]:
                    return self._json(200, {"words": [w.to_json() for w in store.words(pid)]})
                if rest == ["video"]:
                    return send_file(self, store.playable(pid), "video/mp4" if (store.dir(pid) / "preview.mp4").exists() else None, head)
                if len(rest) == 3 and rest[0] == "exports":
                    f = store.export_file(pid, rest[1], rest[2])
                    dl = f"{pid}-{rest[1]}-{f.name}" if f.suffix == ".mp4" else None
                    return send_file(self, f, None, head, download=dl if parse_qs(u.query).get("dl") else None)
            if len(parts) == 3 and parts[1] == "jobs":
                return self._json(200, jobs.get(parts[2]))
            raise KeyError(u.path)

        # -- POST -------------------------------------------------------------
        def _post(self) -> None:
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            if parts == ["api", "upload"]:
                return self._upload(parse_qs(u.query).get("name", ["video.mp4"])[0])
            if len(parts) == 4 and parts[:2] == ["api", "projects"]:
                pid, action = parts[2], parts[3]
                if action == "analyze":
                    return self._analyze(pid)
                if action == "export":
                    return self._export(pid)
            raise KeyError(u.path)

        def _upload(self, name: str) -> None:
            n = self.headers.get("Content-Length")
            if n is None or not n.isdigit() or int(n) == 0:
                raise ApiError(411, "upload needs a Content-Length")
            n = int(n)
            if n > MAX_UPLOAD:
                self.close_connection = True
                raise ApiError(413, f"file larger than {MAX_UPLOAD >> 30} GB (LOCALCLIP_MAX_UPLOAD_GB)")
            if shutil.disk_usage(store.root).free < n * 2 + (2 << 30):
                self.close_connection = True
                raise ApiError(507, "not enough free disk space for this upload")
            pid = store.create(name)
            try:
                dest = store.upload_path(pid, name)
            except ValueError as e:
                shutil.rmtree(store.dir(pid))
                self.close_connection = True
                raise ApiError(415, str(e)) from None
            left = n
            try:
                with open(dest, "wb") as f:
                    os.chmod(dest, 0o600)
                    while left:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            raise ApiError(400, "upload ended early")
                        f.write(chunk)
                        left -= len(chunk)
                info = probe(str(dest))
                if not any(s["codec_type"] == "video" for s in info["streams"]):
                    raise ApiError(415, "file has no video stream")
                if not has_audio(info):
                    raise ApiError(415, "file has no audio stream; nothing to transcribe")
            except ApiError:
                shutil.rmtree(store.dir(pid), ignore_errors=True)
                self.close_connection = True
                raise
            except Exception:
                shutil.rmtree(store.dir(pid), ignore_errors=True)
                self.close_connection = True
                raise ApiError(415, "file is not a readable video") from None
            self._json(201, {"id": pid})

        def _analyze(self, pid: str) -> None:
            store.dir(pid)
            opts = options_from_json(self._body())
            meta = store.load(pid)
            if meta["status"] == "analyzing":
                raise ApiError(409, "already analyzing")
            job = jobs.submit("analyze", pid, lambda log: project.analyze_project(store, pid, opts, log=log) and None)
            self._json(202, {"job": job["id"]})

        def _export(self, pid: str) -> None:
            body = self._body()
            meta = store.load(pid)
            if meta["status"] != "ready":
                raise ApiError(409, "project is not analyzed yet")
            specs = clips_from_json(body.get("clips"), meta.get("clips", []), meta["duration"])
            opts = options_from_json(body, meta.get("options"))
            job = jobs.submit(
                "export", pid,
                lambda log: project.export_project(store, pid, [s.to_dict() for s in specs], opts, log=log),
            )
            self._json(202, {"job": job["id"]})

    return Handler


def serve(port: int = DEFAULT_PORT, root: Path | None = None, ready=None) -> None:
    store = project.Store(root)
    jobs = Jobs()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, jobs, port))
    httpd.daemon_threads = True
    if ready:
        ready(httpd)
    print(f"localclip editor: http://127.0.0.1:{port}/  (data: {store.root})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


__all__ = ["DEFAULT_PORT", "ASPECTS", "Jobs", "clips_from_json", "make_handler", "options_from_json", "serve"]
