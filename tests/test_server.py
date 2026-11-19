"""The local editor server: guards, upload, analyze, trim export, range serving.

Real HTTP over loopback against a server on a free port. Transcription is synthetic and
the LLM is off, so these need ffmpeg but no models or services.
"""

from __future__ import annotations

import http.client
import json
import re
import shutil
import socket
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from localclip import pipeline, project, server
from localclip.transcribe import Transcript

from media import make_video, sentence_words

pytestmark = pytest.mark.skipif(not all(shutil.which(x) for x in ("ffmpeg", "ffprobe")), reason="ffmpeg missing")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("media") / "talk.mp4")


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA_HOME", tmp_path / "home")
    monkeypatch.setattr(
        pipeline, "_load_or_transcribe",
        lambda *a, **k: Transcript(sentence_words(), "synthetic", "test words", "en", 0.0),
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    store = project.Store()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), server.make_handler(store, server.Jobs(), port))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield type("S", (), {"port": port, "store": store})
    httpd.shutdown()
    httpd.server_close()


def call(srv, method, path, body=None, headers=None, host=None, origin="auto"):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=60)
    h = {"Host": host or f"127.0.0.1:{srv.port}"}
    if method == "POST" and origin == "auto":
        h["Origin"] = f"http://127.0.0.1:{srv.port}"
    elif origin not in ("auto", None):
        h["Origin"] = origin
    h.update(headers or {})
    data = body
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r, raw


def jcall(srv, *a, **k):
    r, raw = call(srv, *a, **k)
    return r.status, (json.loads(raw) if raw and r.getheader("Content-Type", "").startswith("application/json") else raw)


def upload(srv, video: Path, name="talk.mp4"):
    return jcall(srv, "POST", f"/api/upload?name={name}", body=video.read_bytes(),
                 headers={"Content-Type": "application/octet-stream"})


def wait_job(srv, jid, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        st, j = jcall(srv, "GET", f"/api/jobs/{jid}")
        assert st == 200
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.3)
    raise AssertionError("job did not finish")


ANALYZE = {"count": 2, "min_dur": 6, "max_dur": 12, "use_llm": False, "use_faces": False, "long_side": 640}


@pytest.fixture
def ready(srv, video):
    st, d = upload(srv, video)
    assert st == 201
    st, j = jcall(srv, "POST", f"/api/projects/{d['id']}/analyze", ANALYZE)
    assert st == 202
    job = wait_job(srv, j["job"])
    assert job["status"] == "done", job
    return d["id"]


# -- guards -----------------------------------------------------------------

def test_foreign_host_is_refused_everywhere(srv):
    for path in ("/", "/api/projects", "/app.js"):
        r, _ = call(srv, "GET", path, host="evil.example")
        assert r.status == 403
    r, _ = call(srv, "GET", "/api/projects", host=f"127.0.0.1:{srv.port + 1}")
    assert r.status == 403
    r, _ = call(srv, "GET", "/api/projects", host=f"localhost:{srv.port}")
    assert r.status == 200


def test_post_needs_same_origin(srv):
    r, _ = call(srv, "POST", "/api/upload?name=a.mp4", body=b"x", origin="http://evil.example")
    assert r.status == 403
    r, _ = call(srv, "POST", "/api/upload?name=a.mp4", body=b"x", origin=None)
    assert r.status == 403
    assert srv.store.list() == []  # nothing was created


def test_static_files_load_nothing_from_the_network():
    web = Path(server.WEB_DIR)
    for f in ("index.html", "app.js", "style.css"):
        text = (web / f).read_text()
        assert not re.search(r"""(src|href)=["']https?://|url\(\s*["']?https?://|@import""", text), f
        # transcript and titles are untrusted text: no HTML sinks
        assert not re.search(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML|document\.write", text), f


def test_index_and_unknown_paths(srv):
    r, raw = call(srv, "GET", "/")
    assert r.status == 200 and b"LocalClip" in raw
    for p in ("/nope", "/api/nope", "/api/projects/../../etc/passwd", "/api/projects/pzzzzzzzzzz", "/api/jobs/j999"):
        r, _ = call(srv, "GET", p)
        assert r.status == 404, p


# -- upload -----------------------------------------------------------------

def test_upload_rejects_non_video_and_cleans_up(srv, tmp_path):
    st, d = jcall(srv, "POST", "/api/upload?name=notes.txt", body=b"hello",
                  headers={"Content-Type": "application/octet-stream"})
    assert st == 415
    st, d = jcall(srv, "POST", "/api/upload?name=fake.mp4", body=b"not really a video",
                  headers={"Content-Type": "application/octet-stream"})
    assert st == 415 and "readable video" in d["error"]
    assert srv.store.list() == []


def test_upload_rejects_silent_video(srv, tmp_path):
    import subprocess

    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=160x120:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True)
    st, d = upload(srv, silent, "silent.mp4")
    assert st == 415 and "audio" in d["error"]
    assert srv.store.list() == []


def test_upload_filename_cannot_escape(srv, video):
    st, d = upload(srv, video, "..%2F..%2Fevil.mp4")
    assert st == 201
    meta = srv.store.load(d["id"])
    assert meta["name"] == "evil.mp4"
    assert (srv.store.dir(d["id"]) / "source.mp4").exists()


# -- analyze, edit, export --------------------------------------------------

def test_analyze_populates_project(srv, ready):
    st, meta = jcall(srv, "GET", f"/api/projects/{ready}")
    assert st == 200 and meta["status"] == "ready"
    assert meta["selection"]["engine"] == "heuristic"
    assert len(meta["clips"]) == 2 and meta["candidates"]
    assert [c["id"] for c in meta["clips"]] == [1, 2]
    assert meta["transcription"]["engine"] == "synthetic"
    st, tr = jcall(srv, "GET", f"/api/projects/{ready}/transcript")
    assert len(tr["words"]) == 72


def test_analyze_rejects_bad_options(srv, ready):
    for body in ({"count": 0}, {"count": "3"}, {"min_dur": 50, "max_dur": 10}, {"transcribe_engine": "x"},
                 {"whisper_model": "../../etc"}, {"use_llm": "yes"}, {"long_side": 99999}, {"count": 3.5}):
        st, d = jcall(srv, "POST", f"/api/projects/{ready}/analyze", body)
        assert st == 400, (body, d)


def test_video_range_requests(srv, ready):
    r, full = call(srv, "GET", f"/api/projects/{ready}/video")
    assert r.status == 200 and r.getheader("Accept-Ranges") == "bytes"
    size = len(full)
    r, part = call(srv, "GET", f"/api/projects/{ready}/video", headers={"Range": "bytes=100-199"})
    assert r.status == 206 and part == full[100:200]
    assert r.getheader("Content-Range") == f"bytes 100-199/{size}"
    r, part = call(srv, "GET", f"/api/projects/{ready}/video", headers={"Range": "bytes=-50"})
    assert r.status == 206 and part == full[-50:]
    r, part = call(srv, "GET", f"/api/projects/{ready}/video", headers={"Range": f"bytes={size - 10}-"})
    assert r.status == 206 and part == full[-10:]
    r, _ = call(srv, "GET", f"/api/projects/{ready}/video", headers={"Range": f"bytes={size + 5}-"})
    assert r.status == 416
    r, _ = call(srv, "GET", f"/api/projects/{ready}/video", headers={"Range": "bytes=abc"})
    assert r.status == 416


def test_export_trim_edit_and_batch(srv, ready):
    from localclip.render import probe, video_size

    st, meta = jcall(srv, "GET", f"/api/projects/{ready}")
    c1, c2 = meta["clips"]
    clips = [
        {"id": c1["id"], "start": c1["start"], "end": c1["end"], "title": "untouched pick"},
        {"id": c2["id"], "start": c2["start"] + 1.0, "end": c2["start"] + 5.0, "title": "trimmed to 4s",
         "captions": False, "crop": "center", "snap": False},
        {"start": 2.0, "end": 9.0, "title": "my own clip", "crop_x": 0.25},
    ]
    st, j = jcall(srv, "POST", f"/api/projects/{ready}/export", {"clips": clips, "aspect": "9:16", "upper": False})
    assert st == 202
    job = wait_job(srv, j["job"])
    assert job["status"] == "done", job["error"]
    rec = job["result"]
    assert [c["name"] for c in rec["clips"]] == ["clip01.mp4", "clip02.mp4", "clip03.mp4"]
    assert rec["clips"][1]["captions"] is False and rec["clips"][2]["crop"] == "manual"
    assert rec["clips"][1]["duration"] == pytest.approx(4.0, abs=0.01)
    base = f"/api/projects/{ready}/exports/{rec['id']}/"
    for c in rec["clips"]:
        r, raw = call(srv, "GET", base + c["name"])
        assert r.status == 200 and raw[4:8] == b"ftyp"
    out = srv.store.dir(ready) / "exports" / rec["id"]
    assert video_size(probe(str(out / "clip01.mp4"))) == (360, 640)
    r, raw = call(srv, "GET", base + "sheet.png")
    assert r.status == 200 and raw[:4] == b"\x89PNG"
    manifest = json.loads((out / "manifest.json").read_text())
    assert [c["edited"] for c in manifest["clips"]] == [False, True, True]
    assert manifest["clips"][2]["selected_by"] == "manual"
    st, meta = jcall(srv, "GET", f"/api/projects/{ready}")
    assert [e["id"] for e in meta["exports"]] == [rec["id"]]
    r, _ = call(srv, "GET", f"/api/projects/{ready}/exports/{rec['id']}/clip01.mp4?dl=1")
    assert "attachment" in r.getheader("Content-Disposition")


def test_export_validation(srv, ready):
    ok = {"start": 1, "end": 8}
    for clips, text in (
        ([], "non-empty"),
        ([{"start": 5, "end": 4}], "at least"),
        ([{"start": 40, "end": 50}], "at least"),  # entirely past the 32 s source
        ([{"start": "1", "end": 8}], "number"),
        ([{"end": 8}], "needs start"),
        ([{**ok, "crop": "zoom"}], "crop"),
        ([{**ok, "crop_x": 3}], "crop_x"),
        ([{**ok, "title": 5}], "title"),
        ("nope", "non-empty"),
    ):
        st, d = jcall(srv, "POST", f"/api/projects/{ready}/export", {"clips": clips})
        assert st == 400 and re.search(text, d["error"]), (clips, d)
    st, d = jcall(srv, "POST", f"/api/projects/{ready}/export", {"clips": [ok], "aspect": "7:3"})
    assert st == 400
    st, d = jcall(srv, "POST", "/api/projects/pzzzzzzzzzz/export", {"clips": [ok]})
    assert st == 404


def test_export_before_analyze_is_refused(srv, video):
    st, d = upload(srv, video)
    st, e = jcall(srv, "POST", f"/api/projects/{d['id']}/export", {"clips": [{"start": 1, "end": 8}]})
    assert st == 409


def test_export_files_cannot_be_guessed(srv, ready):
    for p in ("exports/20250101T000000Z/clip01.mp4", "exports/..%2F..%2Fproject.json/clip01.mp4",
              "exports/20250101T000000Z/project.json"):
        r, _ = call(srv, "GET", f"/api/projects/{ready}/{p}")
        assert r.status == 404, p


def test_clips_from_json_inherits_pick_metadata():
    stored = [{"id": 1, "start": 3.0, "end": 12.0, "why": "heuristic: hook", "score": 61.0,
               "selected_by": "heuristic", "text": "alpha beta", "sentences": [4, 5]}]
    same = server.clips_from_json([{"id": 1, "start": 3.0, "end": 12.0}], stored, 100.0)[0]
    assert (same.edited, same.selected_by, same.sentences, same.score) == (False, "heuristic", (4, 5), 61.0)
    moved = server.clips_from_json([{"id": 1, "start": 3.5, "end": 12.0}], stored, 100.0)[0]
    assert moved.edited is True and moved.why == "heuristic: hook"
    new = server.clips_from_json([{"start": 3.0, "end": 12.0, "selected_by": "candidate"},
                                  {"start": 3.0, "end": 12.0, "selected_by": "cluster-llm"}], stored, 100.0)
    assert [s.selected_by for s in new] == ["candidate", "manual"]  # a client cannot claim the LLM picked it
