"""localclip CLI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .pipeline import Options, batch_make, make


def _aspect(s: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in s.split(":"))
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError("aspect must look like 9:16") from None


def _pick_args(m: argparse.ArgumentParser) -> None:
    m.add_argument("--count", type=int, default=3)
    m.add_argument("--min", dest="min_dur", type=float, default=20.0, help="min clip seconds")
    m.add_argument("--max", dest="max_dur", type=float, default=60.0, help="max clip seconds")
    m.add_argument("--no-llm", action="store_true", help="heuristic selection only")
    m.add_argument("--llm-timeout", type=float, default=240.0)
    m.add_argument("--transcriber", choices=["auto", "parakeet", "whisper"], default="auto")
    m.add_argument("--whisper-model", default="small")


def _render_args(m: argparse.ArgumentParser) -> None:
    m.add_argument("--aspect", type=_aspect, default=(9, 16))
    m.add_argument("--no-face", action="store_true", help="center crop, skip face tracking")
    m.add_argument("--caption-case", choices=["upper", "asis"], default="upper")
    m.add_argument("--end-card", help="image appended after each clip")
    m.add_argument("--end-card-sec", type=float, default=2.5)
    m.add_argument("--long-side", type=int, default=1920, help="output pixels on the long edge (1920 = 1080x1920)")


def _opts(a: argparse.Namespace) -> Options:
    return Options(
        count=getattr(a, "count", 3), aspect=a.aspect, min_dur=getattr(a, "min_dur", 20.0),
        max_dur=getattr(a, "max_dur", 60.0), out_dir=getattr(a, "out_dir", None),
        end_card=getattr(a, "end_card", None), end_card_sec=getattr(a, "end_card_sec", 2.5),
        use_llm=not getattr(a, "no_llm", False), llm_timeout=getattr(a, "llm_timeout", 240.0),
        use_faces=not a.no_face, transcribe_engine=getattr(a, "transcriber", "auto"),
        whisper_model=getattr(a, "whisper_model", "small"), upper=a.caption_case == "upper",
        long_side=getattr(a, "long_side", 1920),
    )


def _parse_trim(items: list[str] | None) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for it in items or []:
        try:
            n, rng = it.split("=", 1)
            s, e = rng.split(":", 1)
            out[int(n)] = (float(s), float(e))
        except ValueError:
            raise SystemExit(f"--trim wants N=START:END in seconds, got {it!r}") from None
    return out


def _parse_crop(items: list[str] | None) -> dict[int, tuple[str, float | None]]:
    out: dict[int, tuple[str, float | None]] = {}
    for it in items or []:
        try:
            n, v = it.split("=", 1)
            if v.startswith("x"):
                out[int(n)] = ("auto", float(v[1:]))
            elif v in ("auto", "face", "center"):
                out[int(n)] = (v, None)
            else:
                raise ValueError
        except ValueError:
            raise SystemExit(f"--crop wants N=auto|face|center|x0.35 (manual center 0..1), got {it!r}") from None
    return out


def _ints(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",")] if s else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localclip", description="Local auto short clips from a long video.")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="cut a video into captioned vertical clips, one shot")
    m.add_argument("input")
    _pick_args(m)
    _render_args(m)
    m.add_argument("--out", dest="out_dir", help="output dir (default ~/.local/share/localclip/runs/...)")
    m.add_argument("--json", action="store_true", help="print the manifest to stdout")

    b = sub.add_parser("batch", help="run make over many files or folders; one failure does not stop the rest")
    b.add_argument("inputs", nargs="+")
    _pick_args(b)
    _render_args(b)
    b.add_argument("--out", dest="out_dir", help="parent dir; each input gets its own subfolder")
    b.add_argument("--json", action="store_true")

    a = sub.add_parser("analyze", help="transcribe and auto-pick clips into an editable project (no export)")
    a.add_argument("input")
    _pick_args(a)
    a.add_argument("--no-face", action="store_true")

    e = sub.add_parser("export", help="export clips of a project, optionally with trims and per-clip overrides")
    e.add_argument("project")
    e.add_argument("--clips", help="comma list of clip numbers (default all)")
    e.add_argument("--trim", action="append", metavar="N=START:END", help="override clip N bounds in source seconds")
    e.add_argument("--crop", action="append", metavar="N=MODE", help="N=auto|face|center|x0.35 (manual center 0..1)")
    e.add_argument("--no-captions", help="comma list of clip numbers to export without captions")
    e.add_argument("--snap", action="store_true", help="nudge trimmed cuts off the middle of a word")
    _render_args(e)

    sub.add_parser("projects", help="list editor projects")

    s = sub.add_parser("serve", help="local web editor (upload, pick, trim, export)")
    s.add_argument("--port", type=int, default=8871)
    return p


def _print_clips(meta: dict) -> None:
    for c in meta["clips"]:
        print(f"  {c['id']}: {c['start']:.1f}-{c['end']:.1f}s ({c['end'] - c['start']:.1f}s) "
              f"[{c['selected_by']}] {c['title']}")


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)

    if a.cmd == "serve":
        from .server import serve
        serve(a.port)
        return 0
    if a.cmd == "projects":
        from .project import Store
        for pr in Store().list():
            print(f"{pr['id']}  {pr['status']:9} {pr['clips']} clips {pr['exports']} exports  {pr['name']}")
        return 0

    if hasattr(a, "count") and a.count < 1:
        p.error("--count must be at least 1")
    if hasattr(a, "min_dur") and not 0 < a.min_dur < a.max_dur:
        p.error("need 0 < --min < --max")
    if a.cmd == "make":
        log = (lambda s: print(s, file=sys.stderr)) if a.json else print
        manifest = make(a.input, _opts(a), log=log)
        if a.json:
            print(json.dumps(manifest, indent=2))
        return 0
    if a.cmd == "batch":
        log = (lambda s: print(s, file=sys.stderr)) if a.json else print
        results = batch_make(a.inputs, _opts(a), log=log)
        if a.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(("ok    " if r["ok"] else "FAIL  ") + r["input"] + ("" if r["ok"] else f"  {r['error']}"))
                for c in r.get("clips", []):
                    print("      " + c)
        return 0 if all(r["ok"] for r in results) else 1

    from .project import Store, analyze_project, export_project, import_path
    store = Store()
    if a.cmd == "analyze":
        a.aspect, a.caption_case = (9, 16), "upper"
        pid = import_path(store, a.input)
        meta = analyze_project(store, pid, _opts(a), log=lambda s: print(s, file=sys.stderr))
        print(f"project {pid}: {meta['selection']['engine']} ({meta['selection']['note']})")
        _print_clips(meta)
        print(f"edit in the browser: localclip serve   |   or: localclip export {pid} --trim 1=12.5:40")
        return 0

    meta = store.load(a.project)
    trims, crops = _parse_trim(a.trim), _parse_crop(a.crop)
    nocap = set(_ints(a.no_captions) or [])
    want = _ints(a.clips) or [c["id"] for c in meta["clips"]]
    known = {c["id"]: c for c in meta["clips"]}
    for n in [*want, *trims, *crops, *nocap]:
        if n not in known:
            p.error(f"project has no clip {n}; it has {sorted(known)}")
    clips = []
    for n in want:
        c = dict(known[n])
        if n in trims:
            c["start"], c["end"] = trims[n]
            c["edited"] = True
            c["snap"] = a.snap
        if n in crops:
            c["crop"], c["crop_x"] = crops[n]
        c["captions"] = n not in nocap
        clips.append(c)
    opts = Options(**{**(meta.get("options") or asdict(Options())), "aspect": a.aspect, "use_faces": not a.no_face,
                      "upper": a.caption_case == "upper", "end_card": a.end_card, "end_card_sec": a.end_card_sec,
                      "out_dir": None})
    opts.aspect = tuple(opts.aspect)
    rec = export_project(store, a.project, clips, opts, log=lambda s: print(s, file=sys.stderr))
    out = store.dir(a.project) / "exports" / rec["id"]
    print(f"export {rec['id']} ({rec['seconds']}s) -> {out}")
    for c in rec["clips"]:
        print(f"  {out / c['name']}  {c['start']:.1f}-{c['end']:.1f}s  crop={c['crop']} captions={c['captions']}")
    print(f"  contact sheet: {out / 'sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
