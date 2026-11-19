"""ffmpeg rendering: crop, scale, burn captions, loudness-normalize, end card."""

from __future__ import annotations

import json
import re
import subprocess

from .crop import CropPlan

LOUD_I, LOUD_TP, LOUD_LRA = -14.0, -1.5, 11.0  # common short-form platform target
FPS = 30


def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=index,codec_type,codec_name,pix_fmt,width,height,r_frame_rate",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def video_size(info: dict) -> tuple[int, int]:
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    return int(v["width"]), int(v["height"])


def has_audio(info: dict) -> bool:
    return any(s["codec_type"] == "audio" for s in info["streams"])


def measure_loudness(path: str, start: float, dur: float) -> dict:
    """First loudnorm pass over the clip window."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
         "-vn", "-af", f"loudnorm=I={LOUD_I}:TP={LOUD_TP}:LRA={LOUD_LRA}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, flags=re.S)
    if not m:
        raise RuntimeError("loudnorm measurement missing from ffmpeg output")
    return json.loads(m.group(0))


def _filter_path(p: str) -> str:
    """Escape an unquoted filename through the option and filtergraph parsers."""
    for specials in ("\\':", "\\'[],;"):
        p = "".join("\\" + c if c in specials or c.isspace() else c for c in p)
    return p


def build_filtergraph(
    plan: CropPlan, out_w: int, out_h: int, ass_path: str | None, loud: dict, end_card: bool
) -> str:
    burn = f"ass=filename={_filter_path(ass_path)}," if ass_path else ""
    v = (
        f"[0:v]crop=w={plan.w}:h={plan.h}:x='{plan.x_expr()}':y={plan.y},"
        f"scale={out_w}:{out_h}:flags=bicubic,setsar=1,fps={FPS},"
        f"{burn}format=yuv420p[v0]"
    )
    a = (
        f"[0:a]loudnorm=I={LOUD_I}:TP={LOUD_TP}:LRA={LOUD_LRA}"
        f":measured_I={loud['input_i']}:measured_TP={loud['input_tp']}"
        f":measured_LRA={loud['input_lra']}:measured_thresh={loud['input_thresh']}"
        f":offset={loud['target_offset']}:linear=true,aresample=48000,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo[a0]"
    )
    if not end_card:
        return f"{v};{a};[v0]null[v];[a0]anull[a]"
    card = (
        f"[1:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},"
        f"setsar=1,fps={FPS},format=yuv420p,fade=t=in:st=0:d=0.25[v1]"
    )
    silence = "[2:a]aformat=sample_fmts=fltp:channel_layouts=stereo,aresample=48000[a1]"
    return f"{v};{a};{card};{silence};[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"


ENCODERS = {
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21",
                   "-b:v", "0", "-profile:v", "high"],
    "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high"],
}


def render_clip(
    src: str, start: float, dur: float, plan: CropPlan, out_w: int, out_h: int, ass_path: str | None,
    loud: dict, out_path: str, end_card: str | None = None, end_card_sec: float = 2.5,
    encoders: tuple[str, ...] = ("h264_nvenc", "libx264"),
) -> tuple[str, str]:
    """Render one clip; try encoders in order. Returns (encoder used, ffmpeg stderr tail on fallback)."""
    inputs = ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src]
    if end_card:
        inputs += ["-loop", "1", "-t", f"{end_card_sec:.2f}", "-framerate", str(FPS), "-i", end_card,
                   "-f", "lavfi", "-t", f"{end_card_sec:.2f}", "-i", "anullsrc=r=48000:cl=stereo"]
    graph = build_filtergraph(plan, out_w, out_h, ass_path, loud, bool(end_card))
    fallback_note = ""
    for enc in encoders:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
               "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
               *ENCODERS[enc], "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-movflags", "+faststart", out_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return enc, fallback_note
        fallback_note += f"{enc} failed: {r.stderr.strip()[-300:]} "
    raise RuntimeError("all encoders failed: " + fallback_note)
