"""Smart crop to a vertical (or any) aspect: follow the largest face, else center.

Faces are sampled a few times per second with OpenCV's Haar detector on
downscaled grayscale frames decoded by ffmpeg. The face track is gap-filled,
median-filtered and eased, then turned into a piecewise-linear ffmpeg ``crop``
x-expression so the virtual camera pans smoothly instead of jittering.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

SAMPLE_W = 480


@dataclass
class CropPlan:
    mode: str  # "face", "manual", "center", "center (no faces found)", "full-frame"
    src_w: int
    src_h: int
    w: int
    h: int
    y: int
    keyframes: list[tuple[float, int]] = field(default_factory=list)  # (t, x) in source px
    face_ratio: float | None = None  # share of samples with a face

    def x_expr(self) -> str:
        return crop_x_expr(self.keyframes)


def out_size(aspect: tuple[int, int], long_side: int = 1920) -> tuple[int, int]:
    aw, ah = aspect
    if aw <= ah:
        w, h = long_side * aw / ah, long_side
    else:
        w, h = long_side, long_side * ah / aw
    return int(round(w / 2) * 2), int(round(h / 2) * 2)


def crop_box(src_w: int, src_h: int, aspect: tuple[int, int]) -> tuple[int, int]:
    """Largest box with the target aspect that fits in the source (even dims)."""
    aw, ah = aspect
    if src_w * ah > src_h * aw:  # source is wider than target: crop width
        h = src_h
        w = src_h * aw / ah
    else:
        w = src_w
        h = src_w * ah / aw
    return int(w // 2 * 2), int(h // 2 * 2)


def sample_faces(path: str, start: float, dur: float, src_w: int, src_h: int, fps: float = 2.0):
    """Return [(t, face_center_x_norm | None)] for the clip window."""
    import cv2

    sh = int(round(src_h * SAMPLE_W / src_w / 2) * 2)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
        "-vf", f"fps={fps},scale={SAMPLE_W}:{sh},format=gray",
        "-f", "rawvideo", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frame_bytes = SAMPLE_W * sh
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    min_side = max(24, sh // 12)
    out: list[tuple[float, float | None]] = []
    for i in range(len(raw) // frame_bytes):
        frame = np.frombuffer(raw, np.uint8, frame_bytes, i * frame_bytes).reshape(sh, SAMPLE_W)
        faces = det.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        t = i / fps
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            out.append((t, (x + w / 2) / SAMPLE_W))
        else:
            out.append((t, None))
    return out


def smooth_track(samples: list[tuple[float, float | None]], median: int = 5, ease: float = 0.35) -> list[tuple[float, float]]:
    """Fill gaps by interpolation, median filter, then exponential easing."""
    ts = np.array([t for t, _ in samples], dtype=float)
    vals = np.array([np.nan if v is None else v for _, v in samples], dtype=float)
    ok = ~np.isnan(vals)
    if not ok.any():
        return []
    vals = np.interp(ts, ts[ok], vals[ok])
    if median > 1 and len(vals) >= median:
        pad = median // 2
        padded = np.pad(vals, pad, mode="edge")
        vals = np.array([np.median(padded[i : i + median]) for i in range(len(vals))])
    eased = [vals[0]]
    for v in vals[1:]:
        eased.append(eased[-1] + ease * (v - eased[-1]))
    return list(zip(ts.tolist(), eased))


def keyframes_from_track(
    track: list[tuple[float, float]], src_w: int, crop_w: int, step: float = 1.0, min_move_px: int = 8
) -> list[tuple[float, int]]:
    """Convert a normalized center track into crop-left keyframes about every ``step`` s."""
    max_x = src_w - crop_w
    keys: list[tuple[float, int]] = []
    next_t = 0.0
    for t, cx in track:
        if t + 1e-9 < next_t and t != track[-1][0]:
            continue
        x = int(round(min(max(cx * src_w - crop_w / 2, 0), max_x)))
        if keys and abs(x - keys[-1][1]) < min_move_px:
            x = keys[-1][1]
        keys.append((round(t, 3), x))
        next_t = t + step
    # Drop interior keyframes that sit on a straight hold (same x on both sides).
    pruned = [keys[0]]
    for i in range(1, len(keys) - 1):
        if not (keys[i][1] == pruned[-1][1] == keys[i + 1][1]):
            pruned.append(keys[i])
    if len(keys) > 1:
        pruned.append(keys[-1])
    return pruned


def crop_x_expr(keys: list[tuple[float, int]]) -> str:
    """Piecewise-linear ffmpeg expression in ``t`` through the keyframes."""
    if not keys:
        return "0"
    if len(keys) == 1 or all(x == keys[0][1] for _, x in keys):
        return str(keys[0][1])
    expr = str(keys[-1][1])
    for i in range(len(keys) - 2, -1, -1):
        (t0, x0), (t1, x1) = keys[i], keys[i + 1]
        dt = max(t1 - t0, 1e-3)
        seg = f"{x0}+({x1 - x0})*(t-{t0})/{dt:.3f}" if x1 != x0 else str(x0)
        expr = f"if(lt(t,{t1}),{seg},{expr})"
    return expr


CROP_MODES = ("auto", "face", "center")


def plan_crop(
    path: str, start: float, dur: float, src_w: int, src_h: int, aspect: tuple[int, int],
    use_faces: bool = True, mode: str = "auto", manual_x: float | None = None,
) -> CropPlan:
    """Plan the 9:16 (or any aspect) reframe.

    ``mode`` is ``auto`` (faces when ``use_faces``, else center), ``face`` (always
    track) or ``center``. ``manual_x`` (0..1, the crop's horizontal center in the
    source) overrides everything with a fixed, user-chosen framing.
    """
    if mode not in CROP_MODES:
        raise ValueError(f"crop mode must be one of {CROP_MODES}, got {mode!r}")
    cw, ch = crop_box(src_w, src_h, aspect)
    y = (src_h - ch) // 2
    center_x = (src_w - cw) // 2
    if cw >= src_w - 1:  # source is already as narrow as the target: nothing to track sideways
        pmode = "full-frame" if ch >= src_h - 1 else "center"
        return CropPlan(pmode, src_w, src_h, cw, ch, y, [(0.0, center_x)])
    if manual_x is not None:
        mx = min(max(float(manual_x), 0.0), 1.0)
        x = int(round(min(max(mx * src_w - cw / 2, 0), src_w - cw)))
        return CropPlan("manual", src_w, src_h, cw, ch, y, [(0.0, x)])
    if mode == "center" or (mode == "auto" and not use_faces):
        return CropPlan("center", src_w, src_h, cw, ch, y, [(0.0, center_x)])
    samples = sample_faces(path, start, dur, src_w, src_h)
    hits = sum(1 for _, v in samples if v is not None)
    ratio = hits / len(samples) if samples else 0.0
    if ratio < 0.2:
        return CropPlan("center (no faces found)", src_w, src_h, cw, ch, y, [(0.0, center_x)], round(ratio, 3))
    keys = keyframes_from_track(smooth_track(samples), src_w, cw)
    return CropPlan("face", src_w, src_h, cw, ch, y, keys, round(ratio, 3))
