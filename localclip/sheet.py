"""Contact sheet: one row of evenly spaced frames per exported clip."""

from __future__ import annotations

import subprocess

import numpy as np

from .render import probe


def _frame(path: str, t: float, height: int):
    import cv2

    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
         "-vf", f"scale=-2:{height}", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True,
    ).stdout
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def contact_sheet(paths: list[str], out_png: str, labels: list[str] | None = None, frames: int = 6,
                  thumb_h: int = 320) -> str:
    """Write a PNG with ``frames`` thumbnails per clip, one clip per row."""
    import cv2

    rows = []
    for i, p in enumerate(paths):
        dur = float(probe(p)["format"]["duration"])
        ts = [dur * (k + 0.5) / frames for k in range(frames)]
        thumbs = [f for f in (_frame(p, t, thumb_h) for t in ts) if f is not None]
        if not thumbs:
            continue
        row = np.hstack(thumbs)
        label = (labels[i] if labels else p.rsplit("/", 1)[-1]) + f"  {dur:.1f}s"
        cv2.rectangle(row, (0, 0), (min(row.shape[1], 14 + 11 * len(label)), 28), (0, 0, 0), -1)
        cv2.putText(row, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    if not rows:
        raise RuntimeError("no frames could be decoded for the contact sheet")
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0))) for r in rows]
    if not cv2.imwrite(out_png, np.vstack(rows)):
        raise RuntimeError(f"could not write {out_png}")
    return out_png
