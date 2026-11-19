"""GPU usage sampling via nvidia-smi, plus the optional GPU-busy gate."""

from __future__ import annotations

import os
import subprocess
import threading
import time


def reserved_names() -> list[str]:
    """Process-name fragments from LOCALCLIP_GPU_BLOCK (comma separated, case-insensitive). Empty by default."""
    raw = os.environ.get("LOCALCLIP_GPU_BLOCK", "")
    return [n.strip().lower() for n in raw.split(",") if n.strip()]


def gpu_reserved_running() -> bool:
    """True when a process whose command line contains a listed fragment is running.

    With LOCALCLIP_GPU_BLOCK unset or empty this is always False and /proc is not scanned.
    """
    names = reserved_names()
    if not names:
        return False
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode(errors="ignore").lower()
        except OSError:
            continue
        if any(n in cmd for n in names):
            return True
    return False


def _query(args: list[str]) -> list[list[str]]:
    try:
        r = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []  # no nvidia-smi on PATH, or it hung: report "no GPU" instead of crashing
    if r.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")] for line in r.stdout.strip().splitlines() if line.strip()]


def _descendants(root: int) -> set[int]:
    kids: dict[int, list[int]] = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(pid))
    out, stack = {root}, [root]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


class GPUMonitor:
    """Samples total GPU memory/utilization and this process tree's GPU memory once a second."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.available = bool(_query(["--query-gpu=name"]))

    def _run(self) -> None:
        me = os.getpid()
        while not self._stop.is_set():
            try:
                g = _query(["--query-gpu=memory.used,memory.total,utilization.gpu,utilization.encoder"])
                apps = _query(["--query-compute-apps=pid,used_memory"])
                tree = _descendants(me)
                own = sum(int(m) for p, m in apps if p.isdigit() and int(p) in tree and m.isdigit())
                if g:
                    used, total, util, enc = (int(x) if x.isdigit() else 0 for x in g[0])
                    self.samples.append({"t": time.time(), "used_mib": used, "total_mib": total,
                                         "util_pct": util, "enc_pct": enc, "own_mib": own})
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.available:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.available:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": self.available, "samples": 0}
        s = self.samples
        return {
            "available": True,
            "samples": len(s),
            "total_mib": s[0]["total_mib"],
            "baseline_used_mib": s[0]["used_mib"],
            "peak_used_mib": max(x["used_mib"] for x in s),
            "peak_localclip_mib": max(x["own_mib"] for x in s),
            "peak_util_pct": max(x["util_pct"] for x in s),
            "mean_util_pct": round(sum(x["util_pct"] for x in s) / len(s), 1),
            "peak_encoder_pct": max(x["enc_pct"] for x in s),
            "note": "util is whole-GPU (shared with other processes); localclip_mib is this process tree only",
        }
