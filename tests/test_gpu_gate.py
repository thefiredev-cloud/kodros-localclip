"""The optional GPU-busy gate: empty by default, driven by LOCALCLIP_GPU_BLOCK."""

import subprocess
import sys

from localclip import gpu

MARKER = "localclip_gate_probe_marker"


def test_empty_by_default_and_proc_not_scanned(monkeypatch):
    monkeypatch.delenv("LOCALCLIP_GPU_BLOCK", raising=False)

    def boom(_path):
        raise AssertionError("/proc must not be scanned when the list is empty")

    monkeypatch.setattr(gpu.os, "listdir", boom)
    assert gpu.reserved_names() == []
    assert gpu.gpu_reserved_running() is False


def test_names_are_split_trimmed_and_lowercased(monkeypatch):
    monkeypatch.setenv("LOCALCLIP_GPU_BLOCK", " Blender, ComfyUI ,,")
    assert gpu.reserved_names() == ["blender", "comfyui"]


def test_blocks_only_while_a_listed_process_runs(monkeypatch):
    monkeypatch.setenv("LOCALCLIP_GPU_BLOCK", MARKER)
    assert gpu.gpu_reserved_running() is False
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", MARKER])
    try:
        assert gpu.gpu_reserved_running() is True
    finally:
        proc.kill()
        proc.wait()
    assert gpu.gpu_reserved_running() is False
