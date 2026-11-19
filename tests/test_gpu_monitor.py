"""GPUMonitor must degrade to "no GPU" when nvidia-smi is missing, not raise."""

import subprocess

from localclip import gpu


def test_query_returns_empty_when_nvidia_smi_is_missing(monkeypatch):
    def missing(*_a, **_kw):
        raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", missing)
    assert gpu._query(["--query-gpu=name"]) == []


def test_query_returns_empty_when_nvidia_smi_times_out(monkeypatch):
    def slow(*_a, **_kw):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(gpu.subprocess, "run", slow)
    assert gpu._query(["--query-gpu=name"]) == []


def test_monitor_is_unavailable_and_usable_without_nvidia_smi(monkeypatch):
    def missing(*_a, **_kw):
        raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", missing)
    with gpu.GPUMonitor() as mon:
        assert mon.available is False
    assert mon.summary() == {"available": False, "samples": 0}
