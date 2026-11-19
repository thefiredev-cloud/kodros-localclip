"""Minimal OpenAI-compatible chat client for a local model server (llama.cpp, vLLM, Ollama and similar)."""

from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_URL = os.environ.get("LOCALCLIP_LLM_URL", "http://127.0.0.1:8080/v1")
DEFAULT_MODEL = os.environ.get("LOCALCLIP_LLM_MODEL", "local-model")


def make_cluster_llm(url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, timeout: float = 240.0):
    """Return ``fn(system, user) -> text``. Raises on HTTP errors, timeouts or empty replies."""

    def call(system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                # Reasoning model: leave room for thinking plus the JSON answer.
                "max_tokens": 6000,
            }
        ).encode()
        req = urllib.request.Request(
            url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        if not text.strip():
            raise ValueError("cluster LLM returned an empty answer")
        return text

    return call
