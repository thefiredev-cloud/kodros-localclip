# Build guide: kodros-localclip

## Requirements

- Linux, Python 3.11 or newer, `ffmpeg` with libass on `PATH`.
- Optional NVIDIA GPU. The `gpu` extra pulls the cuBLAS 12 and cuDNN 9 wheels that ctranslate2 needs. LocalClip preloads them, so no `LD_LIBRARY_PATH` is needed.
- Optional: an OpenAI-compatible model server for moment picking.

## 1. Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[gpu,test]'     # use '.[test]' without CUDA
```

OpenCV stays below version 5 because OpenCV 5 removed `CascadeClassifier` from the main module.

## 2. Test

```bash
python -m pytest -q
```

The render-path tests build one-second black video and a generated sine tone, decode the resulting H.264/AAC file and require visible caption pixels and non-silent audio. This includes file names with spaces, apostrophes, colons and backslashes. No models or services run.

## 3. Make a test video (optional)

`scripts/make_synthetic_talk.py` builds a synthetic talking video: Kokoro text-to-speech narration of `fixtures/synthetic-talk.txt` over a public-domain NASA portrait (`fixtures/astronaut-public-domain.png`, from scikit-image sample data) drifting across a 16:9 frame. It needs `kokoro_onnx`, `onnxruntime` and the Kokoro model files in `KOKORO_DIR` (default `~/.local/share/kokoro`).

```bash
python scripts/make_synthetic_talk.py fixtures/synthetic-talk.txt fixtures/astronaut-public-domain.png talk.mp4
```

## 4. Run

```bash
localclip make INPUT.mp4 --count 3 --aspect 9:16 [--end-card card.png] [--out DIR]
localclip batch a.mp4 b.mp4 folder/ --out DIR     # one failure does not stop the rest
localclip analyze INPUT.mp4                       # editable project, no export yet
localclip export PROJECT_ID --trim 2=12.5:40 --crop 1=x0.3 --no-captions 3
localclip serve                                   # web editor, port 8871
```

Useful flags: `--no-llm` (heuristics only), `--no-face` (center crop), `--transcriber auto|parakeet|whisper`, `--whisper-model tiny|base|small|medium|large-v3`, `--caption-case upper|asis`, `--min` and `--max` clip seconds.

Default output: `~/.local/share/localclip/runs/<input>-<utc>/` (mode 0700). Data root: `LOCALCLIP_HOME`.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `LOCALCLIP_LLM_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible base URL |
| `LOCALCLIP_LLM_MODEL` | `local-model` | Model id sent in the request |
| `LOCALCLIP_PARAKEET_URL` | `http://127.0.0.1:8798/v1/audio/transcriptions` | Optional local speech service, probed first |
| `LOCALCLIP_PARAKEET_TOKEN_FILE` | `~/.config/localclip/stt-token` | Bearer token for that service |
| `LOCALCLIP_HOME` | `~/.local/share/localclip` | Projects, cache, runs |
| `LOCALCLIP_MAX_UPLOAD_GB` | `20` | Web upload cap |
| `LOCALCLIP_GPU_BLOCK` | empty | Comma-separated process-name fragments. While one is running, LocalClip uses CPU Whisper and libx264. Empty blocks nothing. Logs say "a GPU-reserved process is running" |

## Pipeline

1. Transcribe with word timestamps. The optional local speech service is probed first and used only if it returns per-word times. Otherwise faster-whisper runs on CUDA (float16), then CPU (int8) if CUDA fails. Transcripts are cached.
2. Pick moments: heuristic scores (hook words, question opener, clean start, full-sentence end, pace, dead air, filler), then the model chooses among the top windows.
3. Smart crop: OpenCV Haar face detection at 2 fps, gap fill, median filter, easing, then a piecewise-linear ffmpeg `crop`. It falls back to a center crop when faces appear in under 20% of samples.
4. Captions: up to three words on screen with the spoken word highlighted, rendered as ASS and burned with libass.
5. Audio: two-pass `loudnorm` to -14 LUFS, -1.5 dBTP.
6. Encode with `h264_nvenc`, falling back to `libx264`. An optional end card image is appended.

Sources the browser cannot play (not H.264/AAC) get a 540p proxy for preview only. Exports always come from the original.

## Layout

```text
localclip/     pipeline, transcribe, select, crop, captions, render, server, web/
tests/         57 tests and media helpers
scripts/       synthetic test video builder
fixtures/      public-domain portrait and narration text
docs/architecture.svg
```
