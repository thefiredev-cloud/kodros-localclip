"""Word-by-word burned captions as ASS subtitles.

Words are grouped into short on-screen chunks (up to three words). Each word in
a chunk gets its own event that shows the whole chunk with the active word
highlighted, so the highlight follows speech word by word. Events never
overlap and never leave the clip window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .words import Word

BREAK_PUNCT = (".", "?", "!", ",", ";", ":")


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Inter"
    size_ratio: float = 0.048  # font size as a fraction of video height
    primary: str = "&H00FFFFFF"  # white  (ASS colours are &HAABBGGRR)
    highlight: str = "&H0000E6FF"  # warm yellow
    outline: str = "&H00000000"
    outline_px: float = 6
    shadow_px: float = 2
    margin_ratio: float = 0.27  # distance from bottom as a fraction of height
    pop: int = 112  # active word scale percent
    upper: bool = True


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    text: str  # ASS-formatted line


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean_word(text: str, upper: bool) -> str:
    t = text.strip()
    t = t.replace("\\", "/").replace("{", "(").replace("}", ")")
    t = re.sub(r"[,;:.]+$", "", t)  # trailing soft punctuation reads as noise on screen
    return t.upper() if upper else t


def relative_words(words: list[Word], clip_start: float, clip_end: float) -> list[Word]:
    """Words overlapping [clip_start, clip_end], shifted to clip time and clamped."""
    out: list[Word] = []
    dur = clip_end - clip_start
    for w in words:
        if w.end <= clip_start or w.start >= clip_end:
            continue
        s = min(max(w.start - clip_start, 0.0), dur)
        e = min(max(w.end - clip_start, s), dur)
        out.append(Word(s, e, w.text))
    return out


def chunk_words(
    words: list[Word], max_words: int = 3, max_chars: int = 20, max_gap: float = 0.45
) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        if cur:
            joined = len(" ".join(x.text.strip() for x in cur + [w]))
            gap = w.start - cur[-1].end
            if len(cur) >= max_words or joined > max_chars or gap > max_gap:
                chunks.append(cur)
                cur = []
        cur.append(w)
        if w.text.rstrip().endswith(BREAK_PUNCT):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def caption_events(
    words: list[Word],
    clip_dur: float,
    style: CaptionStyle = CaptionStyle(),
    hold: float = 0.3,
    bridge: float = 0.25,
) -> list[Event]:
    """``words`` must already be clip-relative (see ``relative_words``)."""
    chunks = [c for c in chunk_words([w for w in words if w.text.strip()]) if c]
    events: list[Event] = []
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else clip_dur
        chunk_end = min(chunk[-1].end + hold, next_start)
        if next_start - chunk_end < bridge:  # avoid a one-frame blink between chunks
            chunk_end = next_start
        chunk_end = min(chunk_end, clip_dur)
        labels = [clean_word(w.text, style.upper) for w in chunk]
        for k, w in enumerate(chunk):
            start = chunk[0].start if k == 0 else w.start
            end = chunk[k + 1].start if k + 1 < len(chunk) else chunk_end
            if end - start < 0.01:
                continue
            parts = []
            for j, label in enumerate(labels):
                if j == k:
                    parts.append(
                        f"{{\\c{style.highlight}\\fscx{style.pop}\\fscy{style.pop}}}{label}{{\\r}}"
                    )
                else:
                    parts.append(label)
            events.append(Event(round(start, 3), round(end, 3), " ".join(parts)))
    return events


def build_ass(events: list[Event], width: int, height: int, style: CaptionStyle = CaptionStyle()) -> str:
    size = round(height * style.size_ratio)
    margin_v = round(height * style.margin_ratio)
    margin_h = round(width * 0.06)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{style.font},{size},{style.primary},{style.primary},{style.outline},&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline_px},{style.shadow_px},2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [
        f"Dialogue: 0,{ass_time(e.start)},{ass_time(e.end)},Word,,0,0,0,,{e.text}" for e in events
    ]
    return header + "\n".join(lines) + "\n"
