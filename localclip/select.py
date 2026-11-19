"""Pick self-contained short-clip moments from a transcript.

Candidates are windows that start and end on sentence boundaries and last
``min_dur``..``max_dur`` seconds. A deterministic heuristic scores every
candidate. The cluster LLM then picks the final clips from the top candidates;
if it fails or returns too few valid picks, the heuristic fills the gap. Every
clip records which selector chose it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .words import Sentence, Word

DANGLING_START = {"and", "but", "so", "because", "or", "which", "that", "then", "also", "plus"}
FILLER = {"um", "uh", "erm", "hmm", "like", "you know"}
HOOK_WORDS = {
    "how", "why", "what", "secret", "mistake", "never", "always", "best", "worst",
    "truth", "actually", "biggest", "first", "only", "nobody", "everyone", "stop",
    "need", "must", "real", "problem", "lesson", "learned", "story",
}


@dataclass
class Candidate:
    first_sentence: int
    last_sentence: int
    start: float
    end: float
    text: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Pick:
    start: float
    end: float
    text: str
    title: str
    reason: str
    score: float
    selector: str  # "cluster-llm" or "heuristic"
    first_sentence: int
    last_sentence: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def build_candidates(
    sentences: list[Sentence], min_dur: float = 20.0, max_dur: float = 60.0
) -> list[Candidate]:
    out: list[Candidate] = []
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            dur = sentences[j].end - sentences[i].start
            if dur > max_dur:
                break
            if dur >= min_dur:
                out.append(
                    Candidate(
                        first_sentence=i,
                        last_sentence=j,
                        start=sentences[i].start,
                        end=sentences[j].end,
                        text=" ".join(s.text for s in sentences[i : j + 1]),
                    )
                )
    return out


def score_candidate(c: Candidate, sentences: list[Sentence], words: list[Word]) -> Candidate:
    """Score 0..100 from structure, pacing and hook signals. Fills ``reasons``."""
    reasons: list[str] = []
    score = 50.0
    first = sentences[c.first_sentence]
    last = sentences[c.last_sentence]
    first_tok = _tokens(first.text)

    if first_tok and first_tok[0] in DANGLING_START:
        score -= 15
        reasons.append(f"starts mid-thought ('{first_tok[0]}')")
    else:
        score += 5
        reasons.append("clean opening")
    if last.complete:
        score += 10
        reasons.append("ends on a full sentence")
    else:
        score -= 10
        reasons.append("ending trails off")
    if first.text.rstrip().endswith("?"):
        score += 10
        reasons.append("opens with a question")
    hook_hits = sorted(set(first_tok) & HOOK_WORDS)
    if hook_hits:
        score += min(12, 4 * len(hook_hits))
        reasons.append("hook words: " + ", ".join(hook_hits))
    if re.search(r"\d", first.text):
        score += 5
        reasons.append("concrete number up front")

    span = words[first.first_word : last.last_word + 1]
    n = len(span)
    wps = n / c.duration if c.duration > 0 else 0
    if 2.0 <= wps <= 3.6:
        score += 8
        reasons.append(f"lively pace {wps:.1f} w/s")
    elif wps < 1.3:
        score -= 12
        reasons.append(f"slow pace {wps:.1f} w/s")
    speech = sum(w.end - w.start for w in span)
    silence = 1 - (speech / c.duration) if c.duration > 0 else 1
    if silence > 0.35:
        score -= 10
        reasons.append(f"{silence:.0%} dead air")
    toks = _tokens(c.text)
    fill = sum(1 for t in toks if t in FILLER)
    if toks and fill / len(toks) > 0.05:
        score -= 8
        reasons.append("filler-heavy")
    # Mild preference for 30-45 s, the sweet spot for Shorts/Reels.
    score -= abs(c.duration - 38) * 0.25
    c.score = round(max(0.0, min(100.0, score)), 1)
    c.reasons = reasons
    return c


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def greedy_pick(cands: list[Candidate], count: int, max_overlap: float = 0.0) -> list[Candidate]:
    """Highest score first; reject candidates overlapping a chosen one by more
    than ``max_overlap`` of the shorter duration."""
    chosen: list[Candidate] = []
    for c in sorted(cands, key=lambda x: (-x.score, x.start)):
        if all(
            overlap(c.start, c.end, k.start, k.end) <= max_overlap * min(c.duration, k.duration)
            for k in chosen
        ):
            chosen.append(c)
            if len(chosen) == count:
                break
    return chosen


def _title_from(text: str, n: int = 8) -> str:
    words = text.split()
    return " ".join(words[:n]) + ("..." if len(words) > n else "")


def _heuristic_pick(c: Candidate) -> Pick:
    return Pick(
        start=c.start,
        end=c.end,
        text=c.text,
        title=_title_from(c.text),
        reason="heuristic: " + "; ".join(c.reasons),
        score=c.score,
        selector="heuristic",
        first_sentence=c.first_sentence,
        last_sentence=c.last_sentence,
    )


def heuristic_picks(cands: list[Candidate], count: int) -> list[Pick]:
    return [_heuristic_pick(c) for c in greedy_pick(cands, count)]


LLMFn = Callable[[str, str], str]  # (system, user) -> assistant text

SYSTEM_PROMPT = (
    "You pick the best moments from a video transcript for vertical short clips "
    "(TikTok, Reels, Shorts). A great clip is self-contained: it makes sense with no "
    "prior context, opens with a hook, delivers one idea or story beat, and ends cleanly. "
    "Reply with JSON only."
)


def build_llm_prompt(shortlist: list[Candidate], count: int) -> str:
    lines = [
        f"Pick the {count} best clips from these numbered candidates. "
        "Clips must not overlap in time.",
        'Reply with only a JSON array like [{"id": 3, "title": "short punchy title", '
        '"reason": "one sentence on why it works", "score": 0-100}].',
        "",
    ]
    for k, c in enumerate(shortlist):
        text = c.text if len(c.text) <= 900 else c.text[:900] + "..."
        lines.append(f"[{k}] {c.start:.1f}-{c.end:.1f}s ({c.duration:.0f}s): {text}")
    return "\n".join(lines)


def _extract_json_array(reply: str):
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    m = re.search(r"\[.*\]", reply, flags=re.S)
    if not m:
        raise ValueError("no JSON array in LLM reply")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("LLM reply is not a list")
    return data


def parse_llm_picks(reply: str, shortlist: list[Candidate], count: int) -> list[Pick]:
    """Validate the LLM reply: known ids only, no overlaps, at most ``count``."""
    picks: list[Pick] = []
    for item in _extract_json_array(reply):
        if not isinstance(item, dict):
            continue
        try:
            k = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if not 0 <= k < len(shortlist):
            continue
        c = shortlist[k]
        if any(overlap(c.start, c.end, p.start, p.end) > 0 for p in picks):
            continue
        try:
            score = float(item.get("score", c.score))
        except (TypeError, ValueError):
            score = c.score
        picks.append(
            Pick(
                start=c.start,
                end=c.end,
                text=c.text,
                title=str(item.get("title") or _title_from(c.text))[:120],
                reason="cluster-llm: " + str(item.get("reason") or "no reason given")[:400],
                score=round(score, 1),
                selector="cluster-llm",
                first_sentence=c.first_sentence,
                last_sentence=c.last_sentence,
            )
        )
        if len(picks) == count:
            break
    return picks


@dataclass
class Selection:
    picks: list[Pick]
    engine: str  # "cluster-llm", "heuristic", or "cluster-llm+heuristic"
    note: str
    shortlist: list[Candidate] = field(default_factory=list)  # diverse candidates the LLM saw


def select_clips(
    words: list[Word],
    sentences: list[Sentence],
    count: int,
    llm: LLMFn | None = None,
    min_dur: float = 20.0,
    max_dur: float = 60.0,
    shortlist_size: int = 12,
) -> Selection:
    cands = [score_candidate(c, sentences, words) for c in build_candidates(sentences, min_dur, max_dur)]
    if not cands:
        raise ValueError(
            f"no {min_dur:.0f}-{max_dur:.0f}s window found; the transcript is too short or has no speech"
        )
    # Diverse shortlist: allow up to 50% overlap so the LLM sees alternatives.
    shortlist = sorted(greedy_pick(cands, shortlist_size, max_overlap=0.5), key=lambda c: c.start)
    note = ""
    picks: list[Pick] = []
    if llm is not None:
        try:
            picks = parse_llm_picks(llm(SYSTEM_PROMPT, build_llm_prompt(shortlist, count)), shortlist, count)
            note = f"cluster LLM returned {len(picks)} valid pick(s)"
        except Exception as e:  # network, timeout, bad JSON: fall back, but say so.
            note = f"cluster LLM failed ({type(e).__name__}: {str(e)[:160]}); heuristic used"
            picks = []
    else:
        note = "cluster LLM disabled; heuristic used"
    llm_count = len(picks)
    for c in sorted(cands, key=lambda x: (-x.score, x.start)):
        if len(picks) >= count:
            break
        if all(overlap(c.start, c.end, p.start, p.end) == 0 for p in picks):
            picks.append(_heuristic_pick(c))
    engine = (
        "cluster-llm" if llm_count and llm_count == len(picks)
        else "cluster-llm+heuristic" if llm_count
        else "heuristic"
    )
    picks.sort(key=lambda p: p.start)
    return Selection(picks=picks, engine=engine, note=note, shortlist=shortlist)
