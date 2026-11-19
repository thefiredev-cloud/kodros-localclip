from __future__ import annotations

import json

import pytest

from localclip.select import (
    build_candidates,
    greedy_pick,
    parse_llm_picks,
    score_candidate,
    select_clips,
)
from localclip.words import split_sentences

STRONG = (
    "Why do most people pay for software they could run at home? I asked that question last month. "
    "The answer surprised me because seven of nine subscriptions did work my own computer could do. "
    "The graphics card in my desk sits idle most of the day. That is wasted money and wasted hardware. "
    "Own the three pieces underneath and you pay only for electricity."
)
RAMBLE = (
    "and um so then uh I kept going and like there was a lot of stuff and some of it was fine I guess "
    "and some of it was you know not really what I wanted so I kept going and then I kept going some more "
    "and uh yeah it was like that for a while and um anyway"
)
SECOND = (
    "Here is the biggest mistake people make with AI tools. They treat every tool as a separate monthly bill. "
    "Most of these tools are the same three pieces underneath. Speech to text. A language model. A video encoder. "
    "Once you own those pieces you can build almost any of these products yourself."
)


@pytest.fixture
def transcript(make_words):
    words = make_words([STRONG, RAMBLE, SECOND, RAMBLE, STRONG])
    return words, split_sentences(words)


def test_split_sentences_on_punctuation_and_pauses(make_words):
    words = make_words(["One two three. Four five six?", "Seven eight"])
    sents = split_sentences(words)
    assert [s.text for s in sents] == ["One two three.", "Four five six?", "Seven eight"]
    assert sents[0].complete and not sents[2].complete
    assert sents[1].first_word == 3 and sents[1].last_word == 5


def test_split_sentences_caps_unpunctuated_runs(make_words):
    words = make_words([" ".join(["word"] * 100)], wps=3.0)  # 33 s, no punctuation
    sents = split_sentences(words, max_len=15.0)
    assert len(sents) >= 3
    assert all(s.duration <= 15.0 for s in sents)


def test_candidates_respect_duration_and_sentence_boundaries(transcript):
    words, sents = transcript
    cands = build_candidates(sents, 20, 60)
    assert cands
    starts = {s.start for s in sents}
    ends = {s.end for s in sents}
    for c in cands:
        assert 20 <= c.duration <= 60
        assert c.start in starts and c.end in ends


def test_heuristic_prefers_hooked_complete_over_ramble(transcript):
    words, sents = transcript
    cands = [score_candidate(c, sents, words) for c in build_candidates(sents, 20, 60)]
    best = max(cands, key=lambda c: c.score)
    assert not best.text.lower().startswith("and")
    ramble = [c for c in cands if c.text.lower().startswith("and um")]
    assert ramble and best.score > max(c.score for c in ramble)
    assert any("question" in r or "hook" in r for r in best.reasons)


def test_greedy_pick_no_overlap(transcript):
    words, sents = transcript
    cands = [score_candidate(c, sents, words) for c in build_candidates(sents, 20, 60)]
    picks = greedy_pick(cands, 3)
    for i, a in enumerate(picks):
        for b in picks[i + 1 :]:
            assert a.end <= b.start or b.end <= a.start


def test_select_without_llm_is_labeled_heuristic(transcript):
    words, sents = transcript
    sel = select_clips(words, sents, 3, llm=None)
    assert sel.engine == "heuristic"
    assert len(sel.picks) == 3
    assert all(p.selector == "heuristic" for p in sel.picks)
    assert [p.start for p in sel.picks] == sorted(p.start for p in sel.picks)


def test_select_llm_timeout_falls_back(transcript):
    words, sents = transcript

    def boom(system, user):
        raise TimeoutError("timed out")

    sel = select_clips(words, sents, 3, llm=boom)
    assert sel.engine == "heuristic"
    assert "TimeoutError" in sel.note
    assert len(sel.picks) == 3


def test_select_llm_picks_used_and_validated(transcript):
    words, sents = transcript
    seen = {}

    def fake(system, user):
        seen["prompt"] = user
        # id 0 twice (dup overlap), an out-of-range id, then id 1.
        return "<think>hmm</think>```json\n" + json.dumps([
            {"id": 0, "title": "Own your stack", "reason": "strong question hook", "score": 91},
            {"id": 0, "title": "dup"},
            {"id": 999, "title": "bogus"},
            {"id": "x"},
        ]) + "\n```"

    sel = select_clips(words, sents, 2, llm=fake)
    assert "[0]" in seen["prompt"]
    assert sel.engine == "cluster-llm+heuristic"
    by = sorted(p.selector for p in sel.picks)
    assert by == ["cluster-llm", "heuristic"]
    llm_pick = next(p for p in sel.picks if p.selector == "cluster-llm")
    assert llm_pick.title == "Own your stack" and llm_pick.score == 91
    assert llm_pick.reason.startswith("cluster-llm: strong question hook")
    a, b = sel.picks
    assert a.end <= b.start


def test_parse_llm_rejects_garbage(transcript):
    words, sents = transcript
    cands = build_candidates(sents, 20, 60)
    with pytest.raises(ValueError):
        parse_llm_picks("no json here", cands, 3)


def test_too_short_transcript_raises(make_words):
    words = make_words(["Short. Very short."])
    with pytest.raises(ValueError):
        select_clips(words, split_sentences(words), 3)
