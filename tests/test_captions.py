from __future__ import annotations

import re

from localclip.captions import (
    CaptionStyle,
    ass_time,
    build_ass,
    caption_events,
    chunk_words,
    clean_word,
    relative_words,
)
from localclip.words import Word


def test_ass_time_format():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(61.234) == "0:01:01.23"
    assert ass_time(3725.999) == "1:02:06.00"
    assert ass_time(-1) == "0:00:00.00"


def test_clean_word_escapes_ass_control_chars():
    assert clean_word("{\\b1}hi,", upper=False) == "(/b1)hi"
    assert clean_word("ready?", upper=True) == "READY?"


def test_chunking_limits_and_breaks(make_words):
    words = make_words(["one two three four five. six seven", "eight"])
    chunks = chunk_words(words, max_words=3, max_chars=20, max_gap=0.45)
    texts = [[w.text for w in c] for c in chunks]
    assert texts[0] == ["one", "two", "three"]
    assert texts[1] == ["four", "five."]  # punctuation closes a chunk
    assert texts[2] == ["six", "seven"]  # paragraph pause closes a chunk
    assert texts[3] == ["eight"]
    assert all(len(c) <= 3 for c in chunks)


def test_relative_words_shift_and_clamp():
    words = [Word(9.5, 10.2, "a"), Word(10.4, 10.8, "b"), Word(19.9, 20.6, "c"), Word(21, 22, "d")]
    rel = relative_words(words, 10.0, 20.0)
    assert [w.text for w in rel] == ["a", "b", "c"]
    assert rel[0].start == 0.0 and abs(rel[0].end - 0.2) < 1e-9
    assert abs(rel[1].start - 0.4) < 1e-9
    assert rel[2].end == 10.0


def test_events_follow_words_without_overlap(make_words):
    words = make_words(["Why do most people pay for software? They could run it at home."], wps=2.5)
    dur = words[-1].end + 1.0
    events = caption_events(words, dur)
    # One event per word, active word highlighted exactly once.
    assert len(events) == len(words)
    for e, w in zip(events, words):
        assert e.text.count("\\c&H0000E6FF") == 1
        assert clean_word(w.text, True) in e.text
    for a, b in zip(events, events[1:]):
        assert a.end <= b.start + 1e-9
        assert a.start < a.end
    assert events[0].start == words[0].start
    assert events[-1].end <= dur
    # Each word's event starts at that word (except the first of a chunk, which starts the chunk).
    for e, w in zip(events, words):
        assert e.start <= w.start + 1e-9


def test_events_clamped_to_clip(make_words):
    words = make_words(["alpha beta gamma delta"], wps=2.0)
    dur = 1.5  # cut mid-speech
    rel = relative_words(words, 0.0, dur)
    events = caption_events(rel, dur)
    assert events and all(0 <= e.start < e.end <= dur for e in events)


def test_build_ass_header_and_lines(make_words):
    words = make_words(["Hello world."])
    events = caption_events(words, 3.0)
    ass = build_ass(events, 1080, 1920, CaptionStyle())
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert re.search(r"^Style: Word,Inter,92,", ass, re.M)
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(lines) == 2
    assert lines[0].startswith("Dialogue: 0,0:00:00.00,")
