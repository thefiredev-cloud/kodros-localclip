from __future__ import annotations

import pytest

from localclip.words import Word


def synth_words(paragraphs: list[str], wps: float = 2.6, sentence_gap: float = 0.3, para_gap: float = 1.2, t0: float = 0.0) -> list[Word]:
    """Evenly paced synthetic word timings: each word lasts 85% of its slot."""
    words: list[Word] = []
    t = t0
    slot = 1.0 / wps
    for p in paragraphs:
        for tok in p.split():
            words.append(Word(round(t, 3), round(t + slot * 0.85, 3), tok))
            t += slot
            if tok.endswith((".", "?", "!")):
                t += sentence_gap
        t += para_gap
    return words


@pytest.fixture
def make_words():
    return synth_words
