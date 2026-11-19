"""Transcript data model shared by transcription, selection and captions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

TERMINAL = (".", "?", "!")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Sentence:
    index: int
    first_word: int  # index into the word list, inclusive
    last_word: int  # inclusive
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def complete(self) -> bool:
        return self.text.rstrip().endswith(TERMINAL)


def words_from_json(items: list[dict]) -> list[Word]:
    return [Word(float(w["start"]), float(w["end"]), str(w["text"])) for w in items]


def split_sentences(words: list[Word], max_gap: float = 0.8, max_len: float = 15.0) -> list[Sentence]:
    """Group words into sentences.

    A sentence ends at terminal punctuation, at a pause longer than ``max_gap``
    seconds, or when it would run past ``max_len`` seconds (unpunctuated rambling).
    """
    sentences: list[Sentence] = []
    if not words:
        return sentences
    first = 0
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        ends = (
            nxt is None
            or w.text.rstrip().endswith(TERMINAL)
            or (nxt.start - w.end) > max_gap
            or (nxt.end - words[first].start) > max_len
        )
        if ends:
            chunk = words[first : i + 1]
            sentences.append(
                Sentence(
                    index=len(sentences),
                    first_word=first,
                    last_word=i,
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    text=" ".join(x.text.strip() for x in chunk).strip(),
                )
            )
            first = i + 1
    return sentences
