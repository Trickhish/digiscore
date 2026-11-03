"""Offline tests for the AI lyric reading helpers (no network)."""

import pytest

from digiscore.omr.lyrics_ocr import OcrSyllable, assign_to_chords, respace
from digiscore.vision.client import _keep_ocr_cuts


@pytest.mark.parametrize("ocr,read,want", [
    # Merged syllables split, letters fixed
    (["sonchan", "e", "Dieuchan"], ["son", "chan", "e", "Dieu", "chan"], ["son", "chan", "e", "Dieu", "chan"]),
    (["e", "pa"], ["è", "pa"], ["è", "pa"]),
    (["Hai", "le", "lu"], ["Hal", "le", "lu"], ["Hal", "le", "lu"]),
    # A boundary the OCR saw is not moved
    (["al", "luctus", "in", "ia"], ["al", "luc", "tu", "sin", "ia"], ["al", "luctus", "in", "ia"]),
    (["gn", "us"], ["gnus"], ["gn", "us"]),
])
def test_keep_ocr_cuts(ocr, read, want):
    assert _keep_ocr_cuts(ocr, read) == want


def _syl(text, x0, w=6.0):
    n = len(text)
    chars = [(c, x0 + w * k, x0 + w * (k + 1) - 1) for k, c in enumerate(text)]
    return OcrSyllable(text, x0, x0 + w * n - 1, chars=chars)


def test_respace_places_split_syllables():
    syls = [_syl("sonchan", 100), _syl("e", 200)]
    out = respace(syls, ["son", "chan", "e"])
    assert [s.text for s in out] == ["son", "chan", "e"]
    assert out[0].x1 < out[1].x0 < out[2].x0
    # Pieces of one OCR syllable are joined by a hyphen
    assert out[0].hyphen_after


class _Chord:
    def __init__(self, cid, cx):
        self.id, self.cx, self.x0 = cid, cx, cx - 3


def test_split_reading_aligns_to_notes():
    syls = respace([_syl("sonchan", 100), _syl("e", 200)], ["son", "chan", "e"])
    chords = [_Chord("a", 108), _Chord("b", 130), _Chord("c", 203)]
    got = assign_to_chords(syls, chords, 8.0)
    assert {k: v[0] for k, v in got.items()} == {"a": "son", "b": "chan", "c": "e"}
