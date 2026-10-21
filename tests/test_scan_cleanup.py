"""Unit tests for the scan clean-up and OMR post-processing steps."""

import numpy as np
import pytest

from digiscore.omr.enhance import dewarp, flatten, is_photo, staff_interline
from digiscore.omr.postprocess import _SURE_CHORD, dedupe_ties, drop_chord_symbol_lyrics, drop_prose_directions


def _staff_page(h=600, w=800, bow=0.0, paper=255, gradient=0):
    """A synthetic page: three staves of five lines, optionally bowed and grey."""
    img = np.full((h, w), paper, np.float32)
    if gradient:
        img -= np.linspace(0, gradient, w)[None, :]
    xs = np.arange(w)
    for top in (100, 260, 420):
        for k in range(5):
            y0 = top + 12 * k
            ys = (y0 + bow * ((xs - w / 2) / (w / 2)) ** 2).astype(int)
            for dy in (0, 1):
                img[np.clip(ys + dy, 0, h - 1), xs] = 20
    return np.clip(img, 0, 255).astype(np.uint8)


def test_staff_interline():
    assert staff_interline(_staff_page()) == 12


def test_photo_detection():
    assert not is_photo(_staff_page())
    assert is_photo(_staff_page(paper=190, gradient=60))


def test_flatten_evens_out_lighting():
    out = flatten(_staff_page(paper=200, gradient=80))
    # The paper is white everywhere, the lines black
    assert out[50, 10] == 255 and out[50, 790] == 255
    assert out[100, 400] == 0


def test_dewarp_straightens_bowed_lines():
    page = _staff_page(bow=8.0)
    out = dewarp(page)

    def line_y(img, x):
        col = img[80:140, x] < 128
        return 80 + int(np.argmax(col))
    before = abs(line_y(page, 40) - line_y(page, 400))
    after = abs(line_y(out, 40) - line_y(out, 400))
    assert before >= 6 and after <= 2


def test_dewarp_leaves_flat_page_alone():
    page = _staff_page()
    assert (dewarp(page) == page).all()


@pytest.mark.parametrize("text,chord", [
    ("DO", True), ("Sib", True), ("rém", True), ("LA4", True), ("Solm", True), ("solm6", True),
    ("la", False), ("La", False), ("Si", False), ("Mi", False), ("Do", False), ("A", False), ("sol", False),
])
def test_sure_chord(text, chord):
    assert bool(_SURE_CHORD.match(text)) == chord


def _score(parts_notes):
    parts = []
    for pid, notes in parts_notes:
        body = "".join(
            f'<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice>'
            f'<type>quarter</type>{"".join(f"<lyric number=\"{n}\"><text>{t}</text></lyric>" for n, t in lyr)}</note>'
            for lyr in notes)
        parts.append((pid, f'<part id="{pid}"><measure number="1">{body}</measure></part>'))
    plist = "".join(f'<score-part id="{pid}"><part-name>V</part-name></score-part>' for pid, _ in parts)
    return f'<score-partwise version="4.0"><part-list>{plist}</part-list>{"".join(p for _, p in parts)}</score-partwise>'


def test_chord_names_and_stray_text_dropped():
    verse = [[("1", w)] for w in "Je veux voir Dieu je veux con tem pler mon".split()]
    verse[2].append(("1", "x"))  # keep shape
    p1 = [[("1", "Je")], [("1", "Sib")], [("1", "voir")], [("1", "rém")]] + [[("1", w)] for w in "je veux con tem".split()]
    p2 = [[("1", "50m")], []] + [[] for _ in range(20)] + [[("1", "Gloire")]]
    out = drop_chord_symbol_lyrics(_score([("P1", p1), ("P2", p2)]))
    assert "Sib" not in out and "rém" not in out and "voir" in out
    assert "50m" not in out and "Gloire" not in out


def test_dedupe_ties():
    xml = ('<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>V</part-name></score-part></part-list>'
           '<part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>'
           '<tie type="start"/><tie type="start"/><tie type="stop"/><voice>1</voice><type>quarter</type>'
           '<notations><tied type="start"/><tied type="start"/></notations></note></measure></part></score-partwise>')
    out = dedupe_ties(xml)
    assert out.count('<tie type="start"/>') == 1 and out.count('<tie type="stop"/>') == 1
    assert out.index('<tie type="stop"/>') < out.index('<tie type="start"/>')
    assert out.count("<tied") == 1


def test_prose_directions_dropped():
    xml = ('<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>V</part-name></score-part></part-list>'
           '<part id="P1"><measure number="1">'
           '<direction placement="above"><direction-type><words>Allegro</words></direction-type></direction>'
           '<direction placement="below"><direction-type><words>de vant qui .7</words></direction-type></direction>'
           '<direction><direction-type><words>Le Seigneur est le rempart de ma vie</words></direction-type></direction>'
           '</measure></part></score-partwise>')
    out = drop_prose_directions(xml)
    assert "Allegro" in out and "vant" not in out and "rempart" not in out
