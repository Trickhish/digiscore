"""
Scanned-score accuracy.

The ground-truth examples are turned into image-only PDFs (rendered at 200 DPI,
slightly rotated, blurred, noisy, JPEG-compressed) and read through the OMR path.
These tests take a few minutes; run them with:  pytest -m slow
"""

import io
import os
import random
import tempfile
import warnings
import difflib

import music21
import pymupdf
import pytest
from PIL import Image, ImageFilter

from digiscore.omr.notelite import is_notelite_available
from digiscore.pipeline import convert_score

warnings.filterwarnings("ignore")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_notelite_available(), reason="NoteLite not installed"),
]

STEMS = [
    "examples/source_and_pdf/cantique-a-notre-dame-de-haurt-charlesivandury",
    "examples/source_and_pdf/cheval-baton",
]


def make_scan(pdf_path: str, out_path: str, seed: int = 0) -> None:
    rnd = random.Random(seed)
    src = pymupdf.open(pdf_path)
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
        im = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        im = im.rotate(0.4, resample=Image.BICUBIC, fillcolor=255).filter(ImageFilter.GaussianBlur(0.6))
        im = im.point(lambda v: max(0, min(255, v + rnd.randint(-12, 12))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        p = out.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=buf.getvalue())
    out.save(out_path)


def _events(part):
    out = []
    for n in part.flatten().notesAndRests:
        if n.isRest:
            p = "R"
        elif n.isChord:
            p = ".".join(sorted(x.nameWithOctave for x in n.pitches))
        else:
            p = n.pitch.nameWithOctave
        out.append((p, float(n.quarterLength)))
    return out


def _ratio(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(b))


@pytest.mark.parametrize("stem", STEMS)
def test_scanned_score_matches_ground_truth(stem):
    with tempfile.TemporaryDirectory() as tmp:
        scan = os.path.join(tmp, "scan.pdf")
        make_scan(stem + ".pdf", scan)
        assert not pymupdf.open(scan)[0].get_text().strip()  # really image-only
        out = convert_score(scan, os.path.join(tmp, "out.musicxml"), engine="auto", use_ai=False)
        pred = music21.converter.parse(out)
    gt = music21.converter.parse(stem + ".mxl")
    assert len(pred.parts) == len(gt.parts)
    for pp, gp in zip(pred.parts, gt.parts):
        # OMR on scans can still lose a bar now and then (e.g. a barline taken for a stem)
        assert abs(len(pp.getElementsByClass("Measure")) - len(gp.getElementsByClass("Measure"))) <= 1
        assert _ratio(_events(pp), _events(gp)) >= 0.95, gp.partName
