import io
import random

import pymupdf
import pytest
from PIL import Image, ImageFilter

from digiscore.omr.timesig import detect_from_image, to_rational
from digiscore.smufl import get_glyph_name
from digiscore.vector.staff import detect_staves_on_page

CASES = [
    ("examples/synthetic/synthetic-010-2-2.pdf", "C|"),
    ("examples/synthetic/synthetic-001-4-4.pdf", "C"),
    ("examples/synthetic/synthetic-002-4-4.pdf", "4/4"),
    ("examples/synthetic/synthetic-005-6-8.pdf", "6/8"),
    ("examples/synthetic/synthetic-009-12-8.pdf", "12/8"),
    ("examples/synthetic/synthetic-008-3-2.pdf", "3/2"),
    ("examples/source_and_pdf/cheval-baton.pdf", "3/4"),
]


def _scan(page, dpi):
    """A blurred, noisy grayscale raster of the page at dpi, upscaled to 400 DPI."""
    rnd = random.Random(0)
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    im = Image.frombytes("L", (pix.width, pix.height), pix.samples).filter(ImageFilter.GaussianBlur(0.5))
    im = im.point(lambda v: max(0, min(255, v + rnd.randint(-10, 10))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    im = Image.open(buf).convert("L")
    return im.resize((int(page.rect.width * 400 / 72), int(page.rect.height * 400 / 72)), Image.LANCZOS)


@pytest.mark.parametrize("pdf,want", CASES)
@pytest.mark.parametrize("dpi", [100, 200])
def test_time_signature_from_scan(pdf, want, dpi):
    page = pymupdf.open(pdf)[0]
    staff = detect_staves_on_page(page, 0)[0]
    z = 400 / 72
    il = staff.line_spacing * z
    # Search window: from the clef to well past the signature, as in the pipeline
    x0 = (staff.x_min + 1.5 * staff.line_spacing) * z
    label, _ = detect_from_image(_scan(page, dpi), staff.top_y * z, il, x0, x0 + 12 * il)
    assert label == want


def test_to_rational():
    assert to_rational("C|") == ("2/2", "cut")
    assert to_rational("C") == ("4/4", "common")
    assert to_rational("6/8") == ("6/8", None)


def test_apply_time_signature_keeps_bar_length():
    from digiscore.omr.inject_time import apply_time_signature
    xml = ('<?xml version="1.0"?><score-partwise version="4.0"><part-list><score-part id="P1">'
           '<part-name>S</part-name></score-part></part-list><part id="P1"><measure number="1">'
           '<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>'
           '</attributes></measure></part></score-partwise>')
    out = apply_time_signature(xml, "2/2", "cut")
    assert '<time symbol="cut">' in out and "<beats>2</beats>" in out
    # A different bar length is not applied
    assert apply_time_signature(xml, "3/4") == xml
