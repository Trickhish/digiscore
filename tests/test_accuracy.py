"""
Accuracy checks against the ground-truth MusicXML exported from the original sources,
and structural checks (bar lengths, voices, schema) on the PDFs without a source.
"""

import difflib
import glob
import os
import tempfile
import warnings

import music21
import pytest

from digiscore.pipeline import convert_score

warnings.filterwarnings("ignore")

PAIRS = [
    "examples/source_and_pdf/cantique-a-notre-dame-de-haurt-charlesivandury",
    "examples/source_and_pdf/cheval-baton",
    "examples/source_and_pdf/resurrexit-tanguy-dionis-du-sejour",
]
VECTOR_PDFS = [
    f for f in sorted(glob.glob("examples/*/*.pdf")) if "Ding dong" not in f
]


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


def _lyrics(part):
    return [(l.number, l.text, l.syllabic) for n in part.flatten().notes for l in n.lyrics]


def _ratio(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(b))


def _convert(pdf, tmp_dir):
    out = os.path.join(tmp_dir, os.path.basename(pdf) + ".musicxml")
    return music21.converter.parse(convert_score(pdf, out, engine="vector", use_ai=False))


@pytest.mark.parametrize("stem", PAIRS)
def test_notes_match_ground_truth(stem):
    with tempfile.TemporaryDirectory() as tmp:
        pred = _convert(stem + ".pdf", tmp)
    gt = music21.converter.parse(stem + ".mxl")
    assert len(pred.parts) == len(gt.parts)
    for pp, gp in zip(pred.parts, gt.parts):
        assert len(pp.getElementsByClass("Measure")) == len(gp.getElementsByClass("Measure"))
        assert _ratio(_events(pp), _events(gp)) >= 0.99, gp.partName


@pytest.mark.parametrize("stem", PAIRS)
def test_lyrics_match_ground_truth(stem):
    with tempfile.TemporaryDirectory() as tmp:
        pred = _convert(stem + ".pdf", tmp)
    gt = music21.converter.parse(stem + ".mxl")
    for pp, gp in zip(pred.parts, gt.parts):
        g = _lyrics(gp)
        p = _lyrics(pp)
        if not g:
            assert not p, f"spurious lyrics on {gp.partName}"
            continue
        assert _ratio(p, g) >= 0.98, gp.partName


def _bar_length_errors(part):
    bad = 0
    ms = list(part.getElementsByClass("Measure"))
    for i, m in enumerate(ms):
        own = m.getElementsByClass("TimeSignature")
        ts = own[0] if own else m.getContextByClass("TimeSignature")
        if ts is None or i in (0, len(ms) - 1):
            continue
        for v in (m.voices or [m]):
            if abs(sum(x.quarterLength for x in v.notesAndRests) - ts.barDuration.quarterLength) > 1e-3:
                bad += 1
                break
    return bad


@pytest.mark.parametrize("pdf", VECTOR_PDFS)
def test_every_bar_fills_its_time_signature(pdf):
    with tempfile.TemporaryDirectory() as tmp:
        score = _convert(pdf, tmp)
    for p in score.parts:
        assert _bar_length_errors(p) == 0, p.partName


XSD = "/tmp/opencode/xsd/musicxml.xsd"


@pytest.mark.skipif(not os.path.isfile(XSD), reason="MusicXML 4.0 XSD not downloaded")
@pytest.mark.parametrize("pdf", VECTOR_PDFS)
def test_output_validates_against_musicxml_schema(pdf):
    from lxml import etree

    schema = etree.XMLSchema(etree.parse(XSD))
    with tempfile.TemporaryDirectory() as tmp:
        out = convert_score(pdf, os.path.join(tmp, "s.musicxml"), engine="vector", use_ai=False)
        assert schema.validate(etree.parse(out)), schema.error_log


SYNTHETIC = sorted(glob.glob("examples/synthetic/*.pdf"))


@pytest.mark.parametrize("pdf", SYNTHETIC)
def test_synthetic_vector_exact(pdf, tmp_path):
    """MuseScore-rendered scores: notes, ties, lyrics and time signatures all exact."""
    import sys
    sys.path.insert(0, "tools")
    from bench_synthetic import compare
    out = str(tmp_path / "out.musicxml")
    convert_score(pdf, out, engine="vector", use_ai=False)
    r = compare(out, pdf[:-4] + ".mxl")
    a, b = r["ties"].split(" ")[0].split("/")
    assert r["notes"] == 1.0 and r["time"] == "ok" and a == b and "+" not in r["ties"], r
    assert r["lyrics"] in (None, 1.0), r
