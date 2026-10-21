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
    f for d in ("source_and_pdf", "just_pdf", "synthetic")
    for f in sorted(glob.glob(f"examples/{d}/*.pdf")) if "Ding dong" not in f
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


# Real CPDL editions the vector path already reads (nearly) exactly: guard them
CPDL_GOOD = {
    "Ad_te": 1.0, "D-Ju_MS_33_Ab_Hoc_Familia": 1.0, "Vierendeels-ad_coenam": 1.0,
    "Nh_Aiblinger_VP_01_Dixit": 1.0, "Nh_Reichard_Abends": 1.0, "Selig_sind_die_Toten": 0.99,
    "Ad_te_levavi_Balbi": 0.98, "Nh_SchumannCL_Abendfeier": 0.98, "Nh_Heim_Ade_zur_guten_Nacht": 0.98,
    "Ad_Dominum_Scarlatti": 0.97, "Ad_te_levavi_Witt": 0.99, "Psalm_89.2_Becker_Psalter": 0.96,
    "AcworthHolyoke1803bxm": 0.82,
    "Abisso_e_fuoco_della_tua_carità": 0.96, "Ave_VerumSATB": 0.96, "Nh_Andre_Abendlied": 0.95,
    # capella, LilyPond, driver-renamed and Type3 music fonts
    "Hane-ach": 1.0, "Hassler-Ach_Gott_vom_Himmel": 1.0, "Treue_Liebe": 1.0,
    "Lassus_A_ce_matin": 0.97, "Br-120": 0.95, "Stalnaker_ASpotlessRose_Chants_v03": 0.95,
    "Franck-Ach_herr_mich_armen_Sünder": 0.94, "Bell_A_highland_lad": 0.93, "Ws-brah-abs": 0.91,
    "Acceptabis_sacrificium": 0.95, "Verkaik-Absalom": 0.99, "Clemens-Adieu_delices": 0.98,
    "Morera.ALaPromisionSagrada.440hz": 0.7,
}


@pytest.mark.parametrize("stem", sorted(CPDL_GOOD))
def test_cpdl_vector_notes(stem, tmp_path):
    import sys
    sys.path.insert(0, "tools")
    from bench_synthetic import compare
    pdf = f"examples/cpdl/{stem}.pdf"
    out = str(tmp_path / "out.musicxml")
    convert_score(pdf, out, engine="vector", use_ai=False)
    r = compare(out, pdf[:-4] + ".mxl")
    assert r["notes"] >= CPDL_GOOD[stem], r
    assert r["valid"] in (None, True), r


# MuseTrainer piano pieces (vector engine): minimum note score
MUSETRAINER_GOOD = {
    "Bach_Minuet_in_G_Major_BWV_Anh._114": 1.0, "Fur_Elise_Easy_Piano": 1.0,
    "Happy_Birthday_To_You_Piano": 1.0, "Ode_to_Joy_Easy_variation": 1.0,
    "Greensleeves_for_Piano_easy_and_beautiful": 1.0, "Waltz_in_A_MinorChopin": 1.0,
    "Swan_Lake": 1.0, "Canon_in_D_easy": 0.96, "Carol_of_the_Bells_easy_piano": 0.97,
    "Prelude_I_in_C_major_BWV_846_-_Well_Tempered_Clavier_First_Book": 0.97,
    "J._S._Bach_-_Air_on_the_G_String_Piano_arrangement": 0.93,
}


@pytest.mark.parametrize("stem", sorted(MUSETRAINER_GOOD))
def test_musetrainer_vector_notes(stem, tmp_path):
    import sys
    sys.path.insert(0, "tools")
    from bench_synthetic import compare
    pdf = f"examples/musetrainer/{stem}.pdf"
    out = str(tmp_path / "out.musicxml")
    convert_score(pdf, out, engine="vector", use_ai=False)
    r = compare(out, pdf[:-4] + ".mxl")
    assert r["notes"] >= MUSETRAINER_GOOD[stem], r
    assert r["valid"] in (None, True), r


# Choir mode: accompanied scores keep exactly their sung parts
CHOIR_ACCOMPANIED = {
    "Ave_VerumSATB": 4, "Farrant_High_Evening": 4, "Morera.ALaPromisionSagrada.440hz": 4,
    "Nh_Aiblinger_VP_01_Dixit": 4,
}


@pytest.mark.parametrize("stem", sorted(CHOIR_ACCOMPANIED))
def test_choir_mode_drops_accompaniment(stem, tmp_path):
    import sys
    sys.path.insert(0, "tools")
    from bench_synthetic import compare
    pdf = f"examples/cpdl/{stem}.pdf"
    out = str(tmp_path / "out.musicxml")
    convert_score(pdf, out, engine="vector", use_ai=False, choir=True)
    r = compare(out, pdf[:-4] + ".mxl", choir=True)
    assert r["parts"] == f"{CHOIR_ACCOMPANIED[stem]}/{CHOIR_ACCOMPANIED[stem]}", r
    assert r["notes"] >= 0.75 and r["valid"] in (None, True), r


@pytest.mark.parametrize("stem", ["Hane-ach", "Nh_Mendelssohn_Abendlied_19_4", "Treue_Liebe"])
def test_choir_mode_keeps_unaccompanied(stem, tmp_path):
    import music21
    pdf = f"examples/cpdl/{stem}.pdf"
    a, b = str(tmp_path / "a.musicxml"), str(tmp_path / "b.musicxml")
    convert_score(pdf, a, engine="vector", use_ai=False)
    convert_score(pdf, b, engine="vector", use_ai=False, choir=True)
    assert len(music21.converter.parse(a).parts) == len(music21.converter.parse(b).parts)


def test_choir_mode_leaves_piano_alone(tmp_path):
    import music21
    out = str(tmp_path / "out.musicxml")
    convert_score("examples/musetrainer/Fur_Elise_Easy_Piano.pdf", out, engine="vector", use_ai=False, choir=True)
    assert len(music21.converter.parse(out).parts) == 2


@pytest.mark.parametrize("stem,names", [
    ("Nh_Mendelssohn_Abendlied_19_4", ["Soprano", "Alto", "Tenor", "Bass"]),
    ("Bell_A_highland_lad", ["Soprano", "Alto", "Tenor", "Bass"]),
    ("Nh_Andre_Abendlied", ["Soprano", "Alto", "Voice 2", "Voice 3"]),
])
def test_split_voices(stem, names, tmp_path):
    import music21
    out = str(tmp_path / "out.musicxml")
    convert_score(f"examples/cpdl/{stem}.pdf", out, engine="vector", use_ai=False, choir=True, split_voices=True)
    s = music21.converter.parse(out)
    assert [p.partName for p in s.parts] == names
    # The two parts of one staff keep the same bars
    lens = [[m.duration.quarterLength for m in p.getElementsByClass("Measure")] for p in s.parts[:2]]
    assert lens[0] == lens[1]
    # The upper voice of a shared staff lies above the lower one
    hi = [n.pitches[-1].midi for n in s.parts[0].flatten().notes]
    lo = [n.pitches[0].midi for n in s.parts[1].flatten().notes]
    assert sum(hi) / len(hi) > sum(lo) / len(lo)


def test_split_voices_leaves_satb_alone(tmp_path):
    import music21
    out = str(tmp_path / "out.musicxml")
    convert_score("examples/cpdl/Br-120.pdf", out, engine="vector", use_ai=False, split_voices=True)
    assert len(music21.converter.parse(out).parts) == 4
