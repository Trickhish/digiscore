"""
Test suite for digiScore pipeline and components.
"""

import os
import tempfile
import zipfile
from pathlib import Path
import music21
import pytest

from digiscore.pipeline import convert_score, is_vector_score, read_musicxml_from_file, save_as_mxl
from digiscore.omr.notelite import is_notelite_available, run_notelite_omr
from digiscore.vision.client import VisionClient, enrich_musicxml_with_metadata


CANTIQUE_PDF = "examples/source_and_pdf/cantique-a-notre-dame-de-haurt-charlesivandury.pdf"
CANTIQUE_MXL = "examples/source_and_pdf/cantique-a-notre-dame-de-haurt-charlesivandury.mxl"
RESURREXIT_PDF = "examples/source_and_pdf/resurrexit-tanguy-dionis-du-sejour.pdf"
DING_DONG_PDF = "examples/just_pdf/Ding dong Merily on high.pdf"


def test_score_type_classification():
    assert is_vector_score(CANTIQUE_PDF) is True
    assert is_vector_score(RESURREXIT_PDF) is True
    assert is_vector_score(DING_DONG_PDF) is False


def test_vector_conversion_cantique():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_xml = os.path.join(tmp_dir, "cantique.musicxml")
        res = convert_score(CANTIQUE_PDF, output_path=out_xml, engine="vector", use_ai=False)
        assert os.path.isfile(res)
        
        # Verify parseable with music21
        score = music21.converter.parse(res)
        notes = list(score.flatten().notes)
        assert len(notes) == 87
        
        # Verify key of D major (2 sharps)
        first_note = notes[0]
        assert first_note.pitch.nameWithOctave == "F#4"


def test_vector_conversion_resurrexit():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_xml = os.path.join(tmp_dir, "resurrexit.musicxml")
        res = convert_score(RESURREXIT_PDF, output_path=out_xml, engine="vector", use_ai=False)
        assert os.path.isfile(res)
        
        score = music21.converter.parse(res)
        assert len(score.parts) == 4
        assert score.parts[0].partName == "Soprano"
        assert score.parts[1].partName == "Alto"
        assert score.parts[3].partName == "Bass"


def test_mxl_packaging_and_reading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_mxl = os.path.join(tmp_dir, "test.mxl")
        sample_xml = '<?xml version="1.0"?><score-partwise version="4.0"><work><work-title>Test</work-title></work></score-partwise>'
        save_as_mxl(sample_xml, out_mxl)
        assert os.path.isfile(out_mxl)
        
        read_back = read_musicxml_from_file(out_mxl)
        assert "<work-title>Test</work-title>" in read_back


def test_notelite_omr_available():
    assert is_notelite_available() is True


def test_notelite_omr_scanned_dingdong():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_mxl = os.path.join(tmp_dir, "dingdong.mxl")
        res = run_notelite_omr(DING_DONG_PDF, output_path=out_mxl, sheets="1", timeout_sec=120)
        assert os.path.isfile(res)
        
        score = music21.converter.parse(res)
        assert len(score.parts) >= 1
        assert len(score.parts[0].getElementsByClass("Measure")) >= 10


def test_enrich_musicxml_metadata():
    base_xml = '<?xml version="1.0"?><score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list><part id="P1"></part></score-partwise>'
    meta = {
        "title": "Gloria in Excelsis",
        "composer": "Antonio Vivaldi",
        "lyricist": "Liturgical",
        "parts": ["Soprano I"],
    }
    enriched = enrich_musicxml_with_metadata(base_xml, meta)
    assert "<work-title>Gloria in Excelsis</work-title>" in enriched
    assert 'type="composer">Antonio Vivaldi</creator>' in enriched
    assert '<part-name>Soprano I</part-name>' in enriched
