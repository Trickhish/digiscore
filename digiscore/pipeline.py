"""
Main processing pipeline for digiScore.
Detects score type (vector vs scanned/raster), routes to the appropriate engine,
and enriches the resulting MusicXML with AI metadata.
"""

import os
import warnings
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import pymupdf

from digiscore.omr.notelite import is_notelite_available, run_notelite_omr
from digiscore.omr import lyrics_ocr
from digiscore.omr.book import read_book
from digiscore.omr.postprocess import merge_split_parts
from digiscore.omr.repair import repair_missing_dots, repair_missing_rests, split_merged_measures
from digiscore.vector.parser import parse_vector_score
from digiscore.vector.staff import detect_staves_on_page
from digiscore.vision.client import VisionClient, enrich_musicxml_with_metadata
from digiscore.writer.musicxml import build_musicxml


def is_vector_score(pdf_path: str) -> bool:
    """
    Determines whether a PDF is a native vector score or a scanned/raster score.
    Checks for the presence of vector staff lines and vector drawings.
    """
    doc = pymupdf.open(pdf_path)
    if len(doc) == 0:
        return False

    # Check the first up to 2 pages
    for page_idx in range(min(2, len(doc))):
        page = doc[page_idx]
        drawings = page.get_drawings()
        if len(drawings) < 10:
            continue
        staves = detect_staves_on_page(page, page_idx)
        if len(staves) > 0:
            doc.close()
            return True

    doc.close()
    return False


def save_as_mxl(musicxml_content: str, output_path: str) -> str:
    """
    Saves a MusicXML string as a compressed .mxl file.
    """
    container_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<container>\n'
        '  <rootfiles>\n'
        '    <rootfile full-path="score.xml"/>\n'
        '  </rootfiles>\n'
        '</container>'
    )
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("score.xml", musicxml_content.encode("utf-8"))
    return output_path


def read_musicxml_from_file(file_path: str) -> str:
    """
    Reads MusicXML from either an uncompressed (.musicxml/.xml) or compressed (.mxl) file.
    """
    p = Path(file_path)
    if p.suffix.lower() == ".mxl":
        with zipfile.ZipFile(file_path, "r") as z:
            # Check META-INF/container.xml for rootfile
            if "META-INF/container.xml" in z.namelist():
                import xml.etree.ElementTree as ET
                try:
                    c_tree = ET.fromstring(z.read("META-INF/container.xml"))
                    rf = c_tree.find(".//rootfile")
                    if rf is not None and "full-path" in rf.attrib:
                        return z.read(rf.attrib["full-path"]).decode("utf-8")
                except Exception:
                    pass
            # Fallback to any .xml in archive
            for name in z.namelist():
                if name.endswith(".xml") and not name.startswith("META-INF"):
                    return z.read(name).decode("utf-8")
        raise ValueError(f"No valid XML rootfile found in MXL archive: {file_path}")
    else:
        return p.read_text(encoding="utf-8")


def _same_bar_length(a: str, b: str) -> bool:
    """'2/2' and '4/4' both last four quarter notes."""
    from fractions import Fraction

    try:
        (na, da), (nb, db) = (map(int, s.split("/")) for s in (a, b))
    except ValueError:
        return False
    return Fraction(na, da) == Fraction(nb, db)


def convert_score(
    input_path: str,
    output_path: Optional[str] = None,
    engine: str = "auto",  # 'auto', 'vector', or 'notelite'
    use_ai: bool = True,
    sheets: Optional[str] = None,
) -> str:
    """
    Converts a singing score PDF to MusicXML 4.0.

    Args:
        input_path: Path to the input PDF file.
        output_path: Destination path for .musicxml or .mxl. If omitted, uses input stem.
        engine: 'auto', 'vector', or 'notelite'.
        use_ai: Whether to enrich metadata and lyrics using AI (ai.dury.dev).
        sheets: Sheet selection for NoteLite OMR (e.g. "1" or "1-3").

    Returns:
        The path of the created MusicXML file.
    """
    abs_input = os.path.abspath(input_path)
    if not os.path.isfile(abs_input):
        raise FileNotFoundError(f"Input file not found: {abs_input}")

    input_stem = Path(abs_input).stem
    if not output_path:
        output_path = str(Path(abs_input).parent / f"{input_stem}.musicxml")
    out_p = Path(output_path)
    is_mxl = out_p.suffix.lower() == ".mxl"

    # Engine selection
    selected_engine = engine.lower()
    if selected_engine == "auto":
        if is_vector_score(abs_input):
            selected_engine = "vector"
        else:
            selected_engine = "notelite"

    xml_content = ""

    if selected_engine == "vector":
        doc = pymupdf.open(abs_input)
        score_data = parse_vector_score(doc)
        doc.close()
        xml_content = build_musicxml(score_data)
    elif selected_engine in ("notelite", "omr"):
        if not is_notelite_available():
            raise RuntimeError("NoteLite OMR engine is requested but not available.")
        with tempfile.TemporaryDirectory(prefix="digiscore_omr_") as tmp_dir:
            tmp_mxl = os.path.join(tmp_dir, f"{input_stem}.mxl")
            tmp_book = os.path.join(tmp_dir, f"{input_stem}.omr")
            # Low-resolution scans: upscale and binarise before OMR
            omr_input = abs_input
            try:
                from digiscore.omr.enhance import enhance_pdf, needs_enhancement
                if needs_enhancement(abs_input):
                    omr_input = enhance_pdf(abs_input, os.path.join(tmp_dir, f"{input_stem}.pdf"))
            except Exception as e:
                warnings.warn(f"Scan enhancement skipped: {e}")
            run_notelite_omr(omr_input, output_path=tmp_mxl, sheets=sheets, keep_book=tmp_book)
            xml_content = read_musicxml_from_file(tmp_mxl)

            # If the OMR missed the time signature, ask the AI (or infer it from the
            # bar lengths), write it into the project and let NoteLite redo the rhythm
            if os.path.isfile(tmp_book) and "<time>" not in xml_content:
                try:
                    from digiscore.omr.inject_time import (
                        detect_time_signature, guess_time_from_durations, inject_time_into_omr,
                    )
                    ts = None
                    if use_ai:
                        vc = VisionClient(timeout=90)
                        if vc.is_available:
                            ts = detect_time_signature(abs_input, vc)
                    # Vision models misread time signatures now and then: keep the AI
                    # answer only if its bar length matches what most bars already fill
                    guess = guess_time_from_durations(xml_content)
                    if guess and (not ts or not _same_bar_length(ts, guess)):
                        ts = guess
                    if ts:
                        print(f"[*] Time signature missing in OMR output; using {ts}")
                    if ts and inject_time_into_omr(tmp_book, ts):
                        # The project only holds the selected sheets: no -sheets here
                        run_notelite_omr(tmp_book, output_path=tmp_mxl, keep_book=tmp_book)
                        xml_content = read_musicxml_from_file(tmp_mxl)
                except Exception as e:
                    warnings.warn(f"Time signature injection skipped: {e}")

            book = None
            if os.path.isfile(tmp_book):
                keep = os.environ.get("DIGISCORE_KEEP_OMR")
                if keep:
                    import shutil
                    os.makedirs(keep, exist_ok=True)
                    shutil.copy2(tmp_book, keep)
                    shutil.copy2(tmp_mxl, keep)
                    if omr_input != abs_input:
                        shutil.copy2(omr_input, os.path.join(keep, f"{input_stem}.enhanced.pdf"))
                try:
                    book = read_book(tmp_book)
                except Exception as e:
                    warnings.warn(f"Could not read the OMR project file: {e}")
            if book is not None:
                # Rhythm repairs checked against the page image
                try:
                    xml_content, _ = repair_missing_dots(xml_content, book, omr_input)
                    xml_content, _ = repair_missing_rests(xml_content, book, omr_input)
                except Exception as e:
                    warnings.warn(f"Measure repair skipped: {e}")
                # Bars that still do not add up: onsets from the other staves
                if os.environ.get("DIGISCORE_NO_ALIGN") != "1":
                    try:
                        from digiscore.omr.align import realign_from_other_staves
                        xml_content, _ = realign_from_other_staves(xml_content, book)
                    except Exception as e:
                        warnings.warn(f"Cross-staff realignment skipped: {e}")
                if lyrics_ocr.is_available():
                    try:
                        vision = VisionClient(timeout=60) if use_ai else None
                        if vision is not None and not vision.is_available:
                            vision = None
                        xml_content = lyrics_ocr.relyric(xml_content, book, abs_input, vision=vision)
                    except Exception as e:
                        warnings.warn(f"Lyrics re-recognition skipped: {e}")
                # Last: changes measure indexes, which the steps above rely on
                try:
                    xml_content, _ = split_merged_measures(xml_content, book, omr_input)
                except Exception as e:
                    warnings.warn(f"Measure split skipped: {e}")
            xml_content = merge_split_parts(xml_content)
    else:
        raise ValueError(f"Unknown engine: {engine}. Choose 'auto', 'vector', or 'notelite'.")

    # AI Metadata Enrichment
    if use_ai:
        vc = VisionClient()
        if vc.is_available:
            meta = vc.extract_metadata(abs_input)
            if meta and not meta.get("error"):
                xml_content = enrich_musicxml_with_metadata(xml_content, meta)

    # Save output
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if is_mxl:
        save_as_mxl(xml_content, str(out_p))
    else:
        out_p.write_text(xml_content, encoding="utf-8")

    return str(out_p)
