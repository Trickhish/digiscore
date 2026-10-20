"""
Wrapper for NoteLite OMR engine.
Executes NoteLite / Audiveris in batch headless mode to transcribe scanned/raster scores to MusicXML.
"""

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional


DEFAULT_NOTELITE_PATHS = [
    os.environ.get("NOTELITE_BIN"),
    "/tmp/opencode/tools/app-5.13.0/bin/NoteLite",
    "/opt/notelite/bin/NoteLite",
    "/usr/local/bin/NoteLite",
    shutil.which("NoteLite"),
    shutil.which("audiveris"),
]

DEFAULT_TESSDATA_DIRS = [
    os.environ.get("TESSDATA_PREFIX"),
    "/opt/tessdata",
    "/usr/share/tessdata",
    "/usr/share/tesseract-ocr/5/tessdata",
]


def find_notelite_binary() -> Optional[str]:
    """Finds the NoteLite / Audiveris executable path."""
    for path in DEFAULT_NOTELITE_PATHS:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def is_notelite_available() -> bool:
    """Checks if NoteLite is installed and available."""
    return find_notelite_binary() is not None


def find_tessdata_dir() -> Optional[str]:
    """Finds the tessdata directory containing traineddata models."""
    for p in DEFAULT_TESSDATA_DIRS:
        if p and os.path.isdir(p) and (Path(p) / "eng.traineddata").is_file():
            return p
    return None


def _count_sheets(spec: str, n_pages: int) -> int:
    """Number of pages selected by a NoteLite sheet spec such as "1 3-5"."""
    total = 0
    for tok in spec.replace(",", " ").split():
        if "-" in tok:
            a, b = tok.split("-", 1)
            total += max(0, min(int(b), n_pages) - int(a) + 1)
        elif tok.isdigit():
            total += 1
    return total or n_pages


def _read_score_xml(path: Path) -> str:
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith(".xml") and not n.startswith("META-INF")][0]
            return z.read(name).decode("utf-8")
    return path.read_text(encoding="utf-8")


def join_movements(xmls: List[str]) -> str:
    """Append the measures of later movements to the first one, part by part."""
    from lxml import etree

    from digiscore.omr.postprocess import merge_split_parts

    roots = [etree.fromstring(merge_split_parts(x).encode("utf-8")) for x in xmls]
    base = roots[0]
    base_parts = base.findall("part")
    names = [(sp.findtext("part-name") or "").strip().lower() for sp in base.iter("score-part")]
    for other in roots[1:]:
        o_parts = other.findall("part")
        o_names = [(sp.findtext("part-name") or "").strip().lower() for sp in other.iter("score-part")]
        if len(o_parts) == len(base_parts):
            pairs = list(zip(base_parts, o_parts))
        elif len(set(names)) == len(names) and all(n in names for n in o_names):
            pairs = [(base_parts[names.index(n)], op) for n, op in zip(o_names, o_parts)]
        else:
            # Unmatched layout: keep the movements' measures aligned in time by
            # giving every base part as many empty bars as this movement has
            pairs = []
            count = max((len(op.findall("measure")) for op in o_parts), default=0)
            for bp in base_parts:
                for _ in range(count):
                    m = etree.SubElement(bp, "measure")
                    etree.SubElement(etree.SubElement(m, "note"), "rest").set("measure", "yes")
                    n = m.find("note")
                    etree.SubElement(n, "duration").text = "0"
                    etree.SubElement(n, "voice").text = "1"
        for bp, op in pairs:
            for m in op.findall("measure"):
                bp.append(m)
    for p in base_parts:
        first = p.find("measure")
        num = 0 if first is not None and first.get("implicit") == "yes" else 1
        for m in p.findall("measure"):
            # Pickup flags of later movements are not pickups of this score
            if num > 1 and m.get("implicit") == "yes":
                del m.attrib["implicit"]
            m.set("number", str(num))
            num += 1
    return etree.tostring(base.getroottree(), xml_declaration=True, encoding="UTF-8").decode("utf-8")


def _project_sheets(omr_path: str) -> List[int]:
    """Sheet numbers present in an .omr project file."""
    try:
        with zipfile.ZipFile(omr_path) as z:
            return sorted(
                int(re.search(r"#(\d+)/", n).group(1))
                for n in z.namelist() if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", n)
            )
    except Exception:
        return []


def _drop_invalid_sheets(omr_path: str) -> bool:
    """
    Remove the sheets NoteLite flagged as invalid from a project, keeping at
    least one. Returns True if something was removed.
    """
    with zipfile.ZipFile(omr_path) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    book = entries.get("book.xml", b"").decode("utf-8")
    bad = re.findall(r'<sheet number="(\d+)"[^>]*invalid="true"', book)
    total = re.findall(r'<sheet number="(\d+)"', book)
    if not bad or len(bad) >= len(total):
        return False
    for num in bad:
        book = re.sub(rf'<sheet number="{num}"[^>]*invalid="true"[^>]*/>', "", book)
        book = re.sub(rf'<sheet number="{num}"[^>]*invalid="true"[^>]*>.*?</sheet>', "", book, flags=re.S)
        for n in [n for n in entries if n.startswith(f"sheet#{num}/")]:
            del entries[n]
    entries["book.xml"] = book.encode("utf-8")
    with zipfile.ZipFile(omr_path, "w", zipfile.ZIP_STORED) as zw:
        for n, data in entries.items():
            zw.writestr(n, data)
    return True


def run_notelite_omr(
    pdf_path: str,
    output_path: Optional[str] = None,
    sheets: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    pdf_dpi: int = 400,
    keep_book: Optional[str] = None,
) -> str:
    """
    Runs NoteLite OMR on the input PDF file and outputs MusicXML (.mxl or .musicxml).

    Args:
        pdf_path: Absolute or relative path to the PDF score.
        output_path: Target path for the generated MusicXML / MXL file.
        sheets: Sheet selection filter, e.g. "1" or "1-3".
        timeout_sec: Maximum time allowed for OMR in seconds.

    Returns:
        Path to the generated MusicXML file.
    """
    notelite_bin = find_notelite_binary()
    if not notelite_bin:
        raise RuntimeError(
            "NoteLite / Audiveris binary not found. Please install NoteLite or set NOTELITE_BIN."
        )

    abs_pdf = os.path.abspath(pdf_path)
    if not os.path.isfile(abs_pdf):
        raise FileNotFoundError(f"Input not found: {abs_pdf}")

    # An .omr project input: NoteLite continues it (steps not yet done), and the
    # -constant PDF options do not apply
    is_project = abs_pdf.lower().endswith(".omr")
    if not is_project:
        try:
            import pymupdf
            pymupdf.open(abs_pdf)
        except Exception as exc:
            raise RuntimeError(f"Input is neither a PDF nor an .omr project: {exc}")

    if timeout_sec is None:
        # Roughly 40-60 s per page at 400 DPI; leave a wide margin
        if is_project:
            n_pages = len(_project_sheets(abs_pdf))
        else:
            try:
                import pymupdf
                n_pages = len(pymupdf.open(abs_pdf))
            except Exception:
                n_pages = 10
        if sheets:
            n_pages = min(n_pages, _count_sheets(str(sheets), n_pages))
        timeout_sec = 120 + 120 * n_pages

    # Prepare environment with TESSDATA_PREFIX
    env = os.environ.copy()
    tessdata = find_tessdata_dir()
    if tessdata:
        env["TESSDATA_PREFIX"] = tessdata

    with tempfile.TemporaryDirectory(prefix="notelite_") as tmp_dir:
        cmd = [
            notelite_bin, "-batch", "-export", "-output", tmp_dir,
        ]
        if not is_project:
            # Scans render too small at the default 300 DPI: notes get dropped
            cmd += [
                "-constant", f"com.notelite.omr.image.ImageLoading.pdfResolution={pdf_dpi}",
                # On blurred or noisy scans the beam thickness is under-estimated and
                # real beams are then rejected; force a realistic lower bound
                "-constant", "com.notelite.omr.sheet.ScaleBuilder.beamMinFraction=0.4",
            ]
        if sheets:
            cmd.extend(["-sheets", str(sheets)])
        if is_project:
            # NoteLite exports a project next to the project file: work on a copy
            local = os.path.join(tmp_dir, os.path.basename(abs_pdf))
            shutil.copy2(abs_pdf, local)
            cmd.extend(["--", local])
        else:
            cmd.extend(["--", abs_pdf])

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            env=env,
        )

        # Look for exported .mxl or .xml file in output directory
        out_files = list(Path(tmp_dir).glob("*.mxl")) + list(Path(tmp_dir).glob("*.xml"))
        # Filter out sheet XMLs or book.xml
        score_files = [f for f in out_files if not f.name.endswith(".omr") and f.name != "book.xml"]

        if not score_files and not is_project:
            # A page without a recognisable system (e.g. a nearly empty last page)
            # makes the whole export fail: drop the invalid sheets and export the rest
            books = list(Path(tmp_dir).glob("*.omr"))
            if books and _drop_invalid_sheets(str(books[0])):
                return run_notelite_omr(str(books[0]), output_path=output_path, timeout_sec=timeout_sec,
                                        keep_book=keep_book)

        if not score_files:
            error_log = proc.stderr or proc.stdout
            raise RuntimeError(
                f"NoteLite completed without producing an export file.\nLog:\n{error_log[-1000:]}"
            )

        # NoteLite may cut the score into movements (name.mvt1.mxl, name.mvt2.mxl...)
        # when a system looks like a new start; on choral scores these are nearly
        # always continuations, so join them back into one score
        mvts = sorted(
            (f for f in score_files if re.search(r"\.mvt\d+\.mxl$", f.name)),
            key=lambda f: int(re.search(r"\.mvt(\d+)\.mxl$", f.name).group(1)),
        )
        if len(mvts) > 1:
            joined = Path(tmp_dir) / "joined.mxl"
            with zipfile.ZipFile(joined, "w", zipfile.ZIP_DEFLATED) as zj:
                zj.writestr(
                    "META-INF/container.xml",
                    '<?xml version="1.0" encoding="UTF-8"?><container><rootfiles>'
                    '<rootfile full-path="score.xml"/></rootfiles></container>',
                )
                zj.writestr("score.xml", join_movements([_read_score_xml(f) for f in mvts]))
            produced_file = joined
        else:
            produced_file = score_files[0]

        # Determine target output path
        if not output_path:
            pdf_stem = Path(abs_pdf).stem
            output_path = str(Path(abs_pdf).parent / f"{pdf_stem}.mxl")

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if str(produced_file).endswith(".mxl") and (out_p.suffix.lower() == ".musicxml" or out_p.suffix.lower() == ".xml"):
            # Unzip internal xml to target
            with zipfile.ZipFile(produced_file, "r") as z:
                # Find the main score xml file
                xml_names = [n for n in z.namelist() if n.endswith(".xml") and not n.startswith("META-INF")]
                if xml_names:
                    xml_content = z.read(xml_names[0])
                    out_p.write_bytes(xml_content)
                else:
                    shutil.copy2(produced_file, output_path)
        else:
            shutil.copy2(produced_file, output_path)

        # Keep the project file next to the output: it holds the page geometry
        if keep_book:
            books = list(Path(tmp_dir).glob("*.omr"))
            if books:
                shutil.copy2(books[0], keep_book)

        return str(output_path)
