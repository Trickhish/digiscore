"""
Wrapper for NoteLite OMR engine.
Executes NoteLite / Audiveris in batch headless mode to transcribe scanned/raster scores to MusicXML.
"""

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional


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


def run_notelite_omr(
    pdf_path: str,
    output_path: Optional[str] = None,
    sheets: Optional[str] = None,
    timeout_sec: int = 300,
    pdf_dpi: int = 400,
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
        raise FileNotFoundError(f"Input PDF not found: {abs_pdf}")

    # Prepare environment with TESSDATA_PREFIX
    env = os.environ.copy()
    tessdata = find_tessdata_dir()
    if tessdata:
        env["TESSDATA_PREFIX"] = tessdata

    with tempfile.TemporaryDirectory(prefix="notelite_") as tmp_dir:
        cmd = [
            notelite_bin, "-batch", "-export", "-output", tmp_dir,
            # Scans render too small at the default 300 DPI: notes get dropped
            "-constant", f"com.notelite.omr.image.ImageLoading.pdfResolution={pdf_dpi}",
        ]
        if sheets:
            cmd.extend(["-sheets", str(sheets)])
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

        if not score_files:
            error_log = proc.stderr or proc.stdout
            raise RuntimeError(
                f"NoteLite completed without producing an export file.\nLog:\n{error_log[-1000:]}"
            )

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

        return str(output_path)
