"""
SMuFL (Standard Music Font Layout) and Legacy (Sibelius Opus, etc.)
glyph tables and mapping utilities.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

SMUFL_PATH = Path(__file__).resolve().parent.parent / "smufl_glyphs.json"

_SMUFL_BY_CODEPOINT: Dict[str, str] = {}
_SMUFL_BY_NAME: Dict[str, str] = {}


def load_smufl() -> None:
    global _SMUFL_BY_CODEPOINT, _SMUFL_BY_NAME
    if _SMUFL_BY_CODEPOINT:
        return
    if SMUFL_PATH.exists():
        with open(SMUFL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for name, item in data.items():
                cp = item["codepoint"].upper()
                _SMUFL_BY_CODEPOINT[cp] = name
                _SMUFL_BY_NAME[name] = cp


# Common SMuFL names for easy classification
NOTEHEAD_BLACK = {"noteheadBlack", "noteheadDoubleWholeSquare", "noteheadHalf", "noteheadWhole"}

OPUS_MAPPING: Dict[str, str] = {
    # Sibelius OpusStd / OpusSpecialStd mapping to SMuFL glyph names
    "\u0153": "noteheadBlack",      # œ
    "\u0152": "noteheadHalf",       # Œ
    "w": "noteheadWhole",           # w
    "W": "noteheadDoubleWhole",     # W
    "&": "gClef",                   # &
    "?": "fClef",                   # ?
    "B": "cClef",                   # B
    "8": "gClef8va",
    "#": "accidentalSharp",
    "b": "accidentalFlat",
    "n": "accidentalNatural",
    "\u00d3": "restQuarter",        # Ó
    "\u2030": "rest8th",            # ‰
    "\u00bb": "rest16th",           # »
    "\u00ee": "restHalf",           # î
    "\u00f7": "restWhole",          # ÷
    "j": "flag8thDown",
    "J": "flag8thUp",
    "r": "flag16thDown",
    "R": "flag16thUp",
    "\u02d9": "augmentationDot",    # ˙
    ".": "augmentationDot",
    "4": "timeSig4",
    "3": "timeSig3",
    "2": "timeSig2",
    "6": "timeSig6",
    "8": "timeSig8",
    "c": "timeSigCommon",
    "C": "timeSigCutCommon",
}


def get_glyph_name(char: str, font_name: str = "") -> Optional[str]:
    """Identify music symbol name from character and font."""
    load_smufl()
    cp = f"U+{ord(char):04X}"
    
    # First check standard SMuFL
    if cp in _SMUFL_BY_CODEPOINT:
        return _SMUFL_BY_CODEPOINT[cp]
    
    # Check Opus font if font is Sibelius Opus
    if "Opus" in font_name and char in OPUS_MAPPING:
        return OPUS_MAPPING[char]
    
    return None


def is_notehead(glyph_name: str) -> bool:
    return glyph_name.startswith("notehead")


def is_rest(glyph_name: str) -> bool:
    return glyph_name.startswith("rest")


def is_accidental(glyph_name: str) -> bool:
    return glyph_name.startswith("accidental")


def is_clef(glyph_name: str) -> bool:
    return glyph_name.endswith("Clef") or "Clef" in glyph_name


def is_time_sig(glyph_name: str) -> bool:
    return glyph_name.startswith("timeSig")
