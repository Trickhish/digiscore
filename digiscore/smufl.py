"""
SMuFL (Standard Music Font Layout) and Legacy (Sibelius Opus, etc.)
glyph tables and mapping utilities.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

SMUFL_PATH = Path(__file__).resolve().parent / "smufl_glyphs.json"

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

_TIME_DIGITS = {str(d): f"timeSig{d}" for d in range(10)}

OPUS_MAPPING: Dict[str, str] = {
    # Sibelius OpusStd (Sonata-style layout) mapping to SMuFL glyph names
    "\u0153": "noteheadBlack",      # œ
    "\u02d9": "noteheadHalf",       # ˙
    "w": "noteheadWhole",           # w
    "W": "noteheadDoubleWhole",     # W
    "&": "gClef",                   # &
    "V": "gClef8vb",                # V (tenor treble clef, Maestro / Helsinki)
    "t": "fClef8vb",
    "?": "fClef",                   # ?
    "B": "cClef",                   # B
    "#": "accidentalSharp",
    "b": "accidentalFlat",
    "n": "accidentalNatural",
    "\u2211": "restWhole",          # ∑ (whole-bar rest)
    "\u00d3": "restHalf",           # Ó
    "\u0152": "restQuarter",        # Œ
    "\u2030": "rest8th",            # ‰
    "\u2248": "rest16th",           # ≈
    "J": "flag8thDown",
    "j": "flag8thUp",
    "R": "flag16thDown",
    "r": "flag16thUp",
    "k": "augmentationDot",
    ".": "augmentationDot",         # Finale Maestro, Helsinki
    "\u00ba": "noteheadBlack",      # º (Maestro: small/cue head on some versions)
    "\u00d4": "restHalf",           # Ô
    "\u00ee": "restWhole",          # î
    "\u00a8": "rest8th",            # ¨
    "\u00e4": "rest8th",            # ä
    "\u00c5": "rest16th",           # Å
    "c": "timeSigCommon",
    "C": "timeSigCutCommon",
    **_TIME_DIGITS,
}

OPUS_SPECIAL_MAPPING: Dict[str, str] = {
    # Sibelius OpusSpecialStd
    "\u2122": "augmentationDot",    # ™ (also used in pairs for repeat dots)
    "\u2039": "clef8",              # ‹ (the small 8 under a treble clef)
}

# MuseScore 2.x "MScore" font: private-use codepoints that predate SMuFL
MSCORE2_MAPPING: Dict[str, str] = {
    "\ue12d": "noteheadBlack",
    "\ue12c": "noteheadHalf",
    "\ue12b": "noteheadWhole",
    "\ue127": "augmentationDot",
    "\ue10e": "accidentalSharp",
    "\ue112": "accidentalFlat",
    "\ue111": "accidentalNatural",
    "\ue19e": "gClef",
    "\ue1d7": "gClef8vb",
    "\ue19c": "fClef",
    "\ue190": "flag8thUp",
    "\ue194": "flag8thDown",
    **_TIME_DIGITS,
}


# Fonts laid out like Adobe Sonata: Sibelius Opus and Helsinki, Finale Maestro,
# Petrucci, Engraver, Jazz, Broadway
SONATA_LAYOUT_SPECIAL = ("OpusSpecial", "HelsinkiSpecial", "InkpenSpecial", "ReprisSpecial")
SONATA_LAYOUT_FONTS = ("Opus", "Helsinki", "Inkpen", "Reprise", "Maestro", "Petrucci", "Sonata",
                       "Engraver", "Jazz", "Broadway", "Golden")


def _symbol_char(char: str) -> str:
    """
    Symbol-encoded fonts (older Sibelius and Finale PDFs) put the glyph at
    U+F000 + its Mac Roman code: map back to the character the layout uses.
    """
    cp = ord(char)
    if 0xF020 <= cp <= 0xF0FF:
        return bytes([cp - 0xF000]).decode("mac_roman")
    return char


def _base_font(font_name: str) -> str:
    # Strip subset prefixes such as "ABCDEF+"
    return font_name.split("+", 1)[-1]


def get_glyph_name(char: str, font_name: str = "") -> Optional[str]:
    """Identify music symbol name from character and font."""
    load_smufl()
    base = _base_font(font_name)

    # Legacy (pre-SMuFL) music fonts first: their codepoints overlap SMuFL ones
    if base.startswith(SONATA_LAYOUT_SPECIAL):
        return OPUS_SPECIAL_MAPPING.get(_symbol_char(char))
    if base.startswith(SONATA_LAYOUT_FONTS):
        return OPUS_MAPPING.get(_symbol_char(char))
    if base == "MScoreRegular":
        return MSCORE2_MAPPING.get(char)

    cp = f"U+{ord(char):04X}"
    if cp in _SMUFL_BY_CODEPOINT:
        return _SMUFL_BY_CODEPOINT[cp]
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
