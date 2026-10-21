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
    # Identified by rendering each code from MuseScore 2.x PDFs
    "\ue12d": "noteheadBlack",
    "\ue12c": "noteheadHalf",
    "\ue12b": "noteheadWhole",
    "\ue12a": "noteheadDoubleWhole",
    "\ue127": "augmentationDot",
    "\ue10e": "accidentalSharp",
    "\ue114": "accidentalFlat",
    "\ue113": "accidentalNatural",
    "\ue100": "restWhole",
    "\ue101": "restHalf",
    "\ue107": "restQuarter",
    "\ue109": "rest8th",
    "\ue10a": "rest16th",
    "\ue19e": "gClef",
    "\ue1d7": "gClef8vb",
    "\ue19c": "fClef",
    "\ue19a": "cClef",
    "\ue190": "flag8thUp",
    "\ue191": "flag16thUp",
    "\ue194": "flag8thDown",
    "\ue197": "flag16thDown",
    "\ue09a": "timeSigCommon",
    "\ue09b": "timeSigCutCommon",
    "\ue158": "fermataAbove",
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


# Whether a font named "MScore" uses MuseScore 2's private codes (True) or
# SMuFL ones (MuseScore 3+ embeds Emmentaler-SMuFL under the same name).
# Set per document by detect_mscore_flavour.
_legacy_mscore: Dict[str, bool] = {}


def detect_mscore_flavour(doc) -> None:
    """Look at a PDF's MScore glyphs: SMuFL noteheads (U+E0A2-E0A4) mean SMuFL."""
    _legacy_mscore.clear()
    legacy = smufl = 0
    for page in doc:
        for block in page.get_text("rawdict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if _base_font(span.get("font", "")) not in ("MScoreRegular", "MScore"):
                        continue
                    for ch in span.get("chars", []):
                        cp = ord(ch["c"])
                        if 0xE0A2 <= cp <= 0xE0A4:
                            smufl += 1
                        elif cp in (0xE12B, 0xE12C, 0xE12D):
                            legacy += 1
        if legacy + smufl > 20:
            break
    if legacy or smufl:
        _legacy_mscore["MScore"] = _legacy_mscore["MScoreRegular"] = legacy > smufl


def get_glyph_name(char: str, font_name: str = "") -> Optional[str]:
    """Identify music symbol name from character and font."""
    load_smufl()
    base = _base_font(font_name)

    # Legacy (pre-SMuFL) music fonts first: their codepoints overlap SMuFL ones
    if base.startswith(SONATA_LAYOUT_SPECIAL):
        return OPUS_SPECIAL_MAPPING.get(_symbol_char(char))
    if base.startswith(SONATA_LAYOUT_FONTS):
        return OPUS_MAPPING.get(_symbol_char(char))
    if base in ("MScoreRegular", "MScore") and _legacy_mscore.get(base, base == "MScoreRegular"):
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
