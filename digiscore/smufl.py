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


# capella 3 music font, by byte code (identified by rendering each code).
# "capella3Regular" PDFs give the byte as Latin-1, "capella3" ones and printer
# driver renamings (TT9Eo00...) as Mac Roman: _capella_byte undoes both.
CAPELLA_MAPPING: Dict[int, str] = {
    0x2E: "augmentationDot",
    0x3A: "timeSigCommon",
    0x3B: "timeSigCutCommon",
    0x41: "gClef",
    0x43: "cClef",
    0x45: "fClef",
    0x49: "restWhole",
    0x4A: "restHalf",
    0x4B: "restQuarter",
    0x4C: "rest8th",
    0x51: "accidentalFlat",
    0x52: "accidentalNatural",
    0x53: "accidentalSharp",
    0x6B: "fermataBelow",
    0x75: "fermataAbove",
    0xE1: "noteheadDoubleWhole",
    0xE3: "noteheadWhole",
    0xE4: "noteheadHalf",
    0xE5: "noteheadBlack",
    0xE6: "flag8thDown",
    0xEA: "flag8thUp",
    0xFE: "gClef8vb",
    **{0x30 + d: f"timeSig{d}" for d in range(10)},
}

# Fonts known to be capella / Sonata-layout under another name (set per document)
_capella_alias: Dict[str, str] = {}
_sonata_alias: set = set()


def _capella_byte(char: str, mac: bool) -> int:
    cp = ord(char)
    if cp < 0x80:
        return cp
    if mac:
        try:
            return char.encode("mac_roman")[0]
        except UnicodeEncodeError:
            return -1
    return cp if cp < 0x100 else -1


# LilyPond Feta/Emmentaler glyph names -> SMuFL names
FETA_NAMES: Dict[str, str] = {
    "noteheads.s0": "noteheadWhole", "noteheads.s1": "noteheadHalf", "noteheads.s2": "noteheadBlack",
    "noteheads.sM1": "noteheadDoubleWhole", "noteheads.uM2": "noteheadDoubleWhole",
    "noteheads.dM2": "noteheadDoubleWhole",
    "noteheads.s0neomensural": "noteheadWhole", "noteheads.s1neomensural": "noteheadHalf",
    "noteheads.s2neomensural": "noteheadBlack", "noteheads.sM1neomensural": "noteheadDoubleWhole",
    "noteheads.srM1neomensural": "noteheadDoubleWhole",
    "dots.dot": "augmentationDot",
    "accidentals.sharp": "accidentalSharp", "accidentals.flat": "accidentalFlat",
    "accidentals.natural": "accidentalNatural", "accidentals.doublesharp": "accidentalDoubleSharp",
    "accidentals.flatflat": "accidentalDoubleFlat",
    "rests.0": "restWhole", "rests.1": "restHalf", "rests.2": "restQuarter", "rests.3": "rest8th",
    "rests.4": "rest16th", "rests.M1": "restDoubleWhole", "rests.0o": "restWhole", "rests.1o": "restHalf",
    "rests.0neomensural": "restWhole", "rests.1neomensural": "restHalf",
    "rests.M1neomensural": "restDoubleWhole",
    "clefs.G": "gClef", "clefs.F": "fClef", "clefs.C": "cClef",
    "clefs.G_change": "gClef", "clefs.F_change": "fClef", "clefs.C_change": "cClef",
    "clefs.petrucci.g": "gClef", "clefs.petrucci.c3": "cClef", "clefs.petrucci.c4": "cClef",
    "clefs.mensural.f": "fClef",
    "flags.u3": "flag8thUp", "flags.d3": "flag8thDown", "flags.u4": "flag16thUp", "flags.d4": "flag16thDown",
    "timesig.C44": "timeSigCommon", "timesig.C22": "timeSigCutCommon",
    "zero": "timeSig0", "one": "timeSig1", "two": "timeSig2", "three": "timeSig3", "four": "timeSig4",
    "five": "timeSig5", "six": "timeSig6", "seven": "timeSig7", "eight": "timeSig8", "nine": "timeSig9",
}

# font name -> {char: glyph name}, from the embedded font's encoding (set per document)
_glyph_names: Dict[str, Dict[str, str]] = {}


def detect_named_fonts(doc) -> None:
    """
    LilyPond embeds subsets of its Emmentaler font whose character codes
    differ from file to file, but which keep the glyph names: read them.
    """
    import io

    _glyph_names.clear()
    for page in doc:
        for f in page.get_fonts():
            base = _base_font(f[3])
            if not base.startswith(("Emmentaler", "Feta", "feta")) or base in _glyph_names:
                continue
            try:
                from fontTools.cffLib import CFFFontSet
                _, ext, _, buf = doc.extract_font(f[0])
                if ext != "cff":
                    continue
                cs = CFFFontSet()
                cs.decompile(io.BytesIO(buf), None)
                fnt = cs[cs.fontNames[0]]
                # Keyed by glyph id: the text layer's character codes are not
                # reliable for these subsets, the glyph ids (see glyph_trace) are
                _glyph_names[base] = {i: n for i, n in enumerate(fnt.charset) if n != ".notdef"}
            except Exception:
                continue


def glyph_trace(page) -> list:
    """
    Glyphs of the fonts known by glyph name (LilyPond), from the page's text
    trace: [{name, char, font, x, y, bbox}].
    """
    out = []
    if not _glyph_names:
        return out
    for t in page.get_texttrace():
        base = _base_font(t["font"])
        names = _glyph_names.get(base)
        if names is None:
            continue
        for ch in t["chars"]:
            name = FETA_NAMES.get(names.get(ch[1], ""))
            if name:
                out.append({"name": name, "char": chr(ch[0]), "font": t["font"],
                            "x": ch[2][0], "y": ch[2][1], "bbox": tuple(ch[3])})
    return out


def _base_font(font_name: str) -> str:
    # Strip subset prefixes such as "ABCDEF+"
    return font_name.split("+", 1)[-1]


# Whether a font named "MScore" uses MuseScore 2's private codes (True) or
# SMuFL ones (MuseScore 3+ embeds Emmentaler-SMuFL under the same name).
# Set per document by detect_mscore_flavour.
_legacy_mscore: Dict[str, bool] = {}


def detect_capella_aliases(doc) -> None:
    """
    Printer drivers rename embedded fonts (TT9Eo00...). A renamed font whose
    most used characters are capella's noteheads (Mac Roman Â = 0xE5 black,
    ‰ = 0xE4 half) on staff lines is taken as capella.
    """
    import collections

    _capella_alias.clear()
    _sonata_alias.clear()
    use: Dict[str, collections.Counter] = {}
    for page in doc:
        for block in page.get_text("rawdict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    base = _base_font(span.get("font", ""))
                    if not base.startswith("TT"):
                        continue
                    c = use.setdefault(base, collections.Counter())
                    c.update(ch["c"] for ch in span.get("chars", []))
    for base, c in use.items():
        total = sum(v for k, v in c.items() if not k.isspace())
        if total < 10:
            continue
        capella = c.get("\u00c2", 0) + c.get("\u2030", 0) + c.get("\u00c1", 0)
        # Sonata layout (Maestro, Opus... renamed): œ black head, ˙ half, w whole
        sonata = c.get("\u0153", 0) + c.get("\u02d9", 0) + c.get("w", 0) + c.get("\uf0cf", 0) + c.get("\uf0fa", 0)
        if capella >= 0.25 * total and capella > sonata:
            _capella_alias[base] = "capella3"
        elif sonata >= 0.25 * total:
            _sonata_alias.add(base)


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
    if base.startswith(SONATA_LAYOUT_FONTS) or base in _sonata_alias:
        return OPUS_MAPPING.get(_symbol_char(char))
    if base in _glyph_names:
        return None  # read through glyph_trace instead
    if base.startswith("capella") or base in _capella_alias:
        if "Text" in base:
            return None
        mac = not base.startswith("capella3Regular")
        return CAPELLA_MAPPING.get(_capella_byte(char, mac))
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
