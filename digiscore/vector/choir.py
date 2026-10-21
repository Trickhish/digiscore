"""
Choir mode: keep the sung parts of a score and drop the accompaniment
(organ, piano, strings, continuo...).

A part is sung when its staff label names a voice (Soprano, Alto, T., Bass...),
or, unlabelled, when it carries lyrics. A part labelled with an instrument, or
without any words, is accompaniment.
"""

import re
from typing import List

_VOICE = re.compile(
    r"\b(s|a|t|b|bar|sop|sopr|alt|ten|bas|"
    r"soprano?s?|sopran|alto|alti|altus|contra\s*tenor|tenore?s?|tenori|bass[oiu]?s?|basses|baritone?s?|"
    r"mezzo|cantus|canto|superius|discantus|bassus|quintus|sextus|voice|voix|stimme|chor|choir|"
    r"chœur|choeur|coro|men|women|solo|soli|tutti|ténor|ténors|sopranos|basse|baryton)\b\.?",
    re.I,
)
# A voice pair as OCR may read it: "S/A", "T B", "TIB", "S.A.", "T1B"
_VOICE_PAIR = re.compile(r"^\s*[SATB]\s*[./|Il1&+\-]?\s*[SATB]\.?\s*$")

_INSTRUMENT = re.compile(
    r"(organ|orgel|organo|orgue|piano|pno|klavier|clavier|harpsichord|cembalo|keyboard|"
    r"accomp|harp|harfe|guitar|gitarre|violin|violino|vln|vls|vl\.|viola|vla|cello|violoncell|vc\.|"
    r"contrabass|kontrabass|double\s*bass|bc\b|b\.c\.|continuo|basso\s*continuo|"
    r"flute|flauto|flöte|oboe|clarinet|bassoon|fagott|horn|corno|trumpet|tromba|trompete|"
    r"trombone|posaune|timpani|pauke|percussion|strings|instr)",
    re.I,
)


def part_kind(part: dict) -> str:
    """'voice', 'instrument' or 'unknown' from the labels written before its staves."""
    labels = " ".join(part.get("labels", []))
    if not labels.strip():
        return "unknown"
    if _INSTRUMENT.search(labels):
        return "instrument"
    if _VOICE.search(labels) or any(_VOICE_PAIR.match(x) for x in part.get("labels", [])):
        return "voice"
    return "unknown"


def _lyric_share(part: dict) -> float:
    notes = [n for m in part.get("measures", []) for v in m.get("voices", []) for e in v for n in e
             if not n.get("is_rest") and not n.get("unison_copy")]
    if not notes:
        return 0.0
    return sum(1 for n in notes if n.get("lyrics")) / len(notes)


def sung_parts(parts: List[dict]) -> List[int]:
    """Indexes of the parts to keep in choir mode."""
    # Nothing named as a voice or an instrument, and few words: not a choir
    # score (a piano piece, say) — leave it whole
    kinds = [part_kind(p) for p in parts]
    if all(k == "unknown" for k in kinds) and max((_lyric_share(p) for p in parts), default=0) < 0.2:
        return list(range(len(parts)))
    keep = []
    for k, p in enumerate(parts):
        kind = part_kind(p)
        if kind == "voice":
            keep.append(k)
        elif kind == "instrument":
            continue
        # Unlabelled: sung parts have words; a braced pair without any is a keyboard
        elif _lyric_share(p) >= 0.05:
            keep.append(k)
    # Never return an empty score
    return keep or list(range(len(parts)))


def apply_choir_mode(score: dict) -> dict:
    parts = score.get("parts", [])
    keep = sung_parts(parts)
    if len(keep) == len(parts):
        return score
    kept = [dict(parts[k]) for k in keep]
    for i, p in enumerate(kept):
        p["part_id"] = f"P{i + 1}"
    out = dict(score)
    out["parts"] = kept
    return out
