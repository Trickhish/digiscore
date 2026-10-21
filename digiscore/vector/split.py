"""
Split the shared staves of a choir score (SA / TB on two staves) into one part
per voice: Soprano, Alto, Tenor, Bass.

On a staff holding two voices, the stem-up voice goes to the upper part and the
stem-down voice to the lower one. Where the staff has a single voice, a chord
gives its top note to the upper part and its bottom note to the lower one; a
single note is a unison sung by both. Words written once for the staff are
copied to the lower part's notes that start with the upper part's.
"""

import copy
import re
from typing import List, Optional, Tuple

from digiscore.vector.choir import part_kind, _lyric_share

_NAMES = {
    "s": "Soprano", "sop": "Soprano", "sopr": "Soprano", "soprano": "Soprano", "sopran": "Soprano",
    "sopranos": "Soprano", "a": "Alto", "alt": "Alto", "alto": "Alto", "alti": "Alto",
    "t": "Tenor", "ten": "Tenor", "tenor": "Tenor", "ténor": "Tenor", "tenore": "Tenor",
    "ténors": "Tenor", "tenors": "Tenor", "b": "Bass", "bas": "Bass", "bass": "Bass",
    "basso": "Bass", "basse": "Bass", "basses": "Bass", "bassus": "Bass", "bar": "Baritone",
    "baritone": "Baritone", "baryton": "Baritone",
}


def _sounding(n: dict) -> bool:
    return not n.get("is_rest") and not n.get("grace")


def _dur(e: List[dict]) -> float:
    return 0.0 if e[0].get("grace") else float(e[0].get("duration_quarters", 1.0))


def _bar_length(m: dict) -> float:
    beats = m.get("time_beats") or 4
    beat_type = m.get("time_beat_type") or 4
    return beats * 4.0 / beat_type


def _voice_names(part: dict, clef: str) -> Tuple[str, str]:
    """Names for the upper and lower voice from the staff label."""
    found = _label_voices(part)
    if len(found) >= 2:
        return found[0], found[-1]
    if clef == "F":
        return "Tenor", "Bass"
    return "Soprano", "Alto"


def _label_voices(part: dict) -> List[str]:
    words = re.findall(r"[^\W\d_]+", " ".join(part.get("labels", [])).lower())
    order = ["Soprano", "Alto", "Tenor", "Baritone", "Bass"]
    return sorted({_NAMES[w] for w in words if w in _NAMES}, key=order.index)


def _is_two_voice(part: dict) -> bool:
    """
    A staff of two voices: its label names two (S A, T/B...), or, unlabelled,
    most of its bars hold two voices. A staff named for one voice keeps its
    divisi as written.
    """
    if any(len([v for v in m.get("voices") or [] if v]) > 2 for m in part.get("measures", [])):
        return False
    named = _label_voices(part)
    if len(named) >= 2:
        return True
    if named:
        return False
    busy = two = 0
    for m in part.get("measures", []):
        voices = [v for v in m.get("voices") or [] if any(_sounding(n) for e in v for n in e)]
        if not voices:
            continue
        busy += 1
        if len(voices) >= 2:
            two += 1
    return busy > 0 and two >= 0.6 * busy


def _rest(q: float, whole_bar: bool = False) -> dict:
    return {"is_rest": True, "measure_rest": whole_bar, "duration_quarters": q, "dot_count": 0,
            "type": "whole" if whole_bar else _rest_type(q), "lyrics": [], "x": 0.0, "y": 0.0}


def _pad(events: List[List[dict]], length: float) -> List[List[dict]]:
    """Fill the end of a bar with rests of plain values."""
    left = length - sum(_dur(e) for e in events)
    if left <= 1e-3:
        return events
    if not any(_dur(e) for e in events):
        return events + [[_rest(length, True)]]
    out = list(events)
    for v in (4.0, 2.0, 1.0, 0.5, 0.25, 0.125):
        while left >= v - 1e-3:
            out.append([_rest(v)])
            left -= v
    return out


def _rest_type(q: float) -> str:
    for t, v in (("whole", 4.0), ("half", 2.0), ("quarter", 1.0), ("eighth", 0.5), ("16th", 0.25)):
        if q >= v - 1e-3:
            return t
    return "32nd"


def _split_event(e: List[dict], upper: bool) -> List[dict]:
    notes = [n for n in e if not n.get("unison_copy")] or e
    if len(notes) < 2 or not _sounding(notes[0]):
        return copy.deepcopy(notes[:1] if upper else [dict(n, lyrics=[]) for n in notes[:1]])
    pick = min(notes, key=lambda n: n.get("y", 0.0)) if upper else max(notes, key=lambda n: n.get("y", 0.0))
    out = copy.deepcopy(pick)
    if not upper:
        out["lyrics"] = []
    # The lyric of a chord sits on one of its notes: the upper voice keeps it
    elif not out.get("lyrics"):
        out["lyrics"] = copy.deepcopy(next((n["lyrics"] for n in notes if n.get("lyrics")), []))
    return [out]


def _split_measure(m: dict) -> Tuple[List[List[dict]], List[List[dict]]]:
    voices = [v for v in m.get("voices") or [] if v]
    if not voices:
        return [], []
    if len(voices) >= 2 and any(_sounding(n) for e in voices[1] for n in e):
        up = [copy.deepcopy(e) for e in voices[0]]
        low = [copy.deepcopy([dict(n, lyrics=n.get("lyrics", []) if not n.get("unison_copy") else [])
                              for n in e]) for e in voices[1]]
    else:
        up = [_split_event(e, True) for e in voices[0]]
        low = [_split_event(e, False) for e in voices[0]]
    # Both parts as long as the staff's longest voice (a pickup stays short)
    length = max(sum(_dur(e) for e in v) for v in voices)
    return _pad(up, length), _pad(low, length)


def _copy_words(up: List[List[dict]], low: List[List[dict]]) -> None:
    """Words printed once for the staff are sung by the lower voice too."""
    if any(n.get("lyrics") for e in low for n in e):
        return
    at = {}
    t = 0.0
    for e in up:
        if _sounding(e[0]) and e[0].get("lyrics"):
            at[round(t, 3)] = e[0]["lyrics"]
        t += _dur(e)
    t = 0.0
    for e in low:
        if _sounding(e[0]):
            w = at.get(round(t, 3))
            if w:
                e[0]["lyrics"] = copy.deepcopy(w)
        t += _dur(e)


def split_part(part: dict) -> Optional[Tuple[dict, dict]]:
    if not _is_two_voice(part):
        return None
    first = next((m for m in part.get("measures", []) if m.get("clef_sign")), {})
    names = _voice_names(part, first.get("clef_sign", "G"))
    upper = {k: v for k, v in part.items() if k != "measures"}
    lower = {k: v for k, v in part.items() if k != "measures"}
    upper["name"], lower["name"] = names
    upper["labels"], lower["labels"] = [names[0]], [names[1]]
    upper["measures"], lower["measures"] = [], []
    shared = _lyric_share(part) > 0 and not any(
        n.get("lyrics") for m in part.get("measures", []) for v in (m.get("voices") or [])[1:]
        for e in v for n in e if _sounding(n))
    for m in part.get("measures", []):
        up, low = _split_measure(m)
        if shared:
            _copy_words(up, low)
        mu = {k: v for k, v in m.items() if k not in ("voices", "notes")}
        ml = dict(mu)
        mu["voices"], ml["voices"] = [up], [low]
        upper["measures"].append(mu)
        lower["measures"].append(ml)
    return upper, lower


def split_voices(score: dict) -> dict:
    """One part per voice for the sung staves that carry two voices."""
    parts = score.get("parts", [])
    out = []
    for p in parts:
        sung = part_kind(p) == "voice" or (part_kind(p) == "unknown" and _lyric_share(p) >= 0.05)
        pair = split_part(p) if sung else None
        out.extend(pair if pair else [dict(p)])
    if len(out) == len(parts):
        return score
    for i, p in enumerate(out):
        p["part_id"] = f"P{i + 1}"
    res = dict(score)
    res["parts"] = out
    return res
