"""
Split voices for OMR output: a staff shared by two voices (SA, TB) becomes two
parts, working on the MusicXML so that ties, beams, tuplets, barlines and
directions stay as the engine wrote them.

Per bar: with two voices, voice 1 goes to the upper part and voice 2 to the
lower one; with one, a chord gives its top note to the upper part and its bottom
note to the lower one, and a single note is a unison for both. Gaps left in a
part are filled with rests. Words printed once for the staff are copied to the
lower part where it has none of its own.
"""

import copy
import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from lxml import etree

from digiscore.vector.choir import _INSTRUMENT, _VOICE, _VOICE_PAIR

_PAIR_NAMES = {"S": "Soprano", "A": "Alto", "T": "Tenor", "B": "Bass"}
_TYPES = [(4, "whole"), (2, "half"), (1, "quarter"), (Fraction(1, 2), "eighth"),
          (Fraction(1, 4), "16th"), (Fraction(1, 8), "32nd")]
_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _midi(note) -> Optional[float]:
    p = note.find("pitch")
    if p is None:
        return None
    alter = float(p.findtext("alter") or 0)
    return 12 * (int(p.findtext("octave") or 4) + 1) + _STEPS.get(p.findtext("step") or "C", 0) + alter


def _dur(el) -> int:
    try:
        return int(el.findtext("duration") or 0)
    except ValueError:
        return 0


def _voice(note) -> str:
    return note.findtext("voice") or "1"


def _names(name: str, clef: str) -> Tuple[str, str]:
    letters = [c for c in re.sub(r"[^SATB]", " ", name.upper()).split() if c in _PAIR_NAMES]
    if _VOICE_PAIR.match(name or "") and len(letters) >= 2:
        return _PAIR_NAMES[letters[0]], _PAIR_NAMES[letters[-1]]
    found = []
    for w in re.findall(r"[^\W\d_]+", (name or "").lower()):
        for key, full in (("sop", "Soprano"), ("alt", "Alto"), ("ten", "Tenor"), ("ténor", "Tenor"), ("bas", "Bass")):
            if w.startswith(key) and full not in found:
                found.append(full)
    if len(found) >= 2:
        return found[0], found[-1]
    return ("Tenor", "Bass") if clef == "F" else ("Soprano", "Alto")


def _staff_clef(part) -> str:
    c = part.find("measure/attributes/clef")
    if c is None:
        return "G"
    sign = c.findtext("sign") or "G"
    # A treble clef an octave down is a tenor staff: name it with the low voices
    if sign == "G" and (c.findtext("clef-octave-change") or "0").strip() == "-1":
        return "F"
    return sign


def _events(measure) -> Tuple[List[Tuple[int, object]], int]:
    """Each note with its onset in divisions, and the bar's length."""
    out, cursor, end, last_onset = [], 0, 0, 0
    for el in measure:
        if el.tag == "note":
            if el.find("chord") is not None:
                onset = last_onset
            else:
                onset = cursor
                if el.find("grace") is None:
                    cursor += _dur(el)
            last_onset = onset
            out.append((onset, el))
        elif el.tag == "backup":
            cursor -= _dur(el)
        elif el.tag == "forward":
            cursor += _dur(el)
        end = max(end, cursor)
    return out, end


def _two_voice_part(part, name: str, threshold: float = 0.7) -> bool:
    bars = two = 0
    for m in part.findall("measure"):
        ev, _ = _events(m)
        voices = {_voice(n) for _, n in ev if n.find("rest") is None}
        if voices:
            bars += 1
            # The engine often reads two voices as one voice of chords
            chords = sum(1 for _, n in ev if n.find("chord") is not None and n.find("rest") is None)
            heads = sum(1 for _, n in ev if n.find("rest") is None)
            two += len(voices) >= 2 or chords >= max(1, heads // 3)
    if not bars:
        return False
    if name and (_VOICE_PAIR.match(name) or len(re.findall(r"\b(s|a|t|b|sop\w*|alt\w*|ten\w*|bass\w*)\b", name, re.I)) >= 2):
        return True
    # A single voice name is not trusted: OCR keeps one line of a two-line
    # label ("Soprano / Alto" read as "Alto"); the writing decides
    return two >= threshold * bars


def _rest(dur: int, divisions: int, whole_bar: bool = False):
    n = etree.Element("note")
    r = etree.SubElement(n, "rest")
    if whole_bar:
        r.set("measure", "yes")
    etree.SubElement(n, "duration").text = str(dur)
    etree.SubElement(n, "voice").text = "1"
    if not whole_bar:
        q = Fraction(dur, divisions)
        t = next((name for v, name in _TYPES if q >= v), "32nd")
        etree.SubElement(n, "type").text = t
    return n


def _fill(dur: int, divisions: int) -> List:
    out = []
    for v, _ in _TYPES:
        step = int(v * divisions)
        while step and dur >= step and step == v * divisions:
            out.append(_rest(step, divisions))
            dur -= step
    if dur > 0:
        f = etree.Element("forward")
        etree.SubElement(f, "duration").text = str(dur)
        out.append(f)
    return out


def _clean(note, keep_chord=False):
    n = copy.deepcopy(note)
    for tag in ("stem",):
        for x in n.findall(tag):
            n.remove(x)
    if not keep_chord:
        for x in n.findall("chord"):
            n.remove(x)
    v = n.find("voice")
    if v is not None:
        v.text = "1"
    return n


def _pick(chord: List, top: bool):
    pitched = [n for n in chord if _midi(n) is not None]
    if not pitched:
        return chord[0]
    return max(pitched, key=_midi) if top else min(pitched, key=_midi)


def _with_lyrics_of(note, src):
    """note takes the lyrics of src (the first note of a chord may hold them)."""
    for x in note.findall("lyric"):
        note.remove(x)
    anchor = note.find("notations")
    for lyr in src.findall("lyric"):
        new = copy.deepcopy(lyr)
        # lyric comes after notations / before play
        idx = list(note).index(anchor) + 1 if anchor is not None else len(note)
        for after in ("play", "listen"):
            a = note.find(after)
            if a is not None:
                idx = min(idx, list(note).index(a))
        note.insert(idx, new)
        anchor = new


def _split_measure(measure, divisions: int) -> Tuple[list, list, int]:
    ev, length = _events(measure)
    # group chords: (onset, voice, [notes])
    groups: List[Tuple[int, str, List]] = []
    for onset, n in ev:
        if n.find("chord") is not None and groups:
            groups[-1][2].append(n)
        else:
            groups.append((onset, _voice(n), [n]))
    sounding = {v for _, v, g in groups if any(x.find("rest") is None for x in g)}
    order = sorted({v for _, v, _ in groups}, key=lambda v: int(v) if v.isdigit() else 99)
    up: List[Tuple[int, object]] = []
    low: List[Tuple[int, object]] = []
    if len(sounding) >= 2:
        first = order[0]
        up_pitch = {onset: {_midi(x) for x in g} for onset, v, g in groups if v == first}
        for onset, v, g in groups:
            dest = up if v == first else low
            # A lower-voice chord repeating the upper voice's note there: the
            # engine merged the two heads; the lower voice keeps its own
            if v != first and len(g) > 1:
                own = [x for x in g if _midi(x) not in up_pitch.get(onset, set())]
                g = own or g
            for i, x in enumerate(g):
                dest.append((onset, _clean(x, keep_chord=i > 0)))
    else:
        main = next(iter(sounding)) if sounding else (order[0] if order else "1")
        for onset, v, g in groups:
            if v != main:
                continue
            u, lo = _pick(g, True), _pick(g, False)
            un, ln = _clean(u), _clean(lo)
            if len(g) > 1:
                _with_lyrics_of(un, g[0] if g[0].find("lyric") is not None else u)
                for x in ln.findall("lyric"):
                    ln.remove(x)
            up.append((onset, un))
            low.append((onset, ln))
    # Words under a shared staff belong to the staff, whichever voice the
    # engine hung them on: by onset, onto the upper part's note there
    words: Dict[int, object] = {}
    for onset, _, g in groups:
        src = next((x for x in g if x.find("lyric") is not None and x.find("rest") is None), None)
        if src is not None and onset not in words:
            words[onset] = src
    for dest in (up, low):
        for _, n in dest:
            for x in n.findall("lyric"):
                n.remove(x)
    for onset, n in up:
        if onset in words and n.find("rest") is None and n.find("chord") is None:
            _with_lyrics_of(n, words.pop(onset))
    return up, low, length


def _assemble(measure, notes: List[Tuple[int, object]], length: int, divisions: int, keep_dirs: bool):
    """A new measure: the bar's attributes and barlines, the notes in time order, gaps as rests."""
    m = etree.Element("measure", measure.attrib)
    lead = [el for el in measure if el.tag in ("print", "attributes")]
    for el in lead:
        m.append(copy.deepcopy(el))
    for el in measure:
        if el.tag == "barline" and el.get("location") == "left":
            m.append(copy.deepcopy(el))
    # directions in time order with the notes
    dirs = []
    if keep_dirs:
        cursor = 0
        for el in measure:
            if el.tag == "note" and el.find("chord") is None and el.find("grace") is None:
                cursor += _dur(el)
            elif el.tag == "backup":
                cursor -= _dur(el)
            elif el.tag == "forward":
                cursor += _dur(el)
            elif el.tag in ("direction", "harmony", "sound"):
                dirs.append((cursor, copy.deepcopy(el)))
    notes = sorted(notes, key=lambda t: t[0])
    cursor, di = 0, 0
    if not any(n.find("rest") is None for _, n in notes) and not notes:
        m.append(_rest(length, divisions, whole_bar=True))
        cursor = length
    for onset, n in notes:
        while di < len(dirs) and dirs[di][0] <= onset:
            m.append(dirs[di][1]); di += 1
        if n.find("chord") is None:
            if onset > cursor:
                for r in _fill(onset - cursor, divisions):
                    m.append(r)
                cursor = onset
            elif onset < cursor:
                continue  # overlapping note of the same voice: keep the first
            if n.find("grace") is None:
                cursor += _dur(n)
        m.append(n)
    while di < len(dirs):
        m.append(dirs[di][1]); di += 1
    if length > cursor and notes:
        for r in _fill(length - cursor, divisions):
            m.append(r)
    elif not notes:
        m.append(_rest(length, divisions, whole_bar=True))
    for el in measure:
        if el.tag == "barline" and el.get("location") != "left":
            m.append(copy.deepcopy(el))
    return m


def _onset_lyrics(part) -> Dict[Tuple[str, int], object]:
    out = {}
    for m in part.findall("measure"):
        ev, _ = _events(m)
        for onset, n in ev:
            if n.find("lyric") is not None and n.find("rest") is None:
                out.setdefault((m.get("number"), onset), n)
    return out


def _copy_shared_words(upper, lower) -> None:
    pitched = [n for n in lower.iter("note") if n.find("rest") is None]
    if not pitched or sum(1 for n in pitched if n.find("lyric") is not None) > 0.2 * len(pitched):
        return
    words = _onset_lyrics(upper)
    for m in lower.findall("measure"):
        ev, _ = _events(m)
        for onset, n in ev:
            if n.find("rest") is None and n.find("chord") is None and n.find("lyric") is None:
                src = words.get((m.get("number"), onset))
                if src is not None:
                    _with_lyrics_of(n, src)


def split_voices_xml(xml: str) -> str:
    try:
        root = etree.fromstring(xml.encode("utf-8"))
    except etree.XMLSyntaxError:
        return xml
    plist = root.find("part-list")
    if plist is None:
        return xml
    changed = False
    prev_upper = None
    # In a two-staff choir score both staves are voice pairs or neither is:
    # once one qualifies, its sibling needs less evidence (OMR merges unisons)
    all_parts = root.findall("part")
    two_staff = len(all_parts) == 2 and any(
        _two_voice_part(p, "") for p in all_parts)
    for sp in list(plist.findall("score-part")):
        pid = sp.get("id")
        part = next((p for p in root.findall("part") if p.get("id") == pid), None)
        if part is None:
            continue
        texts = [t.strip() for t in sp.xpath("part-name/text()|part-abbreviation/text()") if t.strip()]
        name = next((t for t in texts if _VOICE_PAIR.match(t)), texts[0] if texts else "")
        if name and _INSTRUMENT.search(name) and not _VOICE_PAIR.match(name):
            continue
        pitched = [n for n in part.iter("note") if n.find("rest") is None]
        if name == "" and pitched and not any(n.find("lyric") is not None for n in pitched):
            continue  # unlabelled and wordless: accompaniment, not a voice pair
        if not _two_voice_part(part, name, 0.4 if two_staff else 0.7):
            continue
        clef = _staff_clef(part)
        up_name, low_name = _names(name, clef)

        new_id = f"{pid}b"
        upper = etree.Element("part", id=pid)
        lower = etree.Element("part", id=new_id)
        divisions = 1
        for m in part.findall("measure"):
            d = m.findtext("attributes/divisions")
            if d:
                divisions = int(d)
            up, low, length = _split_measure(m, divisions)
            upper.append(_assemble(m, up, length, divisions, keep_dirs=True))
            lower.append(_assemble(m, low, length, divisions, keep_dirs=False))
        _copy_shared_words(upper, lower)
        # Two-staff choir layout: the words between the staves are sung by the
        # lower staff's voices too when it has none of its own
        if clef == "F" and prev_upper is not None:
            _copy_shared_words(prev_upper, upper)
            _copy_shared_words(prev_upper, lower)
        prev_upper = upper
        for x in lower.iter("instrument"):
            x.set("id", x.get("id", "").replace(pid, new_id, 1))
        part.getparent().replace(part, upper)
        upper.addnext(lower)

        sp2 = copy.deepcopy(sp)
        sp2.set("id", new_id)
        for el in sp2.iter():
            if el.get("id") and el is not sp2:
                el.set("id", el.get("id").replace(pid, new_id, 1))
        for tag, text in ((sp, up_name), (sp2, low_name)):
            for t in ("part-name", "part-abbreviation"):
                e = tag.find(t)
                if e is not None:
                    e.text = text if t == "part-name" else text[0] + "."
        sp.addnext(sp2)
        changed = True
    if not changed:
        return xml
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          doctype=root.getroottree().docinfo.doctype or None).decode("utf-8")
