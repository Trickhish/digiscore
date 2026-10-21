"""
Post-processing of OMR MusicXML output.

NoteLite/Audiveris sometimes splits one instrument into several parts, typically
when the first system shows full names ("Soprano") and later systems show
abbreviations ("S."). Each such part only has notes in some measures and rests
(or nothing) elsewhere. These parts are merged back measure by measure.
"""

import copy
import re
import unicodedata
from typing import Dict, List, Optional

from lxml import etree


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name or "")
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _is_abbreviation(short: str, full: str) -> bool:
    s, f = _norm(short), _norm(full)
    if not s or not f or s == f:
        return False
    return f.startswith(s)


def _measure_has_music(measure) -> bool:
    for note in measure.findall("note"):
        if note.find("rest") is None:
            return True
        rest = note.find("rest")
        if rest is not None and rest.get("measure") != "yes" and note.get("print-object") != "no":
            # A real rest still counts as content only if other content exists
            continue
    return False


def _measure_is_empty(measure) -> bool:
    """No notes at all, or only rests (whole-measure rests in an unused part)."""
    return not _measure_has_music(measure)


def merge_split_parts(xml_content: str) -> str:
    root = etree.fromstring(xml_content.encode("utf-8"))
    part_list = root.find("part-list")
    if part_list is None:
        return xml_content
    score_parts = part_list.findall("score-part")
    if len(score_parts) < 2:
        return xml_content

    names = {sp.get("id"): (sp.findtext("part-name") or "").strip() for sp in score_parts}
    abbrs = {sp.get("id"): (sp.findtext("part-abbreviation") or "").strip() for sp in score_parts}
    parts = {p.get("id"): p for p in root.findall("part")}

    # Pair each abbreviated part with the full-name part it abbreviates
    ids = [sp.get("id") for sp in score_parts]
    target_of: Dict[str, str] = {}
    for a in ids:
        for b in ids:
            if a == b or b in target_of:
                continue
            if _is_abbreviation(names[a], names[b]) or (
                names[a] and names[a] == abbrs.get(b) and names[b] != names[a]
            ):
                target_of[a] = b
                break

    # Unnamed parts: match by complementary measure usage, in order
    if not target_of:
        target_of = _match_by_usage(ids, parts)
        # Keep the better-named part of each pair (a real name over a one-letter
        # placeholder such as "V")
        def quality(pid):
            return len(_norm(names.get(pid, "")))
        target_of = {
            (d if quality(s) > quality(d) else s): (s if quality(s) > quality(d) else d)
            for s, d in target_of.items()
        }

    if not target_of:
        return xml_content

    for src_id, dst_id in target_of.items():
        src, dst = parts.get(src_id), parts.get(dst_id)
        if src is None or dst is None:
            continue
        src_measures = src.findall("measure")
        dst_measures = dst.findall("measure")
        for sm, dm in zip(src_measures, dst_measures):
            if _measure_has_music(sm) and _measure_is_empty(dm):
                _replace_measure_content(dm, sm)
        root.remove(src)
        for sp in score_parts:
            if sp.get("id") == src_id:
                part_list.remove(sp)

    return etree.tostring(
        root.getroottree(), pretty_print=True, xml_declaration=True, encoding="UTF-8"
    ).decode("utf-8")


def _replace_measure_content(dst_measure, src_measure) -> None:
    """Take notes, backups, forwards and directions from src; keep dst attributes."""
    keep = [c for c in dst_measure if c.tag in ("attributes", "print", "barline")]
    for c in list(dst_measure):
        dst_measure.remove(c)
    for c in keep:
        if c.tag != "barline":
            dst_measure.append(c)
    for c in src_measure:
        if c.tag in ("attributes", "print"):
            # Attributes inside the source (e.g. clef) are only needed if dst had none
            if c.tag == "attributes" and not any(k.tag == "attributes" for k in keep):
                dst_measure.append(copy.deepcopy(c))
            continue
        dst_measure.append(copy.deepcopy(c))
    for c in keep:
        if c.tag == "barline":
            dst_measure.append(c)


def _match_by_usage(ids: List[str], parts) -> Dict[str, str]:
    """
    Parts with mutually exclusive measure usage are the same instrument split in two.
    Pair them in order: the k-th part of the later group with the k-th of the earlier.
    """
    usage = {}
    for pid in ids:
        p = parts.get(pid)
        if p is None:
            return {}
        usage[pid] = [_measure_has_music(m) for m in p.findall("measure")]
    n = len(ids)
    if n % 2 == 0:
        half = n // 2
        first, second = ids[:half], ids[half:]

        def overlap(a, b):
            return sum(1 for x, y in zip(usage[a], usage[b]) if x and y)

        # Exclusive usage, allowing a stray bar or two (a note the engine also
        # put in the other part's empty bar, e.g. the whole note we restored)
        n_meas = max((len(u) for u in usage.values()), default=0)
        if all(overlap(a, b) <= (1 if n_meas >= 12 else 0) for a, b in zip(first, second)) and \
                all(sum(usage[a]) and sum(usage[b]) for a, b in zip(first, second)):
            return {b: a for a, b in zip(first, second)}

    # Sparse parts (music in a few bars only) stand in for main parts in some
    # systems: pair each, in order, with the next main part it never overlaps
    n_meas = max((len(u) for u in usage.values()), default=0)
    sparse = [p for p in ids if sum(usage[p]) <= 0.3 * n_meas]
    main = [p for p in ids if p not in sparse]
    if not sparse or not main:
        return {}
    out: Dict[str, str] = {}
    j = 0
    for s in sparse:
        while j < len(main) and any(x and y for x, y in zip(usage[s], usage[main[j]])):
            j += 1
        if j >= len(main):
            return {}
        out[s] = main[j]
        j += 1
    return out


def fix_measure_rests(xml_content: str) -> str:
    """
    Whole-bar rests and empty bars get the bar's full length: the OMR engine
    writes measure rests with a zero duration, which is not valid MusicXML.
    """
    from fractions import Fraction

    root = etree.fromstring(xml_content.encode("utf-8"))
    changed = False
    for part in root.findall("part"):
        div, beats, beat_type = 1, 4, 4
        for m in part.findall("measure"):
            a = m.find("attributes")
            if a is not None:
                if a.findtext("divisions"):
                    div = int(a.findtext("divisions"))
                if a.find("time") is not None and a.findtext("time/beats", "").isdigit():
                    beats, beat_type = int(a.findtext("time/beats")), int(a.findtext("time/beat-type"))
            bar = Fraction(4 * beats, beat_type) * div
            if bar.denominator != 1:
                continue
            notes = m.findall("note")
            for n in notes:
                d = n.find("duration")
                if d is not None and (d.text or "0").strip() in ("0", ""):
                    r = n.find("rest")
                    if r is not None and n.find("grace") is None:
                        d.text = str(int(bar))
                        r.set("measure", "yes")
                        changed = True
            if not notes and m.get("implicit") != "yes":
                rest = etree.Element("note")
                etree.SubElement(rest, "rest", measure="yes")
                etree.SubElement(rest, "duration").text = str(int(bar))
                etree.SubElement(rest, "voice").text = "1"
                bl = m.find("barline")
                if bl is not None and bl.get("location") != "left":
                    bl.addprevious(rest)
                else:
                    m.append(rest)
                changed = True
    if not changed:
        return xml_content
    return etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True,
                          encoding="UTF-8").decode("utf-8")


# Chord symbols in English, German and solfège names: C, F#m7, Bb/D, Sib, rém,
# Solm, LA4, DO7...
_CHORD = re.compile(
    r"^(?:[A-H]|do|ré|re|mi|fa|sol|la|si)(?:[#♯b♭]|is|es|s)?"
    r"(?:m|min|maj|M|dim|aug|sus|°|ø|\+)?\d{0,2}(?:/[A-H](?:[#♯b♭])?)?$",
    re.I,
)


# Chord names no verse would hold: solfège in capitals (DO, LA4, SOL7), or with
# a suffix (Sib, Solm, rém, Mi7)
_SURE_CHORD = re.compile(
    r"^(?:(?:DO|RÉ|RE|MI|FA|SOL|LA|SI)(?:[#b]|m)?\d{0,2}"
    r"|(?:[Dd]o|[Rr][ée]|[Mm]i|[Ff]a|[Ss]ol|[Ll]a|[Ss]i)(?:[#b]m?|m)\d{0,2}"
    r"|(?:Do|Ré|Re|Mi|Fa|Sol|La|Si)\d{1,2})$")


def _is_chord_symbol(text: str) -> bool:
    t = text.strip().rstrip(",.;")
    return bool(t) and bool(_CHORD.match(t)) and not t.islower() or bool(
        t and re.match(r"^(?:do|ré|re|mi|fa|sol|la|si)[#b]?m?\d?$", t) and t[0].isupper())


def drop_chord_symbol_lyrics(xml_content: str) -> str:
    """
    Chord names printed above the music (Sib, rém, LA4, F#m...) are read as a
    verse by the OMR: drop a lyric line whose syllables are mostly chord names,
    chord names mixed into a verse, and sparse lines of stray page text.
    """
    try:
        root = etree.fromstring(xml_content.encode("utf-8"))
    except etree.XMLSyntaxError:
        return xml_content
    changed = False
    for part in root.findall("part"):
        by_line: Dict[str, List] = {}
        for lyr in part.iter("lyric"):
            by_line.setdefault(lyr.get("number", "1"), []).append(lyr)
        for num, lyrs in by_line.items():
            texts = [(l.findtext("text") or "").strip() for l in lyrs]
            texts = [t for t in texts if t]
            if not texts:
                continue
            chords = [l for l in lyrs if _is_chord_symbol(l.findtext("text") or "")]
            # A whole line of chord names goes; elsewhere single chord names
            # written in capitals (DO, LA4, Sib) mixed into a verse go too
            if len(chords) >= 0.6 * len(texts):
                drop = lyrs
            else:
                drop = [l for l in lyrs if _SURE_CHORD.match((l.findtext("text") or "").strip().rstrip(","))]
            for l in drop:
                l.getparent().remove(l)
                changed = True
    # Stray page text read as words: a verse line touching only a few of a
    # part's notes, in a score where some part has a real verse
    coverage = {}
    for part in root.findall("part"):
        pitched = [n for n in part.iter("note") if n.find("rest") is None and n.find("chord") is None]
        lines: Dict[str, List] = {}
        for n in pitched:
            for lyr in n.findall("lyric"):
                lines.setdefault(lyr.get("number", "1"), []).append(lyr)
        coverage[part.get("id")] = (len(pitched), lines)
    best = max((len(l) / max(1, n) for n, lines in coverage.values() for l in lines.values()), default=0)
    if best >= 0.15:
        for part in root.findall("part"):
            n, lines = coverage[part.get("id")]
            for num, lyrs in lines.items():
                texts = [(l.findtext("text") or "").strip() for l in lyrs]
                junk = sum(1 for t in texts if re.search(r"\d", t) or _is_chord_symbol(t))
                # Words of running prose (capitalised, whole words with no
                # hyphenation) rather than sung syllables
                syllabics = [l.findtext("syllabic") for l in lyrs]
                prose = sum(1 for sy in syllabics if sy in (None, "single")) >= 0.9 * len(lyrs)
                if junk >= 0.3 * len(lyrs) or (len(lyrs) < 0.12 * n and prose and len(lyrs) <= 12):
                    for l in lyrs:
                        l.getparent().remove(l)
                        changed = True
    if not changed:
        return xml_content
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          doctype=root.getroottree().docinfo.doctype or None).decode("utf-8")


def dedupe_ties(xml_content: str) -> str:
    """
    The engine sometimes writes the same tie several times on one note
    (a <tie type="start"/> per tied voice), which the schema rejects: keep one
    of each type, stop before start.
    """
    try:
        root = etree.fromstring(xml_content.encode("utf-8"))
    except etree.XMLSyntaxError:
        return xml_content
    changed = False
    for note in root.iter("note"):
        ties = note.findall("tie")
        if len(ties) > 1:
            kinds = []
            for t in ties:
                if t.get("type") in kinds or len(kinds) >= 2:
                    note.remove(t)
                    changed = True
                else:
                    kinds.append(t.get("type"))
            kept = note.findall("tie")
            if len(kept) == 2 and kept[0].get("type") == "start":
                kept[0].addprevious(kept[1])
                changed = True
        for notations in note.findall("notations"):
            seen = set()
            for t in notations.findall("tied"):
                key = (t.get("type"), t.get("number"))
                if key in seen:
                    notations.remove(t)
                    changed = True
                seen.add(key)
    if not changed:
        return xml_content
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          doctype=root.getroottree().docinfo.doctype or None).decode("utf-8")


def drop_prose_directions(xml_content: str) -> str:
    """
    Text below the music (the verses of a psalm, notes, credits) is read as a
    direction on the last staff: a direction of many words is prose, not a
    tempo or expression mark.
    """
    try:
        root = etree.fromstring(xml_content.encode("utf-8"))
    except etree.XMLSyntaxError:
        return xml_content
    changed = False
    for d in list(root.iter("direction")):
        words = " ".join(w.text or "" for w in d.iter("words"))
        n_words = len(words.split())
        # Long, or several words below the staff (expression marks are short)
        below = d.get("placement") == "below"
        if d.find(".//metronome") is None and (n_words > 6 or (below and n_words >= 3)):
            d.getparent().remove(d)
            changed = True
    if not changed:
        return xml_content
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          doctype=root.getroottree().docinfo.doctype or None).decode("utf-8")
