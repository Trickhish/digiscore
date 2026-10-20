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
        if not any(any(x and y for x, y in zip(usage[a], usage[b])) for a, b in zip(first, second)):
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
