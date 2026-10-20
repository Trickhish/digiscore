"""
Cross-staff rhythm repair for multi-staff OMR output.

In a system, notes that sound together are printed at the same horizontal
position on every staff. When a staff's bar does not add up (a missed rest, a
quarter read as a half...), the bars of the other staves that do add up tell
when each of its notes starts: a notehead at the x of a known onset starts at
that time. With every note's onset known, note durations are cut to fit and the
gaps are filled with rests.
"""

from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from lxml import etree

from digiscore.omr.book import SheetInfo
from digiscore.omr.repair import _running_attributes, _voice_lengths

# quarter-note length -> (type, dots)
_TYPES = {
    Fraction(6): ("whole", 1), Fraction(4): ("whole", 0), Fraction(3): ("half", 1),
    Fraction(2): ("half", 0), Fraction(3, 2): ("quarter", 1), Fraction(1): ("quarter", 0),
    Fraction(3, 4): ("eighth", 1), Fraction(1, 2): ("eighth", 0), Fraction(3, 8): ("16th", 1),
    Fraction(1, 4): ("16th", 0), Fraction(1, 8): ("32nd", 0),
}
_REST_STEPS = [Fraction(4), Fraction(2), Fraction(1), Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)]


def _main_notes(measure):
    return [n for n in measure.findall("note") if n.find("chord") is None and n.find("grace") is None]


def _simple(measure) -> bool:
    """Single voice, no tuplets, no backup/forward: safe to rebuild."""
    if measure.find("backup") is not None or measure.find("forward") is not None:
        return False
    voices = {n.findtext("voice") or "1" for n in measure.findall("note")}
    if len(voices) > 1:
        return False
    return all(n.find("time-modification") is None for n in measure.findall("note"))


def _onsets(measure, division: int) -> List[Tuple[object, Fraction, Fraction]]:
    """(note, onset, length) for main notes in order."""
    t = Fraction(0)
    out = []
    for n in _main_notes(measure):
        d = Fraction(int(n.findtext("duration") or 0), division)
        out.append((n, t, d))
        t += d
    return out


def realign_from_other_staves(xml_content: str, sheets: List[SheetInfo]) -> Tuple[str, int]:
    root = etree.fromstring(xml_content.encode("utf-8"))
    by_id = {p.get("id"): p.findall("measure") for p in root.findall("part")}
    bar_len, divs = _running_attributes(by_id)
    fixed = 0

    for sheet in sheets:
        for system in sheet.systems:
            # Group the staves' measures by stack
            stacks: Dict[int, list] = {}
            for staff in system.staves:
                for sm in staff.measures:
                    stacks.setdefault(sm.measure_index, []).append((staff, sm))

            for m_idx, entries in stacks.items():
                info = []
                for staff, sm in entries:
                    ms = by_id.get(staff.part_id)
                    key = (staff.part_id, m_idx)
                    if not ms or m_idx >= len(ms) or key not in bar_len:
                        continue
                    m = ms[m_idx]
                    if m.get("implicit") == "yes" or not _simple(m):
                        continue
                    lengths = _voice_lengths(m, divs[key])
                    total = max(lengths.values()) if lengths else Fraction(0)
                    info.append((staff, sm, m, key, total == bar_len[key]))

                # Reference points (x, onset) from the bars that add up
                refs: List[Tuple[float, Fraction]] = []
                for staff, sm, m, key, ok in info:
                    if not ok:
                        continue
                    sung = [(n, t) for n, t, _ in _onsets(m, divs[key]) if n.find("rest") is None]
                    chords = sorted(sm.chords, key=lambda c: (c.voice, c.slot))
                    if len(sung) != len(chords):
                        continue
                    refs.extend((c.cx, t) for c, (_, t) in zip(chords, sung))
                if not refs:
                    continue

                for staff, sm, m, key, ok in info:
                    if ok:
                        continue
                    if _rebuild(m, sm, refs, bar_len[key], divs[key], staff.interline):
                        fixed += 1

    out = etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")
    return out, fixed


def _lookup(x: float, refs, tol: float) -> Optional[Fraction]:
    best = min(refs, key=lambda r: abs(r[0] - x))
    if abs(best[0] - x) > tol:
        return None
    # Two references at this x must agree
    near = {t for rx, t in refs if abs(rx - x) <= tol}
    return best[1] if len(near) == 1 else None


def _rebuild(measure, sm, refs, target: Fraction, division: int, interline: float) -> bool:
    events = _onsets(measure, division)
    sung = [(n, t, d) for n, t, d in events if n.find("rest") is None]
    chords = sorted(sm.chords, key=lambda c: (c.voice, c.slot))
    if not sung or len(sung) != len(chords):
        return False

    onsets = []
    for c in chords:
        t = _lookup(c.cx, refs, 0.8 * interline)
        if t is None:
            return False
        onsets.append(t)
    if any(b <= a for a, b in zip(onsets, onsets[1:])) or onsets[-1] >= target:
        return False

    # Rests the OMR did read keep their place: their onsets from the old timing
    # are kept only if they fall in a gap between the realigned notes
    old_rests = [(t, d) for n, t, d in events if n.find("rest") is not None]

    # New event list: each sung note lasts until the next onset (the usual OMR
    # error is a missed dot or a misread value: the note is too short); a gap is
    # a rest only where the OMR saw a rest
    plan: List[Tuple[str, object, Fraction, Fraction]] = []  # (kind, note, start, length)
    if onsets[0] > 0:
        plan.append(("rest", None, Fraction(0), onsets[0]))
    for k, ((n, _, d), t) in enumerate(zip(sung, onsets)):
        nxt = onsets[k + 1] if k + 1 < len(onsets) else target
        gap = nxt - t
        rest_len = Fraction(0)
        if d < gap:
            # A recognised rest that fits right after this note keeps the gap open
            for rt, rd in old_rests:
                if rd <= gap - d and abs((t + d) - rt) <= gap:
                    rest_len = gap - d
                    break
        length = gap - rest_len
        if length not in _TYPES:
            # Prefer the OMR value plus a rest over an unwritable length
            if d in _TYPES and d < gap:
                length, rest_len = d, gap - d
            else:
                return False
        plan.append(("note", n, t, length))
        if rest_len:
            plan.append(("rest", None, t + length, rest_len))

    # Every length must be writable
    for kind, _, start, length in plan:
        if (length * division).denominator != 1:
            return False
        if kind == "note" and length not in _TYPES:
            return False
    if all(kind != "rest" for kind, *_ in plan) and all(
        length == d for (kind, _, _, length), (_, _, d) in zip(plan, sung)
    ):
        return False  # nothing would change

    _apply(measure, plan, division)
    return True


def _rest_pieces(start: Fraction, length: Fraction) -> List[Fraction]:
    out = []
    s, end = start, start + length
    while s < end:
        for v in _REST_STEPS:
            if s % v == 0 and s + v <= end:
                out.append(v)
                s += v
                break
        else:
            out.append(end - s)
            break
    return out


def _set_length(note, length: Fraction, division: int) -> None:
    note.find("duration").text = str(int(length * division))
    if length not in _TYPES:
        return
    ntype, dots = _TYPES[length]
    t = note.find("type")
    if t is None:
        t = etree.Element("type")
        anchor = note.find("voice") if note.find("voice") is not None else note.find("duration")
        anchor.addnext(t)
    t.text = ntype
    for d in note.findall("dot"):
        note.remove(d)
    for _ in range(dots):
        t.addnext(etree.Element("dot"))
    # A member of a chord shares the main note's length
    nxt = note.getnext()
    while nxt is not None and nxt.tag == "note" and nxt.find("chord") is not None:
        nxt.find("duration").text = note.find("duration").text
        tt = nxt.find("type")
        if tt is not None:
            tt.text = ntype
        for d in nxt.findall("dot"):
            nxt.remove(d)
        if tt is not None:
            for _ in range(dots):
                tt.addnext(etree.Element("dot"))
        nxt = nxt.getnext()


def _make_rest(length: Fraction, division: int, voice: str):
    r = etree.Element("note")
    etree.SubElement(r, "rest")
    etree.SubElement(r, "duration").text = str(int(length * division))
    etree.SubElement(r, "voice").text = voice
    if length in _TYPES:
        ntype, dots = _TYPES[length]
        etree.SubElement(r, "type").text = ntype
        for _ in range(dots):
            etree.SubElement(r, "dot")
    return r


def _apply(measure, plan, division: int) -> None:
    voice = next((n.findtext("voice") for n in measure.findall("note") if n.findtext("voice")), "1")
    # Drop old rests; directions in front of a rest move to the next note
    for n in _main_notes(measure):
        if n.find("rest") is not None:
            measure.remove(n)

    # Anchor: insert rests before the next sung note, or before the right barline
    for kind, note, start, length in plan:
        if kind == "note":
            _set_length(note, length, division)
    for i, (kind, note, start, length) in enumerate(plan):
        if kind != "rest":
            continue
        nxt = next((p[1] for p in plan[i + 1:] if p[0] == "note"), None)
        pieces = [_make_rest(v, division, voice) for v in _rest_pieces(start, length)]
        if nxt is not None:
            for r in pieces:
                nxt.addprevious(r)
        else:
            bl = [c for c in measure if c.tag == "barline" and c.get("location") == "right"]
            for r in pieces:
                if bl:
                    bl[0].addprevious(r)
                else:
                    measure.append(r)
