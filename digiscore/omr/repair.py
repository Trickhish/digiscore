"""
Measure-duration repair for OMR output.

The most frequent OMR rhythm error on scans is a missed augmentation dot: the bar
comes out short by exactly half the value of one note. When a single dot added to
a single note makes the bar match its time signature, and the page image shows a
dot-sized dark spot to the right of that note's head, the dot is added.
"""

from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pymupdf
from lxml import etree

from digiscore.omr.book import Chord, SheetInfo

TYPE_VALUE = {
    "breve": Fraction(8), "whole": Fraction(4), "half": Fraction(2), "quarter": Fraction(1),
    "eighth": Fraction(1, 2), "16th": Fraction(1, 4), "32nd": Fraction(1, 8), "64th": Fraction(1, 16),
}


def _note_value(note) -> Optional[Fraction]:
    t = note.findtext("type")
    if t not in TYPE_VALUE:
        return None
    v = TYPE_VALUE[t]
    dots = len(note.findall("dot"))
    total, add = v, v
    for _ in range(dots):
        add /= 2
        total += add
    tm = note.find("time-modification")
    if tm is not None:
        total = total * int(tm.findtext("normal-notes")) / int(tm.findtext("actual-notes"))
    return total


def _voice_lengths(measure, divisions: int) -> Dict[str, Fraction]:
    """Length of each voice in quarter notes, ignoring chord members and grace notes."""
    lengths: Dict[str, Fraction] = {}
    for note in measure.findall("note"):
        if note.find("chord") is not None or note.find("grace") is not None:
            continue
        v = note.findtext("voice") or "1"
        d = note.findtext("duration")
        if d is None:
            continue
        lengths[v] = lengths.get(v, Fraction(0)) + Fraction(int(d), divisions)
    return lengths


def _dot_seen(img: np.ndarray, chord: Chord, interline: float) -> bool:
    """A dark spot just right of the notehead, about the size of a dot."""
    h, w = img.shape
    x0 = int(chord.x1 + 0.15 * interline)
    x1 = int(chord.x1 + 1.3 * interline)
    if x0 >= w:
        return False
    # Search across the whole chord height: the dot sits in the space of one of its heads
    y0 = int(max(0, chord.y0 - 0.5 * interline))
    y1 = int(min(h, chord.y1 + 0.5 * interline))
    win = img[y0:y1, x0:min(w, x1)] < 110
    if not win.any():
        return False
    for xs, ys in _components(win):
        bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        if 0.2 * interline <= bw <= 0.75 * interline and 0.2 * interline <= bh <= 0.75 * interline \
                and len(xs) >= 0.5 * bw * bh:
            return True
    return False


def _components(mask: np.ndarray):
    """4-connected components of a small boolean image, as (xs, ys) lists."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        xs, ys = [], []
        while stack:
            y, x = stack.pop()
            xs.append(x)
            ys.append(y)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        yield xs, ys


def repair_missing_dots(xml_content: str, sheets: List[SheetInfo], pdf_path: str) -> Tuple[str, int]:
    root = etree.fromstring(xml_content.encode("utf-8"))
    parts = {p.get("id"): p.findall("measure") for p in root.findall("part")}

    bar_len, divs = _running_attributes(parts)

    doc = pymupdf.open(pdf_path)
    fixed = 0
    for sheet in sheets:
        if sheet.number - 1 >= len(doc):
            continue
        page = doc[sheet.number - 1]
        zoom = sheet.width / page.rect.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

        for system in sheet.systems:
            for staff in system.staves:
                for sm in staff.measures:
                    key = (staff.part_id, sm.measure_index)
                    measures = parts.get(staff.part_id)
                    if measures is None or sm.measure_index >= len(measures) or key not in bar_len:
                        continue
                    measure = measures[sm.measure_index]
                    if measure.get("implicit") == "yes":
                        continue  # pickup bars are short on purpose
                    division = divs[key]
                    target = bar_len[key]
                    notes = [
                        n for n in measure.findall("note")
                        if n.find("chord") is None and n.find("grace") is None
                    ]
                    sung = [n for n in notes if n.find("rest") is None]
                    if len(sung) != len(sm.chords):
                        continue
                    by_voice = {}
                    for n, c in zip(sung, sorted(sm.chords, key=lambda c: (c.voice, c.slot))):
                        by_voice.setdefault(n.findtext("voice") or "1", []).append((n, c))
                    lengths = _voice_lengths(measure, division)
                    for v, pairs in by_voice.items():
                        missing = target - lengths.get(v, Fraction(0))
                        if missing <= 0:
                            continue
                        candidates = []
                        for n, c in pairs:
                            val = _note_value(n)
                            if val is None or n.findall("dot"):
                                continue
                            if val / 2 == missing and _dot_seen(img, c, staff.interline):
                                candidates.append(n)
                        if len(candidates) != 1:
                            continue
                        _add_dot(candidates[0], division)
                        fixed += 1

    return etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8"), fixed


REST_TYPES = {Fraction(1): ("quarter", 0), Fraction(2): ("half", 0), Fraction(3): ("half", 1), Fraction(4): ("whole", 0)}


def _rest_block_seen(img: np.ndarray, x0: float, x1: float, top: float, bottom: float, interline: float) -> bool:
    """A filled rectangle (half or whole rest) next to the middle staff lines."""
    h, w = img.shape
    y0 = int(top + 0.5 * interline)
    y1 = int(bottom - 0.5 * interline)
    xa, xb = int(max(0, x0)), int(min(w, x1))
    if xb - xa < interline:
        return False
    win = img[y0:y1, xa:xb] < 110
    # Staff lines are long horizontal runs: remove rows that are mostly black
    row_fill = win.mean(axis=1)
    win[row_fill > 0.6, :] = False
    for xs, ys in _components(win):
        bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        if 0.8 * interline <= bw <= 1.6 * interline and 0.25 * interline <= bh <= 0.75 * interline \
                and len(xs) >= 0.7 * bw * bh:
            return True
    return False


def repair_missing_rests(xml_content: str, sheets: List[SheetInfo], pdf_path: str) -> Tuple[str, int]:
    """Append a half/whole rest to single-voice bars that are short by that value when one is visible."""
    root = etree.fromstring(xml_content.encode("utf-8"))
    parts = {p.get("id"): p.findall("measure") for p in root.findall("part")}
    bar_len, divs = _running_attributes(parts)
    doc = pymupdf.open(pdf_path)
    fixed = 0
    for sheet in sheets:
        if sheet.number - 1 >= len(doc):
            continue
        page = doc[sheet.number - 1]
        zoom = sheet.width / page.rect.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        for system in sheet.systems:
            for staff in system.staves:
                for sm in staff.measures:
                    key = (staff.part_id, sm.measure_index)
                    measures = parts.get(staff.part_id)
                    if measures is None or sm.measure_index >= len(measures) or key not in bar_len:
                        continue
                    measure = measures[sm.measure_index]
                    if measure.get("implicit") == "yes":
                        continue
                    lengths = _voice_lengths(measure, divs[key])
                    if len(lengths) != 1:
                        continue
                    missing = bar_len[key] - next(iter(lengths.values()))
                    if missing not in REST_TYPES:
                        continue
                    start = max((c.x1 for c in sm.chords), default=sm.left) + 0.5 * staff.interline
                    if not _rest_block_seen(img, start, sm.right - 0.3 * staff.interline,
                                            staff.top, staff.bottom, staff.interline):
                        continue
                    _append_rest(measure, missing, divs[key], next(iter(lengths.keys())))
                    fixed += 1
    out = etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")
    return out, fixed


def _running_attributes(parts):
    bar_len: Dict[Tuple[str, int], Fraction] = {}
    divs: Dict[Tuple[str, int], int] = {}
    for pid, measures in parts.items():
        cur_div, cur_len = 1, None
        for i, m in enumerate(measures):
            a = m.find("attributes")
            if a is not None:
                if a.findtext("divisions"):
                    cur_div = int(a.findtext("divisions"))
                t = a.find("time")
                if t is not None and t.findtext("beats"):
                    cur_len = Fraction(int(t.findtext("beats")) * 4, int(t.findtext("beat-type")))
            divs[(pid, i)] = cur_div
            if cur_len is not None:
                bar_len[(pid, i)] = cur_len
    return bar_len, divs


def _append_rest(measure, missing: Fraction, divisions: int, voice: str) -> None:
    ntype, dots = REST_TYPES[missing]
    note = etree.Element("note")
    etree.SubElement(note, "rest")
    etree.SubElement(note, "duration").text = str(int(missing * divisions))
    etree.SubElement(note, "voice").text = voice
    etree.SubElement(note, "type").text = ntype
    for _ in range(dots):
        etree.SubElement(note, "dot")
    # Before a right barline, otherwise at the end
    bl = [c for c in measure if c.tag == "barline" and c.get("location") == "right"]
    if bl:
        bl[0].addprevious(note)
    else:
        measure.append(note)


def _barline_seen(img: np.ndarray, xa: float, xb: float, top: float, bottom: float, interline: float) -> Optional[int]:
    """x of a thin vertical stroke covering the whole staff height between xa and xb."""
    h, w = img.shape
    xa, xb = int(max(0, xa)), int(min(w, xb))
    y0, y1 = int(top), int(bottom) + 1
    if xb <= xa or y1 <= y0:
        return None
    win = img[y0:y1, xa:xb] < 128
    cover = win.mean(axis=0)
    best = int(np.argmax(cover))
    if cover[best] < 0.9:
        return None
    # Thin: a barline is a few pixels wide, a notehead or beam is not full height
    run = cover > 0.9
    left = best
    while left > 0 and run[left - 1]:
        left -= 1
    right = best
    while right + 1 < len(run) and run[right + 1]:
        right += 1
    if right - left + 1 > 0.5 * interline:
        return None
    return xa + (left + right) // 2


_UNSEEN = object()


def split_merged_measures(xml_content: str, sheets: List[SheetInfo], pdf_path: str) -> Tuple[str, int]:
    """
    A barline the OMR engine missed merges bars: the stack comes out two or three
    bar lengths long with all its notes. Every staff proposes a cut at each
    multiple of the current bar length where no note crosses the cut and the page
    image shows a full-staff vertical stroke between the notes on either side; a
    cut confirmed by every staff of the system is applied to every part.
    """
    root = etree.fromstring(xml_content.encode("utf-8"))
    parts = [p for p in root.findall("part")]
    by_id = {p.get("id"): p.findall("measure") for p in parts}
    bar_len, divs = _running_attributes(by_id)
    doc = pymupdf.open(pdf_path)

    images: Dict[int, np.ndarray] = {}

    def image_for(sheet):
        if sheet.number not in images:
            page = doc[sheet.number - 1]
            zoom = sheet.width / page.rect.width
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
            images[sheet.number] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return images[sheet.number]

    # Measure index -> cut times all staves confirm; _UNSEEN = no staff looked yet
    candidates: Dict[int, object] = {}
    for sheet in sheets:
        if sheet.number - 1 >= len(doc):
            continue
        for system in sheet.systems:
            for staff in system.staves:
                measures = by_id.get(staff.part_id)
                img = None
                for sm in staff.measures:
                    m_idx = sm.measure_index
                    key = (staff.part_id, m_idx)
                    m = measures[m_idx] if measures and m_idx < len(measures) else None
                    if m is None or m.get("implicit") == "yes" or key not in bar_len:
                        candidates[m_idx] = set()
                        continue
                    if img is None:
                        img = image_for(sheet)
                    cuts = _staff_cuts(m, sm, bar_len[key], divs[key], img, staff)
                    seen = candidates.get(m_idx, _UNSEEN)
                    candidates[m_idx] = cuts if seen is _UNSEEN else (seen & cuts)

    applied = 0
    for m_idx, cuts in sorted(candidates.items(), key=lambda kv: -kv[0]):
        if not cuts:
            continue
        for p in parts:
            ms = p.findall("measure")
            if m_idx >= len(ms) or ms[m_idx].get("implicit") == "yes":
                continue
            for off in sorted(cuts, reverse=True):
                _split_measure(ms[m_idx], off, divs.get((p.get("id"), m_idx), 1))
            applied += 1

    if applied:
        for p in parts:
            # Renumber sequentially, keeping a leading pickup as 0
            first = p.find("measure")
            num = 0 if first is not None and first.get("implicit") == "yes" else 1
            for m in p.findall("measure"):
                m.set("number", str(num))
                num += 1

    out = etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")
    return out, applied


def _staff_cuts(m, sm, target: Fraction, division: int, img, staff) -> Set[Fraction]:
    """Cut times this staff confirms: onsets align and the image shows a barline."""
    if target <= 0:
        return set()
    total = Fraction(0)
    onsets: Dict[str, List[Fraction]] = {}
    for n in m.findall("note"):
        if n.find("chord") is not None or n.find("grace") is not None:
            continue
        v = n.findtext("voice") or "1"
        if n.find("rest") is None:
            onsets.setdefault(v, []).append(total)
        total += Fraction(int(n.findtext("duration") or 0), division)
    if total < target or not onsets:
        return set()

    chords_by_voice: Dict[int, List[Chord]] = {}
    for c in sm.chords:
        chords_by_voice.setdefault(c.voice, []).append(c)
    for lst in chords_by_voice.values():
        lst.sort(key=lambda x: x.slot)

    # Cut at whole multiples of the bar length, strictly inside the measure
    wanted = {k * target for k in range(1, int(total // target) + 1) if k * target < total}
    for v, ons in onsets.items():
        if set(ons) & wanted:
            return set()  # a note crosses the cut
    if set(int(v) for v in onsets) != set(chords_by_voice):
        return set()  # book/MusicXML voices do not correspond

    cuts: Set[Fraction] = set()
    for w in sorted(wanted):
        xa, xb = -1e18, sm.right
        ok = True
        for v, ons in onsets.items():
            chords = chords_by_voice[int(v)]
            if len(chords) != len(ons):
                ok = False
                break
            before = [c for c, o in zip(chords, ons) if o < w]
            after = [c for c, o in zip(chords, ons) if o > w]
            if before:
                xa = max(xa, max(c.x1 for c in before))
            if after:
                xb = min(xb, min(c.x0 for c in after))
        if not ok or xa >= xb:
            continue
        if _barline_seen(img, xa + 0.3 * staff.interline, xb - 0.3 * staff.interline,
                         staff.top, staff.bottom, staff.interline) is not None:
            cuts.add(w)
    return cuts


def _split_measure(measure, offset: Fraction, divisions: int) -> None:
    """Move everything from time `offset` on into a new measure right after this one."""
    children = list(measure)
    t = Fraction(0)
    cut_i = None
    for i, c in enumerate(children):
        if c.tag == "note" and c.find("chord") is None and c.find("grace") is None:
            if t >= offset and cut_i is None:
                cut_i = i
            t += Fraction(int(c.findtext("duration") or 0), divisions)
    if cut_i is None:
        return
    new = etree.Element("measure", number=measure.get("number", "1") + "b")
    for c in children[cut_i:]:
        measure.remove(c)
        new.append(c)
    measure.addnext(new)


_DOT_AFTER = ("grace", "cue", "chord", "pitch", "unpitched", "rest", "duration", "tie",
              "instrument", "footnote", "level", "voice", "type", "dot")


def _add_dot(note, divisions: int) -> None:
    val = _note_value(note)
    dur = note.find("duration")
    new_val = val * Fraction(3, 2)
    dur.text = str(int(new_val * divisions)) if (new_val * divisions).denominator == 1 else dur.text
    # Ties to the next note would now be wrong; a dotted note replaces "note + tie"
    anchor = None
    for c in note:
        if c.tag in _DOT_AFTER:
            anchor = c
    dot = etree.Element("dot")
    if anchor is None:
        note.append(dot)
    else:
        anchor.addnext(dot)
