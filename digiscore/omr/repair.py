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


def _oval_component(img: np.ndarray, x0: float, x1: float, top: float, interline: float
                    ) -> Optional[Tuple[float, float]]:
    """
    A whole-note-sized blob (staff lines erased, then re-closed vertically) whose
    middle is paper and whose left and right flanks are ink. Returns its centre.
    """
    import cv2

    h, w = img.shape
    ya = int(max(0, top - 4 * interline))
    yb = int(min(h, top + 8 * interline))
    xa, xb = int(max(0, x0)), int(min(w, x1))
    if xb - xa < 1.3 * interline:
        return None
    win = (img[ya:yb, xa:xb] < 128).astype(np.uint8)
    k = max(3, int(2.5 * interline))
    lines = cv2.morphologyEx(win, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    ink0 = win & (1 - lines)
    ink = cv2.morphologyEx(ink0, cv2.MORPH_CLOSE, np.ones((max(3, int(0.3 * interline)), 1), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (1.2 * interline <= bw <= 2.1 * interline and 0.75 * interline <= bh <= 1.3 * interline):
            continue
        if not 0.3 <= area / float(bw * bh) <= 0.85:
            continue
        comp = (lab[y:y + bh, x:x + bw] == i) & (ink0[y:y + bh, x:x + bw] > 0)
        r0, r1 = int(0.3 * bh), max(int(0.3 * bh) + 1, int(0.7 * bh))
        mid = comp[r0:r1, int(0.38 * bw):max(int(0.38 * bw) + 1, int(0.62 * bw))].mean()
        lf = comp[r0:r1, :max(1, int(0.25 * bw))].mean()
        rf = comp[r0:r1, int(0.75 * bw):].mean()
        if mid < 0.2 and lf > 0.45 and rf > 0.45:
            flanks = comp[:, :max(1, int(0.25 * bw))].any(axis=1) | comp[:, int(0.75 * bw):].any(axis=1)
            rows = np.where(flanks)[0]
            cy = (rows[0] + rows[-1]) / 2.0 if len(rows) else bh / 2.0
            return xa + x + bw / 2.0, ya + y + cy
    return None


def _oval_ring(img: np.ndarray, x0: float, x1: float, top: float, bottom: float
               ) -> Optional[Tuple[float, float]]:
    """
    Scan every staff position for a whole-note ring: paper at the centre, thick
    ink flanks, thin ink above and below, paper just outside on one side at
    least. Works where a ledger line, accidental or tie touches the oval.
    """
    import cv2

    h, w = img.shape
    sp = (bottom - top) / 4.0
    ya, yb = int(max(0, top - 5 * sp)), int(min(h, bottom + 5 * sp))
    xa, xb = int(max(0, x0)), int(min(w, x1))
    if xb - xa < 1.5 * sp or sp <= 0:
        return None
    win = (img[ya:yb, xa:xb] < 128).astype(np.uint8)
    lines = cv2.morphologyEx(win, cv2.MORPH_OPEN, np.ones((1, max(3, int(1.9 * sp))), np.uint8))
    ii = cv2.integral((win & (1 - lines)).astype(np.float32))
    H, W = win.shape

    def frac(cx, cy, dx0, dx1, dy0, dy1):
        a, b = int(round(cx + dx0 * sp)), int(round(cx + dx1 * sp))
        c, d = int(round(cy + dy0 * sp)), int(round(cy + dy1 * sp))
        if a < 0 or c < 0 or b > W or d > H or b <= a or d <= c:
            return None
        return (ii[d, b] - ii[c, b] - ii[d, a] + ii[c, a]) / ((b - a) * (d - c))

    best = None
    for p in range(-8, 17):
        cy = bottom - p * sp / 2 - ya
        for cx in range(int(0.9 * sp), W - int(0.9 * sp)):
            cen = frac(cx, cy, -0.18, 0.18, -0.2, 0.2)
            if cen is None or cen > 0.15:
                continue
            lf, rf = frac(cx, cy, -0.72, -0.45, -0.2, 0.2), frac(cx, cy, 0.45, 0.72, -0.2, 0.2)
            if lf is None or rf is None or lf < 0.6 or rf < 0.6:
                continue
            tp, bt = frac(cx, cy, -0.15, 0.15, -0.5, -0.32), frac(cx, cy, -0.15, 0.15, 0.32, 0.5)
            if tp is None or bt is None or tp < 0.3 or bt < 0.3:
                continue
            ol, orr = frac(cx, cy, -1.05, -0.88, -0.15, 0.15), frac(cx, cy, 0.88, 1.05, -0.15, 0.15)
            if ol is None or orr is None or min(ol, orr) > 0.4:
                continue
            score = lf + rf + tp + bt - 2 * cen
            if best is None or score > best[0]:
                best = (score, xa + cx, ya + cy)
    return None if best is None else (best[1], best[2])


def _whole_note_seen(img: np.ndarray, x0: float, x1: float, staff) -> Optional[Tuple[float, float]]:
    """
    Centre (x, y) of a whole-note oval in the x range, or None: either a clean
    hollow blob, or a ring found by scanning the staff positions.
    """
    il = staff.interline
    top, bottom = _local_staff_lines(img, (x0 + x1) / 2.0, staff)
    sp = (bottom - top) / 4.0

    def on_staff(f):
        # Within about two ledger lines of the staff: lower down are lyrics
        return f is not None and -5.5 <= (bottom - f[1]) / (sp / 2.0) <= 13.5

    a = _oval_component(img, x0 + 0.1 * il, x1, staff.top, il)
    b = _oval_ring(img, x0, x1, top, bottom)
    a, b = (a if on_staff(a) else None), (b if on_staff(b) else None)
    if a and b:
        return b if abs(a[0] - b[0]) < il else a
    return a or b


def repair_missing_whole_notes(xml_content: str, sheets: List[SheetInfo], pdf_path: str) -> Tuple[str, int]:
    """
    Whole notes are the head the OMR engine misses most on scans (their thin
    hollow oval). A single-voice bar that is empty or short by a whole note or
    more, with a free gap holding a whole-note oval, gets the note: a whole note
    (or a dotted whole in 6/4, 12/8...) of the pitch the oval sits at.
    """
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
                    # A first bar the engine took for a pickup because it missed the
                    # whole note in it is still a candidate
                    if measure.get("implicit") == "yes" and sm.measure_index != 0:
                        continue
                    target = bar_len[key]
                    lengths = _voice_lengths(measure, divs[key])
                    sung = [n for n in measure.findall("note") if n.find("rest") is None]
                    if len(lengths) > 1:
                        continue
                    have = next(iter(lengths.values()), Fraction(0))
                    # A whole note, dotted when it fills a 6/4 or 12/8 bar
                    whole = Fraction(6) if target == 6 else Fraction(4)
                    il = staff.interline
                    if sung:
                        # A bar short by exactly a whole note, with room before its
                        # first note: the whole note was missed at the start
                        if target - have != Fraction(4) or measure.find("backup") is not None:
                            continue
                        first = min((c.x0 for c in sm.chords), default=None)
                        if first is None:
                            continue
                        left = _bar_content_left(sm, staff)
                        found = _whole_note_seen(img, left + 0.1 * il, first - 0.3 * il, staff)
                        if found is None:
                            continue
                        top, bottom = _local_staff_lines(img, found[0], staff)
                        _prepend_whole(measure, found[1], top, bottom, divs[key])
                        fixed += 1
                        continue
                    # Only an empty bar, or a bar of rests only, is safe to fill
                    if whole > target or (have and have != target):
                        continue
                    left = _bar_content_left(sm, staff)
                    found = _whole_note_seen(img, left + 0.1 * il, sm.right - 0.3 * il, staff)
                    if found is None:
                        continue
                    top, bottom = _local_staff_lines(img, found[0], staff)
                    if not _place_whole(measure, found[1], top, bottom, whole, target, divs[key]):
                        continue
                    fixed += 1
    if fixed:
        _unmark_full_pickup(root, bar_len, divs)
    out = etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")
    return out, fixed


def _unmark_full_pickup(root, bar_len, divs) -> None:
    """
    When every part's first bar, marked as a pickup, now has its full length,
    it was a real bar: clear the mark and number the bars from 1.
    """
    parts = root.findall("part")
    firsts = [p.find("measure") for p in parts]
    if not firsts or any(m is None or m.get("implicit") != "yes" for m in firsts):
        return
    full = 0
    for p, m in zip(parts, firsts):
        key = (p.get("id"), 0)
        if key not in bar_len:
            return
        if not any(n.find("rest") is None for n in m.findall("note")):
            continue  # an unused part (split-part placeholder)
        lengths = _voice_lengths(m, divs[key])
        if not lengths or max(lengths.values()) != bar_len[key]:
            return
        full += 1
    if not full:
        return
    for p in parts:
        p.find("measure").attrib.pop("implicit", None)
        for m in p.findall("measure"):
            num = m.get("number", "")
            if num.isdigit():
                m.set("number", str(int(num) + 1))


def _bar_content_left(sm, staff) -> float:
    """Left edge of a bar's notes: after the barline, or after the staff header."""
    il = staff.interline
    left = sm.left + 0.5 * il
    for box in (staff.time_box, staff.key_box, staff.clef_box):
        if box and box[1] > left and box[0] < sm.right:
            left = box[1]
            break
    if staff.header_stop and staff.header_stop > left and staff.header_start <= sm.left + il:
        left = max(left, staff.header_stop)
    return left


def _pitch_at(measure, y: float, top: float, bottom: float):
    """(step, alter, octave) of a head at y, with the clef and key in force at this measure."""
    clef_sign, clef_line, octave_change, fifths = None, 2, 0, None
    m = measure
    while m is not None and (clef_sign is None or fifths is None):
        a = m.find("attributes")
        if a is not None:
            if clef_sign is None and a.find("clef") is not None:
                clef_sign = a.findtext("clef/sign") or "G"
                clef_line = int(a.findtext("clef/line") or 2)
                octave_change = int(a.findtext("clef/clef-octave-change") or 0)
            if fifths is None and a.find("key/fifths") is not None:
                fifths = int(a.findtext("key/fifths"))
        m = m.getprevious()
        while m is not None and m.tag != "measure":
            m = m.getprevious()
    clef_sign = clef_sign or "G"
    fifths = fifths or 0
    space = (bottom - top) / 4.0
    pos = int(round((bottom - y) / (space / 2.0)))
    ref = {"G": ("E", 4), "F": ("G", 2), "C": ("F", 3)}.get(clef_sign, ("E", 4))
    if clef_sign == "C" and clef_line == 4:
        ref = ("D", 3)
    if clef_sign == "G" and clef_line == 1:
        ref = ("G", 4)
    idx = _STEPS.index(ref[0]) + pos
    step = _STEPS[idx % 7]
    octave = ref[1] + idx // 7 + octave_change
    sharps = "FCGDAEB"[:fifths] if fifths > 0 else ""
    flats = "BEADGCF"[:-fifths] if fifths < 0 else ""
    alter = 1 if step in sharps else -1 if step in flats else 0
    return step, alter, octave


def _note_el(step, alter, octave, length: Fraction, division: int, voice: str):
    note = etree.Element("note")
    p = etree.SubElement(note, "pitch")
    etree.SubElement(p, "step").text = step
    if alter:
        etree.SubElement(p, "alter").text = str(alter)
    etree.SubElement(p, "octave").text = str(octave)
    etree.SubElement(note, "duration").text = str(int(length * division))
    etree.SubElement(note, "voice").text = voice
    etree.SubElement(note, "type").text = "whole"
    if length == Fraction(6):
        etree.SubElement(note, "dot")
    return note


def _prepend_whole(measure, y: float, top: float, bottom: float, division: int) -> None:
    """Insert a whole note before the bar's first note."""
    step, alter, octave = _pitch_at(measure, y, top, bottom)
    first = next(n for n in measure.findall("note"))
    voice = first.findtext("voice") or "1"
    first.addprevious(_note_el(step, alter, octave, Fraction(4), division, voice))


_STEPS = "CDEFGAB"


def _local_staff_lines(img: np.ndarray, x: float, staff) -> Tuple[float, float]:
    """Top and bottom staff line near x (a rotated scan drifts from the staff's mean)."""
    il = staff.interline
    h, w = img.shape
    xa, xb = int(max(0, x - 6 * il)), int(min(w, x + 6 * il))
    ya, yb = int(max(0, staff.top - 1.5 * il)), int(min(h, staff.bottom + 1.5 * il))
    # Darkness per row: staff lines are the darkest rows across the window
    # (on a grey scan they may never reach pure black)
    prof = (255.0 - img[ya:yb, xa:xb].astype(np.float32)).mean(axis=1) / 255.0
    cut = max(0.25, 0.55 * float(prof.max())) if prof.size else 1.0
    lines, run = [], []
    for i, v in enumerate(prof):
        if v > cut:
            run.append(i)
        elif run:
            lines.append(ya + (run[0] + run[-1]) / 2.0)
            run = []
    if run:
        lines.append(ya + (run[0] + run[-1]) / 2.0)
    # Five evenly spaced lines nearest the book's staff
    best = None
    for i in range(len(lines) - 4):
        five = lines[i:i + 5]
        gaps = np.diff(five)
        if gaps.min() < 0.7 * il or gaps.max() > 1.3 * il:
            continue
        err = abs(five[0] - staff.top)
        if best is None or err < best[0]:
            best = (err, five[0], five[4])
    if best is None:
        return staff.top, staff.bottom
    return best[1], best[2]


def _place_whole(measure, y: float, top: float, bottom: float, length: Fraction, target: Fraction,
                 division: int) -> bool:
    """Replace the bar's rests with one note of the given length at the pitch of y."""
    step, alter, octave = _pitch_at(measure, y, top, bottom)
    voice = next((n.findtext("voice") for n in measure.findall("note") if n.findtext("voice")), "1")
    for n in measure.findall("note"):
        measure.remove(n)
    rest_after = target - length
    bl = [c for c in measure if c.tag == "barline" and c.get("location") == "right"]
    anchor = bl[0] if bl else None
    els = [_note_el(step, alter, octave, length, division, voice)]
    if rest_after:
        els.append(_rest_el(rest_after, division, voice))
    for el in els:
        if anchor is not None:
            anchor.addprevious(el)
        else:
            measure.append(el)
    return True


def _rest_el(length: Fraction, division: int, voice: str):
    r = etree.Element("note")
    etree.SubElement(r, "rest")
    etree.SubElement(r, "duration").text = str(int(length * division))
    etree.SubElement(r, "voice").text = voice
    return r
