"""
Measure segmentation, barlines, repeat signs and endings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from digiscore.vector.staff import Staff


@dataclass
class Barline:
    x: float
    y0: float
    y1: float
    bar_type: str = "regular"   # regular, double, light-heavy (end), heavy-light (start repeat)
    repeat_direction: Optional[str] = None  # "forward" or "backward"
    ending_number: Optional[str] = None     # "1", "2", etc.
    forward_too: bool = False               # dots on both sides: end and start of repeats


@dataclass
class Measure:
    measure_number: int
    staff_idx: int
    x0: float
    x1: float
    notes: List[dict] = field(default_factory=list)
    start_barline: Optional[Barline] = None
    end_barline: Optional[Barline] = None
    is_pickup: bool = False


def _is_note_stem(x: float, y0: float, y1: float, noteheads: List[dict], sp: float) -> bool:
    """A vertical line with a notehead touching one of its ends is a stem, not a barline."""
    for h in noteheads:
        bbox = h.get("bbox")
        if not bbox:
            continue
        if min(abs(x - bbox[0]), abs(x - bbox[2])) > max(0.9, 0.2 * sp):
            continue
        if abs(h["y"] - y0) < 0.8 * sp or abs(h["y"] - y1) < 0.8 * sp:
            return True
    return False


def extract_barlines_for_staff(
    drawings: List[dict],
    staff: Staff,
    repeat_dots: Optional[List[dict]] = None,
    noteheads: Optional[List[dict]] = None,
) -> List[Barline]:
    """
    Extracts all vertical barlines spanning this staff.
    """
    s_top = staff.top_y
    s_bot = staff.bottom_y
    sp = staff.line_spacing
    noteheads = noteheads or []
    candidates = []

    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.x - p2.x) < 0.3:
                    y0, y1 = min(p1.y, p2.y), max(p1.y, p2.y)
                    # Must reach both top and bottom lines within 1 pt tolerance
                    if y0 <= s_top + 1.0 and y1 >= s_bot - 1.0:
                        candidates.append((p1.x, y0, y1))
            elif item[0] == "re":
                r = item[1]
                if r.width < 4.0:
                    if r.y0 <= s_top + 1.0 and r.y1 >= s_bot - 1.0:
                        candidates.append(((r.x0 + r.x1) / 2, r.y0, r.y1))

    # Cluster x coordinates within 1.5 pt
    candidates.sort(key=lambda c: c[0])
    barlines: List[Barline] = []

    for c in candidates:
        x, y0, y1 = c
        # Only keep barlines within the staff's horizontal span
        if not (staff.x_min - 5 <= x <= staff.x_max + 5):
            continue
        if _is_note_stem(x, y0, y1, noteheads, sp):
            continue
        if not barlines or abs(x - barlines[-1].x) > 2.0:
            # Thin+thick pairs (double, final, repeat barlines) are one barline
            if barlines and x - barlines[-1].x < 1.2 * sp:
                barlines[-1].bar_type = "double"
                continue
            barlines.append(Barline(x=x, y0=y0, y1=y1))

    # Check for repeat dots near barlines
    if repeat_dots:
        for bl in barlines:
            # Forward repeat: dots immediately to the right (0 < dot.x - bl.x < 15)
            # Backward repeat: dots immediately to the left (0 < bl.x - dot.x < 15)
            has_dots_right = any(0 < d["x"] - bl.x < 15 and s_top - 5 <= d["y"] <= s_bot + 5 for d in repeat_dots)
            has_dots_left = any(0 < bl.x - d["x"] < 15 and s_top - 5 <= d["y"] <= s_bot + 5 for d in repeat_dots)

            if has_dots_left:
                bl.repeat_direction = "backward"
                bl.bar_type = "light-heavy"
                bl.forward_too = has_dots_right
            elif has_dots_right:
                bl.repeat_direction = "forward"
                bl.bar_type = "heavy-light"

    return barlines


def find_repeat_dots(page, staff: Staff, barlines: List[Barline], dpi: int = 300,
                     exclude: Optional[List[tuple]] = None) -> List[dict]:
    """
    Repeat dots seen as ink: two small round blots in the 2nd and 3rd staff
    spaces just beside a barline. Works whatever font or drawing produced them
    (each program has its own repeat-dot glyph, some draw a full stop).
    """
    import numpy as np
    import pymupdf

    sp = staff.line_spacing
    out: List[dict] = []
    mid_spaces = (staff.top_y + 1.5 * sp, staff.top_y + 2.5 * sp)
    for bl in barlines:
        clip = pymupdf.Rect(bl.x - 2.6 * sp, staff.top_y + 0.6 * sp, bl.x + 2.6 * sp, staff.bottom_y - 0.6 * sp)
        try:
            pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=pymupdf.csGRAY)
        except Exception:
            continue
        g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        ink = (g < 128).astype(np.uint8)
        # Staff lines and barlines are long thin strokes: keep only compact blobs
        import cv2
        n, _, stats, cent = cv2.connectedComponentsWithStats(ink)
        sc = dpi / 72.0
        blobs = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            wp, hp = w / sc, h / sc
            # A dot: small, round (as wide as tall) and solid (a disk fills ~0.79)
            if 0.25 * sp < wp < 0.75 * sp and 0.25 * sp < hp < 0.75 * sp and 0.7 < wp / hp < 1.4 \
                    and area > 0.65 * w * h:
                cx, cy = clip.x0 + cent[i][0] / sc, clip.y0 + cent[i][1] / sc
                # Part of a known symbol (time signature digits, clef, key)
                if exclude and any(b[0] - 0.5 <= cx <= b[2] + 0.5 and b[1] - 0.5 <= cy <= b[3] + 0.5
                                   for b in exclude):
                    continue
                blobs.append((cx, cy))
        for side in (-1, 1):
            near = [b for b in blobs if 0.2 * sp < side * (b[0] - bl.x) < 2.4 * sp]
            pair = [min((b for b in near if abs(b[1] - y) < 0.3 * sp), key=lambda b: abs(b[1] - y), default=None)
                    for y in mid_spaces]
            # Both dots, one above the other
            if all(pair) and abs(pair[0][0] - pair[1][0]) < 0.3 * sp:
                for b in pair:
                    out.append({"x": b[0], "y": b[1], "name": "repeatDot"})
    return out


def create_measures_for_staff(
    staff: Staff,
    barlines: List[Barline],
    notes: List[dict],
) -> List[Measure]:
    """
    Partitions the staff's notes into measures bounded by barlines.
    """
    if not barlines:
        # Single measure across entire staff
        m = Measure(measure_number=1, staff_idx=staff.staff_idx, x0=staff.x_min, x1=staff.x_max)
        m.notes = sorted(notes, key=lambda n: n.get("x", 0))
        return [m]

    # Sort barlines left to right
    sorted_bl = sorted(barlines, key=lambda b: b.x)
    measures: List[Measure] = []

    # If first barline is to the right of staff start, we have a measure before it
    curr_x0 = staff.x_min
    meas_num = 1

    start_bl: Optional[Barline] = None
    for bl in sorted_bl:
        # If barline is virtually at staff start (e.g. initial system barline), skip making a measure
        if abs(bl.x - curr_x0) < 15.0:
            curr_x0 = bl.x
            start_bl = bl
            continue

        # Notes between curr_x0 and bl.x
        meas_notes = [n for n in notes if curr_x0 - 2 <= n.get("x", 0) < bl.x + 2]
        meas_notes.sort(key=lambda n: n.get("x", 0))

        # Only the clef/key/time header before a start-repeat or double barline
        if not measures and not meas_notes:
            curr_x0 = bl.x
            start_bl = bl
            continue

        m = Measure(
            measure_number=meas_num,
            staff_idx=staff.staff_idx,
            x0=curr_x0,
            x1=bl.x,
            notes=meas_notes,
            start_barline=start_bl,
            end_barline=bl,
        )
        measures.append(m)
        meas_num += 1
        curr_x0 = bl.x
        start_bl = bl

    # Check if notes remain after last barline
    remaining_notes = [n for n in notes if n.get("x", 0) >= curr_x0 + 2]
    if remaining_notes:
        remaining_notes.sort(key=lambda n: n.get("x", 0))
        m = Measure(
            measure_number=meas_num,
            staff_idx=staff.staff_idx,
            x0=curr_x0,
            x1=staff.x_max,
            notes=remaining_notes,
            start_barline=start_bl,
        )
        measures.append(m)

    return measures


def find_endings(page, drawings: List[dict], words: List[tuple], staff: Staff,
                 barline_xs: Optional[List[float]] = None) -> List[dict]:
    """
    Volta brackets above a staff: a number ("1.", "2.", "1., 2.", "3") at the
    start of a bracket: a horizontal stroke above the staff, or (an open last
    ending) just a short hook down. Returns
    [{"number": "1", "text": "1.", "x0": .., "x1": .., "closed": bool}].
    """
    import re

    sp = staff.line_spacing
    top = staff.top_y
    lo, hi = top - 10 * sp, top - 0.3 * sp
    strokes, hooks = [], []
    for d in drawings:
        for it in d["items"]:
            if it[0] == "l" and abs(it[1].y - it[2].y) < 0.3:
                x0, x1, y = min(it[1].x, it[2].x), max(it[1].x, it[2].x), it[1].y
            elif it[0] == "re" and it[1].height < 1.2 and it[1].width > 1.0:
                x0, x1, y = it[1].x0, it[1].x1, (it[1].y0 + it[1].y1) / 2
            elif it[0] == "l" and abs(it[1].x - it[2].x) < 0.3:
                y0, y1 = sorted((it[1].y, it[2].y))
                if lo < y0 < hi and 0.8 * sp < y1 - y0 < 4 * sp and y1 < top:
                    hooks.append((it[1].x, y0, y1))
                continue
            elif it[0] == "re" and it[1].width < 1.2 and lo < it[1].y0 < hi and 0.8 * sp < it[1].height < 4 * sp:
                hooks.append(((it[1].x0 + it[1].x1) / 2, it[1].y0, it[1].y1))
                continue
            else:
                continue
            if x1 - x0 > 3 * sp and lo < y < hi and staff.x_min - 5 < x0 < staff.x_max:
                strokes.append((x0, x1, y))
    # "1, 2" may come as separate tokens: join number tokens close on one line
    nums_w = sorted((w for w in words if re.fullmatch(r"[\d.,]+", w[4].strip())), key=lambda w: (round(w[3]), w[0]))
    joined: List[tuple] = []
    for w in nums_w:
        if joined and abs(joined[-1][3] - w[3]) < 1.0 and 0 <= w[0] - joined[-1][2] < 1.2 * sp \
                and re.search(r"[.,]$", joined[-1][4].strip()):
            j = joined[-1]
            joined[-1] = (j[0], min(j[1], w[1]), w[2], max(j[3], w[3]), j[4].strip() + " " + w[4].strip())
        else:
            joined.append(tuple(w[:5]))
    out = []
    for w in joined:
        text = w[4].strip()
        if not re.fullmatch(r"\d(?:\s*[.,]\s*(?:\d\s*\.?)?)*", text):
            continue
        # A verse number starts a line of words ("3. Where sin did reign")
        if any(abs(o[3] - w[3]) < 1.5 and 0 <= o[0] - w[2] < 1.5 * sp and re.search(r"[^\W\d_]", o[4])
               for o in words):
            continue
        wx0, wy0, wx1, wy1 = w[0], w[1], w[2], w[3]
        if not (lo - 2 * sp < wy0 < top):
            continue
        found = None
        for x0, x1, y in strokes:
            # The number hangs just below the stroke, at its left end
            if -0.5 * sp < wy0 - y < 2.5 * sp and -1.0 * sp < wx0 - x0 < 2.0 * sp:
                closed = any(abs(hx - x1) < 1.0 and hy0 <= y + 0.5 for hx, hy0, _ in hooks)
                found = {"x0": x0, "x1": x1, "y": y, "closed": closed}
                break
        if found is None:
            # No line: an open bracket drawn as a lone hook at the number's left
            for hx, hy0, hy1 in hooks:
                # Only at a barline: elsewhere a short vertical is a stem
                at_bar = barline_xs is not None and any(abs(hx - bx) < 1.5 * sp for bx in barline_xs)
                if at_bar and 0 <= wx0 - hx < 1.5 * sp and hy0 - sp < wy0 < hy1:
                    found = {"x0": hx, "x1": None, "y": hy0, "closed": False}
                    break
        if found is None:
            continue
        nums = re.findall(r"\d", text)
        found.update({"number": ", ".join(nums), "text": text})
        out.append(found)
    # One bracket per start (strokes drawn twice); an open bracket runs one bar
    seen, uniq = set(), []
    for e in sorted(out, key=lambda e: e["x0"]):
        k = round(e["x0"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)
    return uniq
