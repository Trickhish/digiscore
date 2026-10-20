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
            barlines.append(Barline(x=x, y0=y0, y1=y1))

    # Check for repeat dots near barlines
    if repeat_dots:
        for bl in barlines:
            # Forward repeat: dots immediately to the right (0 < dot.x - bl.x < 15)
            # Backward repeat: dots immediately to the left (0 < bl.x - dot.x < 15)
            has_dots_right = any(0 < d["x"] - bl.x < 15 and s_top - 5 <= d["y"] <= s_bot + 5 for d in repeat_dots)
            has_dots_left = any(0 < bl.x - d["x"] < 15 and s_top - 5 <= d["y"] <= s_bot + 5 for d in repeat_dots)

            if has_dots_right:
                bl.repeat_direction = "forward"
                bl.bar_type = "heavy-light"
            elif has_dots_left:
                bl.repeat_direction = "backward"
                bl.bar_type = "light-heavy"

    return barlines


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

    for bl in sorted_bl:
        # If barline is virtually at staff start (e.g. initial system barline), skip making a measure
        if abs(bl.x - curr_x0) < 15.0:
            curr_x0 = bl.x
            continue

        # Notes between curr_x0 and bl.x
        meas_notes = [n for n in notes if curr_x0 - 2 <= n.get("x", 0) < bl.x + 2]
        meas_notes.sort(key=lambda n: n.get("x", 0))

        m = Measure(
            measure_number=meas_num,
            staff_idx=staff.staff_idx,
            x0=curr_x0,
            x1=bl.x,
            notes=meas_notes,
            end_barline=bl,
        )
        measures.append(m)
        meas_num += 1
        curr_x0 = bl.x

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
        )
        measures.append(m)

    return measures
