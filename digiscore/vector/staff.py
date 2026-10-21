"""
Staff and System detection for vector sheet music.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class StaffLine:
    x0: float
    x1: float
    y: float


@dataclass
class Staff:
    page_idx: int
    staff_idx: int          # Global or per-page staff index
    top_y: float            # Line 5 (top)
    bottom_y: float         # Line 1 (bottom)
    lines: List[float]      # Top to bottom (5 lines)
    line_spacing: float     # Spacing between lines (typically 4-10 pt)
    x_min: float
    x_max: float
    system_idx: int = 0
    part_idx: int = 0       # 0=Soprano, 1=Alto, 2=Tenor, 3=Bass, etc.
    part_name: str = ""
    clef_sign: str = "G"    # G, F, C
    clef_line: int = 2      # line 2 for G, line 4 for F
    clef_octave: int = 0    # -1 for tenor G clef (8vb)
    key_fifths: int = 0     # e.g. -3 for Eb major
    time_beats: Optional[int] = None
    time_beat_type: Optional[int] = None
    time_symbol: Optional[str] = None


@dataclass
class System:
    page_idx: int
    system_idx: int
    staves: List[Staff] = field(default_factory=list)
    top_y: float = 0.0
    bottom_y: float = 0.0


def detect_staves_on_page(page, page_idx: int) -> List[Staff]:
    """
    Detects all 5-line music staves on a PyMuPDF page.
    """
    drawings = page.get_drawings()
    h_segments = []

    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 0.2:
                    x0, x1 = min(p1.x, p2.x), max(p1.x, p2.x)
                    if x1 - x0 > 4:
                        h_segments.append((x0, x1, round(p1.y, 2)))
            elif item[0] == "re":
                r = item[1]
                if r.height < 2.0 and r.width > 4:
                    h_segments.append((r.x0, r.x1, round((r.y0 + r.y1) / 2, 2)))

    # Cluster segments by y coordinate (within 0.5 pt of the previous one; lines
    # and beams may be hatched with strokes a fraction of a point apart)
    y_lines = defaultdict(list)
    cur_key, prev_y = None, None
    for x0, x1, y in sorted(h_segments, key=lambda s: s[2]):
        if prev_y is None or y - prev_y >= 0.5:
            cur_key = y
        y_lines[cur_key].append((x0, x1, y))
        prev_y = y

    # Merge touching segments for each line; a staff line is one long continuous run
    # (ledger lines and tie/slur fragments at the same y are short and scattered)
    merged_lines = []
    for y, segs in y_lines.items():
        segs.sort()
        runs = []
        for x0, x1, sy in segs:
            if runs and x0 <= runs[-1][1] + 1.0:
                runs[-1][1] = max(runs[-1][1], x1)
                if x1 - x0 > runs[-1][3]:
                    runs[-1][2], runs[-1][3] = sy, x1 - x0
            else:
                runs.append([x0, x1, sy, x1 - x0])
        best = max(runs, key=lambda r: r[1] - r[0])
        if best[1] - best[0] > 50:
            # y of the run's longest piece: the staff line, not a beam touching it
            merged_lines.append(StaffLine(x0=best[0], x1=best[1], y=best[2]))

    merged_lines.sort(key=lambda l: l.y)
    staves = _find_staves([l for l in merged_lines if l.x1 - l.x0 > 120], page_idx)
    # Short staves (a one-bar last system): five short lines that start and end
    # together, away from the staves already found
    taken = [(st.top_y - st.line_spacing, st.bottom_y + st.line_spacing) for st in staves]
    rest = [l for l in merged_lines if l.x1 - l.x0 <= 120 and not any(a <= l.y <= b for a, b in taken)]
    for st in _find_staves(rest, page_idx, aligned=True):
        staves.append(st)
    staves.sort(key=lambda st: st.top_y)
    for i, st in enumerate(staves):
        st.staff_idx = i
    return staves


def _find_staves(merged_lines: List["StaffLine"], page_idx: int, aligned: bool = False) -> List[Staff]:

    staves: List[Staff] = []
    i = 0
    staff_idx = 0
    while i < len(merged_lines):
        if i + 4 < len(merged_lines):
            cand = merged_lines[i : i + 5]
            spacings = [cand[k + 1].y - cand[k].y for k in range(4)]
            avg_sp = sum(spacings) / 4
            # A valid staff has 5 lines with uniform spacing between 3 and 18 pt
            ok_x = not aligned or (max(c.x0 for c in cand) - min(c.x0 for c in cand) < 1.0
                                   and max(c.x1 for c in cand) - min(c.x1 for c in cand) < 1.0)
            if 3.0 <= avg_sp <= 18.0 and all(abs(sp - avg_sp) < 1.0 for sp in spacings) and ok_x:
                staff = Staff(
                    page_idx=page_idx,
                    staff_idx=staff_idx,
                    top_y=cand[0].y,
                    bottom_y=cand[4].y,
                    lines=[c.y for c in cand],
                    line_spacing=avg_sp,
                    x_min=min(c.x0 for c in cand),
                    x_max=max(c.x1 for c in cand),
                )
                staves.append(staff)
                staff_idx += 1
                i += 5
                continue
        i += 1

    return staves


def group_staves_into_systems(staves: List[Staff], drawings: Optional[List[dict]] = None) -> List[System]:
    """
    Groups staves into systems by checking connecting vertical lines/brackets
    between adjacent staves on the page.
    """
    if not staves:
        return []

    staves_sorted = sorted(staves, key=lambda s: s.top_y)
    n = len(staves_sorted)
    if n == 1:
        sys = System(page_idx=staves_sorted[0].page_idx, system_idx=0, staves=staves_sorted)
        sys.top_y = staves_sorted[0].top_y
        sys.bottom_y = staves_sorted[0].bottom_y
        return [sys]

    connected = [False] * (n - 1)
    if drawings:
        # Vertical pieces (x, y0, y1): lines and thin rectangles
        segs = []
        for d in drawings:
            for item in d["items"]:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.x - p2.x) < 0.5:
                        segs.append((round(p1.x, 1), min(p1.y, p2.y), max(p1.y, p2.y)))
                elif item[0] == "re":
                    r = item[1]
                    if r.width < 5.0:
                        segs.append((round(r.x0, 1), r.y0, r.y1))
        # Some drivers draw a system connector as pieces breaking between
        # staves: chain collinear pieces that touch
        merged = []
        for x, y0, y1 in sorted(segs):
            if merged and merged[-1][0] == x and y0 <= merged[-1][2] + 0.5:
                if y1 > merged[-1][2]:
                    merged[-1][2] = y1
            else:
                merged.append([x, y0, y1])
        for x, y0, y1 in merged:
            for k in range(n - 1):
                s1 = staves_sorted[k]
                s2 = staves_sorted[k + 1]
                if s1.x_min - 10 <= x <= s1.x_max + 10:
                    if y0 <= s1.bottom_y + 2 and y1 >= s2.top_y - 2:
                        connected[k] = True

    systems: List[System] = []
    current_sys = System(page_idx=staves_sorted[0].page_idx, system_idx=0)
    current_sys.staves.append(staves_sorted[0])
    staves_sorted[0].system_idx = 0
    staves_sorted[0].part_idx = 0

    sys_idx = 0
    for k in range(n - 1):
        staff = staves_sorted[k + 1]
        is_conn = connected[k]

        if not is_conn:
            # Start a new system
            current_sys.top_y = current_sys.staves[0].top_y
            current_sys.bottom_y = current_sys.staves[-1].bottom_y
            systems.append(current_sys)

            sys_idx += 1
            current_sys = System(page_idx=staff.page_idx, system_idx=sys_idx)
            staff.system_idx = sys_idx
            staff.part_idx = 0
            current_sys.staves.append(staff)
        else:
            # Same system, next part
            staff.system_idx = sys_idx
            staff.part_idx = len(current_sys.staves)
            current_sys.staves.append(staff)

    current_sys.top_y = current_sys.staves[0].top_y
    current_sys.bottom_y = current_sys.staves[-1].bottom_y
    systems.append(current_sys)

    return systems
