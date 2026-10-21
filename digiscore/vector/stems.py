"""
Stem and beam geometry extraction.
Links each notehead to its own stem, and counts only the beams that touch that stem.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Stem:
    x: float
    y0: float  # top
    y1: float  # bottom


@dataclass
class Beam:
    points: List[Tuple[float, float]] = field(default_factory=list)
    x0: float = 0.0
    x1: float = 0.0

    def y_range_at(self, x: float) -> Optional[Tuple[float, float]]:
        """Vertical extent of the beam polygon at abscissa x (None if outside)."""
        if x < self.x0 - 0.8 or x > self.x1 + 0.8:
            return None
        x = min(max(x, self.x0), self.x1)
        ys = []
        n = len(self.points)
        for i in range(n):
            (ax, ay), (bx, by) = self.points[i], self.points[(i + 1) % n]
            if abs(bx - ax) < 1e-6:
                if abs(x - ax) < 0.05:
                    ys.extend([ay, by])
                continue
            if min(ax, bx) - 1e-6 <= x <= max(ax, bx) + 1e-6:
                t = (x - ax) / (bx - ax)
                ys.append(ay + t * (by - ay))
        if not ys:
            return None
        return min(ys), max(ys)


def _polygon_points(drawing) -> List[Tuple[float, float]]:
    pts = []
    for it in drawing["items"]:
        if it[0] == "l":
            for p in (it[1], it[2]):
                pt = (p.x, p.y)
                if not pts or abs(pts[-1][0] - pt[0]) > 1e-3 or abs(pts[-1][1] - pt[1]) > 1e-3:
                    pts.append(pt)
        elif it[0] == "re":
            r = it[1]
            pts.extend([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
        elif it[0] == "qu":
            q = it[1]
            pts.extend([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)])
        else:
            return []
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-3 and abs(pts[0][1] - pts[-1][1]) < 1e-3:
        pts.pop()
    return pts


def extract_stems_and_beams(drawings, line_spacing: float = 5.0) -> Tuple[List[Stem], List[Beam]]:
    """
    Stems: thin vertical strokes (or thin filled rectangles) shorter than ~8 staff spaces.
    Beams: filled quadrilaterals, wider than about one staff space, about half a space thick.
    """
    sp = line_spacing
    stems: List[Stem] = []
    beams: List[Beam] = []
    hatch: List[tuple] = []
    for d in drawings:
        r = d["rect"]
        dtype = d.get("type", "")
        items = d.get("items", [])

        # Stroked vertical line
        if "s" in dtype and len(items) == 1 and items[0][0] == "l":
            p1, p2 = items[0][1], items[0][2]
            if abs(p1.x - p2.x) < 0.3 and 0.3 * sp <= abs(p1.y - p2.y) <= 8.0 * sp:
                stems.append(Stem(x=(p1.x + p2.x) / 2, y0=min(p1.y, p2.y), y1=max(p1.y, p2.y)))
            elif abs(p2.x - p1.x) >= 0.8 * sp and abs(p2.y - p1.y) < abs(p2.x - p1.x):
                if p1.x > p2.x:
                    p1, p2 = p2, p1
                hatch.append((p1.x, p1.y, p2.x, p2.y, d.get("width") or 0.0))
            continue

        if "f" not in dtype:
            continue
        if r.width > 60 * sp or r.height > 12 * sp:
            continue  # page backgrounds, brackets

        # Thin filled rectangle used as a stem
        if r.width < 0.3 * sp and 0.3 * sp <= r.height <= 8.0 * sp:
            stems.append(Stem(x=(r.x0 + r.x1) / 2, y0=r.y0, y1=r.y1))
            continue

        pts = _polygon_points(d)
        if len(pts) != 4:
            continue
        # Thickness: vertical extent at the middle of the shape
        b = Beam(points=pts, x0=min(p[0] for p in pts), x1=max(p[0] for p in pts))
        if b.x1 - b.x0 < 0.8 * sp:
            continue
        yr = b.y_range_at((b.x0 + b.x1) / 2)
        if not yr:
            continue
        thick = yr[1] - yr[0]
        if 0.25 * sp <= thick <= 0.9 * sp:
            beams.append(b)
    beams.extend(_hatched_beams(hatch, sp))
    return [t for t in _chain_stems(stems) if 1.2 * sp <= t.y1 - t.y0 <= 8.5 * sp], beams


def _chain_stems(stems: List[Stem]) -> List[Stem]:
    """
    Joins stem pieces on one line that touch end to end (some programs draw a
    chord's stem from head to head); drops strokes repeated side by side.
    """
    out: List[Stem] = []
    for st in sorted(stems, key=lambda t: (t.y0, t.x)):
        for o in out:
            if abs(o.x - st.x) < 0.6 and st.y0 <= o.y1 + 0.3 and st.y1 >= o.y0 - 0.3:
                o.y0, o.y1 = min(o.y0, st.y0), max(o.y1, st.y1)
                break
        else:
            out.append(Stem(x=st.x, y0=st.y0, y1=st.y1))
    return out


def _hatched_beams(lines: List[tuple], sp: float) -> List[Beam]:
    """
    Some programs (Mozart) fill a beam with a stack of thin parallel strokes:
    lines with the same ends in x, each a little lower than the last.
    """
    groups = {}
    for ln in lines:
        groups.setdefault((round(ln[0], 1), round(ln[2], 1)), []).append(ln)
    out = []
    for g in groups.values():
        g.sort(key=lambda ln: ln[1])
        run = [g[0]]
        for ln in g[1:] + [None]:
            prev = run[-1]
            if ln is not None and ln[1] - prev[1] <= 0.3 and abs((ln[3] - ln[1]) - (prev[3] - prev[1])) < 0.1:
                run.append(ln)
                continue
            if len(run) >= 3:
                a, b = run[0], run[-1]
                w = a[4] / 2
                thick = b[1] - a[1] + 2 * w
                if 0.25 * sp <= thick <= 0.9 * sp:
                    pts = [(a[0], a[1] - w), (a[2], a[3] - w), (b[2], b[3] + w), (b[0], b[1] + w)]
                    out.append(Beam(points=pts, x0=a[0], x1=a[2]))
            if ln is not None:
                run = [ln]
    return out


def find_stem_for_note(note: dict, stems: List[Stem], line_spacing: float) -> Optional[Stem]:
    """The stem attached to the notehead's right edge (stem up) or left edge (stem down)."""
    bbox = note.get("bbox")
    if not bbox:
        return None
    hx0, hx1 = bbox[0], bbox[2]
    hy = note["y"]
    tol_x = max(0.9, 0.2 * line_spacing)
    tol_y = 0.75 * line_spacing
    best, best_d = None, None
    for s in stems:
        if not (s.y0 - tol_y <= hy <= s.y1 + tol_y):
            continue
        # Some fonts (e.g. Sibelius Opus) report the advance width as the glyph box,
        # so the stem can lie inside the box rather than on its edge.
        if not (hx0 - tol_x <= s.x <= hx1 + tol_x):
            continue
        dx = min(abs(s.x - hx1), abs(s.x - hx0))
        # Prefer stems ending at the notehead
        dy = min(abs(hy - s.y0), abs(hy - s.y1))
        score = dx + 0.1 * dy
        if best is None or score < best_d:
            best, best_d = s, score
    return best


def count_beams_on_stem(stem: Stem, beams: List[Beam], line_spacing: float) -> int:
    """Number of distinct beams (including partial beam hooks) touching the stem."""
    count = 0
    tol = 0.3 * line_spacing
    for b in beams:
        yr = b.y_range_at(stem.x)
        if yr is None:
            continue
        if yr[1] >= stem.y0 - tol and yr[0] <= stem.y1 + tol:
            count += 1
    return count
