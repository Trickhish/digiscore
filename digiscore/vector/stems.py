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
    for d in drawings:
        r = d["rect"]
        dtype = d.get("type", "")
        items = d.get("items", [])

        # Stroked vertical line
        if "s" in dtype and len(items) == 1 and items[0][0] == "l":
            p1, p2 = items[0][1], items[0][2]
            if abs(p1.x - p2.x) < 0.3 and 1.2 * sp <= abs(p1.y - p2.y) <= 8.0 * sp:
                stems.append(Stem(x=(p1.x + p2.x) / 2, y0=min(p1.y, p2.y), y1=max(p1.y, p2.y)))
            continue

        if "f" not in dtype:
            continue
        if r.width > 60 * sp or r.height > 12 * sp:
            continue  # page backgrounds, brackets

        # Thin filled rectangle used as a stem
        if r.width < 0.3 * sp and 1.2 * sp <= r.height <= 8.0 * sp:
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
    return stems, beams


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
