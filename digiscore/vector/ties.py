"""
Ties in vector scores.

Ties and slurs are drawn as filled shapes bounded by two curves. A tie joins two
consecutive noteheads of the same pitch: it starts just right of one notehead,
ends just left of the next one at the same height, and is flat. A tie broken by
a system break is drawn as an open arc from the last note to the end of the
staff, and again from the start of the next system to its first note.
"""

from typing import List


def _curved_outline(items) -> bool:
    """A polyline that bends: its points stray from the chord between its ends."""
    pts = [it[1] for it in items] + [items[-1][2]]
    xs = [p.x for p in pts]
    a = pts[xs.index(min(xs))]
    b = pts[xs.index(max(xs))]
    dx, dy = b.x - a.x, b.y - a.y
    length = (dx * dx + dy * dy) ** 0.5
    if length < 3:
        return False
    dev = max(abs((p.x - a.x) * dy - (p.y - a.y) * dx) / length for p in pts)
    # Beams are parallelograms (4-5 corners); a tie bows by a good part of its length
    return dev > 0.06 * length


def find_arcs(drawings: List[dict]) -> List[tuple]:
    """(x0, y0, x1, y1) of filled curved shapes that may be ties or slurs."""
    arcs = []
    for d in drawings:
        if d.get("fill") is None:
            continue
        items = d.get("items", [])
        if not items:
            continue
        kinds = [it[0] for it in items]
        # Finale closes the lens with short lines at its tips (c l c l)
        if set(kinds) == {"c", "l"} and kinds.count("c") >= kinds.count("l"):
            pass
        elif not all(k == "c" for k in kinds):
            # Printer drivers flatten curves into many short segments: a filled
            # lens outline of a dozen or more lines that is not straight
            if len(items) < 12 or not all(it[0] == "l" for it in items):
                continue
            if not _curved_outline(items):
                continue
        r = d["rect"]
        if r.width < 3 or r.height > max(8.0, 0.35 * r.width):
            continue
        arcs.append((r.x0, r.y0, r.x1, r.y1))
    return arcs


def mark_ties(staff, notes: List[dict], arcs: List[tuple]) -> None:
    """
    Sets note["tie_start"] / note["tie_stop"] on resolved pitched notes of one staff.
    notes: the staff's notes with "x", "y", "bbox", "is_rest", "step", "octave", "alter".
    """
    sp = staff.line_spacing
    heads = sorted((n for n in notes if not n.get("is_rest")), key=lambda n: n["x"])
    if not heads:
        return
    y_lo, y_hi = staff.top_y - 6 * sp, staff.bottom_y + 6 * sp

    def pitch(n):
        return (n.get("step"), n.get("octave"), n.get("alter", 0))

    for x0, y0, x1, y1 in arcs:
        cy = (y0 + y1) / 2
        if not (y_lo <= cy <= y_hi):
            continue
        # Notehead the arc leaves from: right edge just before x0, at the arc's height
        # The arc starts and ends near the centres of the two noteheads it joins
        def cx(h):
            return (h["bbox"][0] + h["bbox"][2]) / 2
        # Long ties bow far from the noteheads: measure from the arc's nearer edge
        def near_y(h):
            return min(abs(h["y"] - y0), abs(h["y"] - y1)) < 1.3 * sp

        left = [h for h in heads if abs(cx(h) - x0) < 1.2 * sp and near_y(h)]
        right = [h for h in heads if abs(cx(h) - x1) < 1.2 * sp and near_y(h)]
        left = [h for h in left if h not in right] or left
        left.sort(key=lambda h: abs(h["y"] - cy))
        right.sort(key=lambda h: abs(h["y"] - cy))
        if left and right and left[0] is not right[0]:
            a, b = left[0], right[0]
            # MuseScore anchors a slur exactly on the notehead centres, a tie a
            # little inside them; an arc centred on both heads is a slur
            if abs(x0 - cx(a)) < 0.03 * sp and abs(cx(b) - x1) < 0.03 * sp:
                continue
            # Same staff position (the stop note may be missing an accidental
            # that the tie carries over the barline)
            if pitch(a)[:2] == pitch(b)[:2] and abs(a["y"] - b["y"]) < 0.3 * sp and a["x"] < b["x"]:
                # No other notehead of the same voice between them
                between = [h for h in heads if a["x"] < h["x"] < b["x"] and abs(h["y"] - a["y"]) < 0.3 * sp]
                if not between:
                    a["tie_start"] = True
                    b["tie_stop"] = True
            continue
        # Half ties at a system break
        if x1 >= staff.x_max - 3.5 * sp and not any(cx(h) > x1 for h in heads):
            # Open arc from the last note of the system towards the staff end
            cand = [h for h in heads if abs(cx(h) - x0) < 1.2 * sp and near_y(h)]
            if cand:
                min(cand, key=lambda h: abs(h["y"] - cy))["tie_start"] = True
        elif right and x0 < cx(right[0]) - 1.5 * sp and not any(
                x0 - 0.5 * sp < cx(h) < cx(right[0]) - 0.5 * sp for h in heads):
            # Open arc leading into the first note of the system
            right[0]["tie_stop"] = True
