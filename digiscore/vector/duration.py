"""
Rhythmic duration and beam analysis.
"""

from typing import List, Optional

from digiscore.vector.stems import Beam, Stem, count_beams_on_stem, find_stem_for_note

BLACK_DURATIONS = {
    0: ("quarter", 1.0),
    1: ("eighth", 0.5),
    2: ("16th", 0.25),
    3: ("32nd", 0.125),
    4: ("64th", 0.0625),
}

FLAG_LEVELS = {"8th": 1, "16th": 2, "32nd": 3, "64th": 4}


def _flag_levels_on_stem(stem: Stem, flags: List[dict], line_spacing: float) -> int:
    """Flags are drawn with their origin on the stem's free end."""
    best = 0
    for f in flags:
        if abs(f["x"] - stem.x) > 0.6 * line_spacing:
            continue
        if not (stem.y0 - line_spacing <= f["y"] <= stem.y1 + line_spacing):
            continue
        for key, lvl in FLAG_LEVELS.items():
            if key in f["name"]:
                best = max(best, lvl)
    return best


def resolve_note_durations(
    notes: List[dict],
    stems: List[Stem],
    beams: List[Beam],
    flags: List[dict],
    dots: List[dict],
    line_spacing: float = 5.0,
) -> None:
    """
    Computes note duration (in quarter divisions), note type (quarter, eighth, etc.),
    dots, and beam states in-place.
    """
    for n in notes:
        # Check if rest
        is_rest = n.get("is_rest", False)
        glyph_name = n.get("glyph_name", "")

        if is_rest:
            if "Whole" in glyph_name or "Measure" in glyph_name:
                n["type"] = "whole"
                n["duration_quarters"] = 4.0
            elif "Half" in glyph_name:
                n["type"] = "half"
                n["duration_quarters"] = 2.0
            elif "Quarter" in glyph_name:
                n["type"] = "quarter"
                n["duration_quarters"] = 1.0
            elif "8th" in glyph_name or "Eighth" in glyph_name:
                n["type"] = "eighth"
                n["duration_quarters"] = 0.5
            elif "16th" in glyph_name:
                n["type"] = "16th"
                n["duration_quarters"] = 0.25
            elif "32nd" in glyph_name:
                n["type"] = "32nd"
                n["duration_quarters"] = 0.125
            else:
                n["type"] = "quarter"
                n["duration_quarters"] = 1.0
        else:
            # Notehead
            if "DoubleWhole" in glyph_name:
                n["type"] = "breve"
                n["duration_quarters"] = 8.0
            elif "Whole" in glyph_name:
                n["type"] = "whole"
                n["duration_quarters"] = 4.0
            elif "Half" in glyph_name:
                n["type"] = "half"
                n["duration_quarters"] = 2.0
                stem = find_stem_for_note(n, stems, line_spacing)
                n["stem"] = stem
                if stem is not None:
                    n["stem_dir"] = "up" if (stem.y0 + stem.y1) / 2 < n["y"] else "down"
            else:
                # Black notehead: the number of beams or flags on its own stem gives the value
                levels = 0
                stem = find_stem_for_note(n, stems, line_spacing)
                n["stem"] = stem
                if stem is not None:
                    n["stem_dir"] = "up" if (stem.y0 + stem.y1) / 2 < n["y"] else "down"
                    levels = count_beams_on_stem(stem, beams, line_spacing)
                    if levels == 0:
                        levels = _flag_levels_on_stem(stem, flags, line_spacing)
                n["beam_levels"] = levels
                n["type"], n["duration_quarters"] = BLACK_DURATIONS[min(levels, 4)]

        # Augmentation dots
        # Dots sit just right of the head, on the head's space (or the space above a line note)
        bbox = n.get("bbox")
        right = bbox[2] if bbox else n["x"] + line_spacing
        ny = n["y"]
        def _near(d, m):
            mb = m.get("bbox")
            mr = mb[2] if mb else m["x"] + line_spacing
            return 0 < d["x"] - mr < 2.2 * line_spacing and -0.8 * line_spacing < m["y"] - d["y"] < 0.8 * line_spacing

        dot_count = sum(
            1 for d in dots
            if _near(d, n)
            # A rest sharing its place with a note leaves the dot to the note
            and not (is_rest and any(not m.get("is_rest") and m is not n and _near(d, m) for m in notes))
        )
        dot_count = min(dot_count, 2)
        n["dot_count"] = dot_count
        if dot_count == 1:
            n["duration_quarters"] *= 1.5
        elif dot_count == 2:
            n["duration_quarters"] *= 1.75
