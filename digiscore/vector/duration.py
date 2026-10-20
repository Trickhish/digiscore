"""
Rhythmic duration and beam analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def resolve_note_durations(
    notes: List[dict],
    beams: List[dict],
    flags: List[dict],
    dots: List[dict],
    time_beats: Optional[int] = 4,
    time_beat_type: Optional[int] = 4,
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
            else:
                # Black notehead: determine if 16th, 8th, or quarter
                nx = n["x"]
                ny = n["y"]

                # Flags
                has_flag16 = any(abs(f["x"] - nx) < 16 and abs(f["y"] - ny) < 32 and "16" in f["name"] for f in flags)
                has_flag8 = any(abs(f["x"] - nx) < 16 and abs(f["y"] - ny) < 32 and "8" in f["name"] for f in flags)

                # Beams covering notehead x
                matching_beams = [
                    b for b in beams
                    if (b["x0"] - 3.0 <= nx <= b["x1"] + 10.0) and abs(b["y0"] - ny) < 40.0
                ]
                beam_count = len(matching_beams)

                if has_flag16 or beam_count >= 2:
                    n["type"] = "16th"
                    n["duration_quarters"] = 0.25
                elif has_flag8 or beam_count == 1:
                    n["type"] = "eighth"
                    n["duration_quarters"] = 0.5
                else:
                    n["type"] = "quarter"
                    n["duration_quarters"] = 1.0

        # Augmentation dots
        nx = n["x"]
        ny = n["y"]
        dot_count = sum(1 for d in dots if (0 < d["x"] - nx < 16.0) and abs(d["y"] - ny) < 5.0)
        n["dot_count"] = dot_count
        if dot_count == 1:
            n["duration_quarters"] *= 1.5
        elif dot_count == 2:
            n["duration_quarters"] *= 1.75
