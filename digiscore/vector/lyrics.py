"""
Lyrics extraction, verse grouping, and syllable-to-notehead alignment.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Syllable:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    verse_number: int = 1
    syllabic: str = "single"  # single, begin, middle, end
    has_hyphen: bool = False
    has_extender: bool = False


@dataclass
class LyricLine:
    verse_number: int
    y: float
    syllables: List[Syllable] = field(default_factory=list)


def extract_lyrics_for_staff(
    page_words: List[Tuple],
    staff_bottom_y: float,
    next_staff_top_y: float,
    staff_x_min: float,
    staff_x_max: float,
) -> List[LyricLine]:
    """
    Extracts lyric lines located vertically between staff_bottom_y and next_staff_top_y.
    Groups syllables into verses by their y-coordinates.
    """
    # Filter words belonging to this staff's lyric band
    # Margin below staff: at least 3 pt below bottom line, up to next staff top minus 5 pt
    band_words = []
    max_y = next_staff_top_y - 5.0 if next_staff_top_y > staff_bottom_y else staff_bottom_y + 120.0
    
    for w in page_words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4].strip()
        # Skip empty or non-text glyphs (SMuFL private use area)
        if not text or (len(text) == 1 and ord(text[0]) >= 0xE000):
            continue
        
        # Check if inside vertical lyric band and within horizontal staff extent
        if staff_bottom_y + 3.0 <= y0 <= max_y and (staff_x_min - 20 <= x0 <= staff_x_max + 20):
            band_words.append((x0, y0, x1, y1, text))

    if not band_words:
        return []

    # Cluster words into horizontal lines by y-coordinate (tolerance ~4 pt)
    line_clusters: List[List[Tuple]] = []
    band_words.sort(key=lambda w: w[1])

    for w in band_words:
        matched = False
        for cluster in line_clusters:
            # Check y difference with existing cluster
            cluster_y = cluster[0][1]
            if abs(w[1] - cluster_y) < 4.5:
                cluster.append(w)
                matched = True
                break
        if not matched:
            line_clusters.append([w])

    # Sort lines from top to bottom (closest to staff is usually verse 1)
    line_clusters.sort(key=lambda c: sum(w[1] for w in c) / len(c))

    lyric_lines: List[LyricLine] = []

    for idx, cluster in enumerate(line_clusters):
        # Sort words in line from left to right
        cluster.sort(key=lambda w: w[0])
        
        # Detect explicit verse number at start of line (e.g. "1.", "2.", "3/")
        verse_num = idx + 1
        start_text = cluster[0][4]
        match = re.match(r"^(\d+)[\./\)]\s*(.*)", start_text)
        
        syllables: List[Syllable] = []
        for i, w in enumerate(cluster):
            x0, y0, x1, y1, t = w
            if i == 0 and match:
                verse_num = int(match.group(1))
                t = match.group(2) if match.group(2) else ""
                if not t:
                    continue
            
            # Hyphen detection: either ends with hyphen or surrounded by hyphens
            has_trailing_hyphen = t.endswith("-")
            has_leading_hyphen = t.startswith("-")
            clean_text = t.strip("- \t")
            if not clean_text:
                continue

            syllables.append(
                Syllable(
                    text=clean_text,
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    verse_number=verse_num,
                    has_hyphen=has_trailing_hyphen,
                )
            )

        # Determine syllabic types (single, begin, middle, end) based on hyphens
        # or distance to next syllable
        n_syl = len(syllables)
        for i in range(n_syl):
            curr = syllables[i]
            prev = syllables[i - 1] if i > 0 else None
            nxt = syllables[i + 1] if i < n_syl - 1 else None

            prev_hyphen = prev.has_hyphen if prev else False
            curr_hyphen = curr.has_hyphen

            if prev_hyphen and curr_hyphen:
                curr.syllabic = "middle"
            elif prev_hyphen and not curr_hyphen:
                curr.syllabic = "end"
            elif not prev_hyphen and curr_hyphen:
                curr.syllabic = "begin"
            else:
                curr.syllabic = "single"

        if syllables:
            lyric_lines.append(
                LyricLine(
                    verse_number=verse_num,
                    y=cluster[0][1],
                    syllables=syllables,
                )
            )

    # Sort lyric lines by verse number
    lyric_lines.sort(key=lambda l: l.verse_number)
    return lyric_lines


def match_syllables_to_notes(
    notes: List[dict],
    lyric_lines: List[LyricLine],
    tolerance: float = 30.0,
) -> None:
    """
    Attaches matching syllables from each verse to the notes in-place.
    Each note dict will have a 'lyrics' list: [{'verse': v, 'text': t, 'syllabic': s}]
    """
    for note in notes:
        note.setdefault("lyrics", [])

    for line in lyric_lines:
        v_num = line.verse_number
        for syl in line.syllables:
            syl_center = (syl.x0 + syl.x1) / 2.0
            
            # Find the best note whose horizontal position aligns with this syllable
            best_note = None
            best_dist = float("inf")
            for note in notes:
                # Note x coordinate
                nx = note.get("x", 0.0)
                dist = abs(nx - syl_center)
                if dist < best_dist and dist <= tolerance:
                    best_dist = dist
                    best_note = note
            
            if best_note is not None:
                best_note["lyrics"].append({
                    "verse": v_num,
                    "text": syl.text,
                    "syllabic": syl.syllabic,
                    "has_extender": syl.has_extender,
                })
