"""
Lyrics extraction, verse grouping, and syllable-to-notehead alignment.
"""

import re
import unicodedata
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
    continues_word: bool = False


@dataclass
class LyricLine:
    verse_number: int
    y: float
    syllables: List[Syllable] = field(default_factory=list)


MUSIC_FONT_HINTS = ("leland", "bravura", "emmentaler", "opus", "mscore", "petaluma", "finale", "maestro",
                    "gonville", "sebastian", "musejazz", "feta", "helsinki", "inkpen", "reprise",
                    "petrucci", "sonata", "engraver", "broadway", "capella")


def _is_music_char(c: str, font: str) -> bool:
    if 0xE000 <= ord(c) <= 0xF8FF:
        return True
    from digiscore.smufl import _base_font, _capella_alias, _cp437_alias, _letter_alias, _sonata_alias
    b = _base_font(font)
    if b in _capella_alias or b in _sonata_alias or b in _letter_alias or b in _cp437_alias:
        return True
    f = font.lower()
    return any(h in f for h in MUSIC_FONT_HINTS)


# Tokens set in bold or italic by the last extract_text_tokens call
_STYLED: set = set()


def styled_tokens(page=None) -> set:
    return set(_STYLED)


def extract_text_tokens(page, gap_ratio: float = 0.032) -> List[Tuple]:
    """
    Text tokens (x0, y0, x1, y1, text) built from individual characters.
    Splits on spaces and on horizontal gaps, because engravers often place separate
    syllables in a single text run with only a small gap between them.
    Music-font glyphs are skipped.
    """
    tokens: List[Tuple] = []
    _STYLED.clear()
    raw = page.get_text("rawdict")
    gaps: dict = {}
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chs = span.get("chars", [])
                gaps.setdefault(span.get("font", ""), []).extend(
                    b["bbox"][0] - a["bbox"][2] for a, b in zip(chs, chs[1:])
                    if not a["c"].isspace() and not b["c"].isspace())
    # Near the top of the gaps seen inside runs: what spacing a word may have
    letter_gap = {f: sorted(g)[int(len(g) * 0.9)] for f, g in gaps.items() if len(g) >= 20}
    # Main text size of the page (by characters); clearly smaller text is
    # footers, credits, editorial notes rather than lyrics
    sizes: dict = {}
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                f = span.get("font", "")
                n = sum(1 for ch in span.get("chars", []) if ch["c"].isalpha() and not _is_music_char(ch["c"], f))
                if n:
                    k = round(span.get("size", 0.0), 1)
                    sizes[k] = sizes.get(k, 0) + n
    main_size = max(sizes, key=sizes.get) if sizes else 0.0
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                size = span.get("size", 10.0) or 10.0
                # Bold or italic text in a lyric band is a direction (dim., Soli,
                # Tutti...) when the lyrics themselves are upright: remember it
                styled = bool(span.get("flags", 0) & 18) or re.search(r"(bold|italic|oblique)", font, re.I)

                gap_limit = gap_ratio * size
                # Text set with loose letter spacing (PDF-XChange from Finale): the
                # gaps inside words are themselves about the limit, so judge a
                # gap against the font's usual one on the page
                usual = letter_gap.get(font)
                if usual is not None and usual > 0.5 * gap_limit:
                    gap_limit = max(gap_limit, 2.0 * usual)
                cur = None
                n_before = len(tokens)
                asc = span.get("ascender", 0.8) or 0.8
                desc = span.get("descender", -0.2) or -0.2
                for ch in span.get("chars", []):
                    c = ch["c"]
                    bx0, by0, bx1, by1 = ch["bbox"]
                    # Squashed glyph boxes (some fonts report near-zero height):
                    # rebuild from origin, size and font metrics
                    if by1 - by0 < 0.3 * size and "origin" in ch:
                        oy = ch["origin"][1]
                        by0, by1 = oy - asc * size, oy - desc * size
                    if c.isspace() or _is_music_char(c, font):
                        if cur:
                            tokens.append(tuple(cur))
                        cur = None
                        continue
                    c = unicodedata.normalize("NFKC", c)
                    if cur and bx0 - cur[2] > gap_limit:
                        tokens.append(tuple(cur))
                        cur = None
                    if cur is None:
                        cur = [bx0, by0, bx1, by1, c]
                    else:
                        cur[2] = max(cur[2], bx1)
                        cur[1] = min(cur[1], by0)
                        cur[3] = max(cur[3], by1)
                        cur[4] += c
                if cur:
                    tokens.append(tuple(cur))
                if styled or size < 0.85 * main_size:
                    _STYLED.update(tokens[n_before:])
    # Lyric hyphens are often drawn as short horizontal strokes rather than text
    for d in page.get_drawings():
        items = d.get("items", [])
        if len(items) != 1 or items[0][0] != "l" or "s" not in d.get("type", ""):
            continue
        p1, p2 = items[0][1], items[0][2]
        if abs(p1.y - p2.y) < 0.1 and 1.0 <= abs(p1.x - p2.x) <= 6.0 and (d.get("width") or 0) < 1.2:
            x0, x1 = min(p1.x, p2.x), max(p1.x, p2.x)
            # Box roughly like a "-" glyph: hyphen sits at x-height middle
            tokens.append((x0, p1.y - 3.0, x1, p1.y + 4.5, "-"))
    return tokens


def extract_lyrics_for_staff(
    page_words: List[Tuple],
    staff_bottom_y: float,
    next_staff_top_y: float,
    staff_x_min: float,
    staff_x_max: float,
    styled: Optional[set] = None,
) -> List[LyricLine]:
    """
    Extracts lyric lines located vertically between staff_bottom_y and next_staff_top_y.
    Groups syllables into verses by their y-coordinates.
    """
    # Filter words belonging to this staff's lyric band
    # Margin below staff: at least 3 pt below bottom line, up to next staff top minus 5 pt
    band_words = []
    styled_band = []
    n_page_words = len(page_words)
    max_y = next_staff_top_y - 5.0 if next_staff_top_y > staff_bottom_y else staff_bottom_y + 120.0

    for w in page_words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4].strip()
        if not text or any(0xE000 <= ord(c) <= 0xF8FF for c in text):
            continue
        if re.fullmatch(r"\d+", text):
            continue  # measure numbers, fingering, etc.

        # Check if inside vertical lyric band and within horizontal staff extent
        # Below the staff: top clear of it, or (tightly spaced scores) the
        # lower part of the text box well under the bottom line
        below = staff_bottom_y + 3.0 <= y0 or (
            y0 >= staff_bottom_y - 0.5 and y1 - staff_bottom_y >= 8.0)
        if below and y0 <= max_y and (staff_x_min - 20 <= x0 <= staff_x_max + 20):
            if styled and tuple(w) in styled:
                styled_band.append((x0, y0, x1, y1, text))
            band_words.append((x0, y0, x1, y1, text))

    if not band_words:
        return []
    if styled_band:
        plain = [w for w in band_words if w not in styled_band]
        # Only when the page's lyrics are upright: some editions set the whole
        # text in italic
        if len(styled) < 0.3 * max(1, n_page_words):
            band_words = plain
            if not band_words:
                return []

    # Cluster real words into lines by baseline; hyphen strokes join the line they sit inside
    hyphens = [w for w in band_words if w[4] == "-"]
    texts = sorted((w for w in band_words if w[4] != "-"), key=lambda w: w[3])
    line_clusters: List[List[Tuple]] = []
    for w in texts:
        if line_clusters and abs(w[3] - line_clusters[-1][-1][3]) < 2.5:
            line_clusters[-1].append(w)
        else:
            line_clusters.append([w])
    for h in hyphens:
        hy = (h[1] + h[3]) / 2.0
        for c in line_clusters:
            top = min(w[1] for w in c)
            bot = max(w[3] for w in c)
            if top <= hy <= bot:
                c.append(h)
                break
    if not line_clusters:
        return []

    # A line slightly offset vertically that fills the horizontal gap of the line
    # above (staggered text of one verse) is merged into it
    def _xspan(c):
        return [(w[0], w[2]) for w in c if w[4] != "-"]

    def _overlaps(a, b):
        return any(x0 < bx1 - 1 and bx0 < x1 - 1 for x0, x1 in _xspan(a) for bx0, bx1 in _xspan(b))

    merged: List[List[Tuple]] = []
    for c in line_clusters:
        if merged:
            prev = merged[-1]
            h = max(w[3] - w[1] for w in prev)
            dy = min(w[3] for w in c) - max(w[3] for w in prev)
            if 0 < dy < 0.9 * h and not _overlaps(prev, c):
                prev.extend(c)
                continue
        merged.append(c)
    line_clusters = merged

    # Keep only the block of lyric lines directly under the staff: stop at the first
    # large vertical gap (text blocks of extra stanzas below the music are not lyrics)
    kept: List[List[Tuple]] = []
    for c in line_clusters:
        c_top = min(w[1] for w in c)
        height = max(w[3] - w[1] for w in c)
        if not kept:
            if c_top - staff_bottom_y > max(6.0 * height, 40.0):
                break
        else:
            prev_bottom = max(w[3] for w in kept[-1])
            if c_top - prev_bottom > 1.2 * height:
                break
        kept.append(c)
    line_clusters = kept

    lyric_lines: List[LyricLine] = []

    for idx, cluster in enumerate(line_clusters):
        # Sort words in line from left to right
        cluster.sort(key=lambda w: w[0])
        
        # Detect explicit verse number at start of line (e.g. "1.", "2.", "3/")
        verse_num = idx + 1
        start_text = cluster[0][4]
        match = re.match(r"^(\d+)[\./\)]\s*(.*)", start_text)
        
        syllables: List[Syllable] = []
        pending_continuation = False
        for i, w in enumerate(cluster):
            x0, y0, x1, y1, t = w
            if i == 0 and match:
                verse_num = int(match.group(1))
                t = match.group(2) if match.group(2) else ""
                if not t:
                    continue
            
            # Hyphen detection: attached to the syllable, or a standalone "-" between syllables.
            # A hyphen before the first syllable of a line continues a word from the previous system.
            has_trailing_hyphen = t.endswith("-")
            has_leading_hyphen = t.startswith("-")
            clean_text = t.strip("-\u2010\u2011\u2013 \t")
            if not clean_text:
                if syllables:
                    syllables[-1].has_hyphen = True
                else:
                    pending_continuation = True
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
            if len(syllables) == 1 and (has_leading_hyphen or pending_continuation):
                syllables[0].continues_word = True

        # Determine syllabic types (single, begin, middle, end) based on hyphens
        # or distance to next syllable
        n_syl = len(syllables)
        for i in range(n_syl):
            curr = syllables[i]
            prev = syllables[i - 1] if i > 0 else None
            nxt = syllables[i + 1] if i < n_syl - 1 else None

            prev_hyphen = prev.has_hyphen if prev else curr.continues_word
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

    # Printed numbers that are not a verse count (a psalm's verse "20.",
    # numbering continued from a previous page...): number lines by position
    nums = [l.verse_number for l in lyric_lines]
    if nums and (max(nums) > len(nums) + 2 or len(set(nums)) < len(nums)):
        lines_by_y = sorted(lyric_lines, key=lambda l: l.y)
        for k, l in enumerate(lines_by_y):
            l.verse_number = k + 1
            for syl in l.syllables:
                syl.verse_number = k + 1

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

    # Chords and the voices of a shared staff take one lyric at a given x: the
    # upper voice's head (stem up, else the highest) carries it
    def _rank(n):
        return (0 if n.get("stem_dir") == "up" else 1, n.get("y", 0.0))

    sung = []
    for note in sorted((n for n in notes if not n.get("is_rest") and not n.get("grace")),
                       key=lambda n: n.get("x", 0.0)):
        if sung and abs(note.get("x", 0.0) - sung[-1].get("x", 0.0)) < 1.0:
            if _rank(note) < _rank(sung[-1]):
                sung[-1] = note
            continue
        sung.append(note)

    def head_center(n):
        b = n.get("bbox")
        return (b[0] + b[2]) / 2.0 if b else n.get("x", 0.0)

    for line in lyric_lines:
        v_num = line.verse_number
        taken = set()
        for syl in line.syllables:
            # Syllables are centred under their note (or left-aligned for melismas)
            syl_center = (syl.x0 + syl.x1) / 2.0
            best_note, best_dist = None, float("inf")
            for idx, note in enumerate(sung):
                if idx in taken:
                    continue
                hc = head_center(note)
                dist = min(abs(hc - syl_center), abs(note.get("x", 0.0) - syl.x0))
                if dist < best_dist and dist <= tolerance:
                    best_dist, best_note = dist, idx
            if best_note is None:
                continue
            taken.add(best_note)
            sung[best_note]["lyrics"].append({
                "verse": v_num,
                "text": syl.text,
                "syllabic": syl.syllabic,
                "has_extender": syl.has_extender,
            })
