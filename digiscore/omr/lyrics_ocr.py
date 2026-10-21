"""
Lyrics re-recognition for scanned scores.

NoteLite's own lyric OCR runs Tesseract in legacy mode and merges syllables that
are close together ("Ilest", "gneauquifut"). Here each lyric line it located is
cropped from the page image and read again with Tesseract's LSTM engine at
character level. Characters are grouped into syllables by the horizontal gaps
between them, hyphens are recognised, and each syllable is attached to the
chord whose notehead sits above it.
"""

import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pymupdf
from lxml import etree
from PIL import Image

from digiscore.omr.book import Chord, LyricLine, SheetInfo, StaffInSystem

LSTM_TESSDATA_DIRS = [
    os.environ.get("DIGISCORE_LSTM_TESSDATA"),
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tessdata",
]

HYPHENS = "-\u2010\u2011\u2012\u2013\u2014\u2212_"


@dataclass
class OcrSyllable:
    text: str
    x0: float
    x1: float
    hyphen_after: bool = False
    hyphen_before: bool = False
    chars: Optional[List[Tuple[str, float, float]]] = None  # (char, x0, x1) in page coordinates

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


def _lstm_tessdata() -> Optional[str]:
    for d in LSTM_TESSDATA_DIRS:
        if d and os.path.isfile(os.path.join(d, "eng.traineddata")):
            return d
    return None


def is_available() -> bool:
    return shutil.which("tesseract") is not None and _lstm_tessdata() is not None


def _languages(tessdata: str, wanted: str) -> str:
    have = {f[:-12] for f in os.listdir(tessdata) if f.endswith(".traineddata")}
    langs = [l for l in wanted.split("+") if l in have]
    return "+".join(langs) or "eng"


def _ocr_boxes(img: Image.Image, langs: str, tessdata: str) -> List[Tuple[str, int, int, int, int]]:
    """Character boxes (char, x0, y0, x1, y1) in image coordinates (y from the top)."""
    env = dict(os.environ, TESSDATA_PREFIX=tessdata)
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", langs, "--oem", "1", "--psm", "7", "makebox"],
        input=_png_bytes(img), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, timeout=60,
    )
    h = img.height
    out = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        parts = line.rsplit(" ", 5)
        if len(parts) != 6:
            continue
        ch, x0, y0, x1, y1, _ = parts
        out.append((ch, int(x0), h - int(y1), int(x1), h - int(y0)))
    return out


LYRIC_IMAGE_MODE = os.environ.get("DIGISCORE_LYRIC_IMAGE", "lanczos")


def _page_image(page, zoom: float) -> Image.Image:
    """
    Page in grayscale at the OMR resolution. For a low-resolution scan the page
    is rendered at its native resolution and upscaled with Lanczos, which keeps
    letter edges sharper than the PDF renderer's resampling.
    """
    mode = LYRIC_IMAGE_MODE
    if mode == "plain":
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
        return Image.frombytes("L", (pix.width, pix.height), pix.samples)
    from digiscore.omr.enhance import page_image_dpi

    native = page_image_dpi(page) or 72 * zoom
    target_w = int(round(page.rect.width * zoom))
    target_h = int(round(page.rect.height * zoom))
    if native >= 72 * zoom * 0.9:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
        return Image.frombytes("L", (pix.width, pix.height), pix.samples)
    pix = page.get_pixmap(dpi=int(round(native)), colorspace=pymupdf.csGRAY)
    small = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    big = small.resize((target_w, target_h), Image.LANCZOS)
    if mode == "lanczos-sharp":
        from PIL import ImageFilter
        big = big.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    return big


def _png_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
    return text


def syllables_from_boxes(boxes, x_offset: float, interline: float) -> List[OcrSyllable]:
    """Group character boxes into syllables using the gaps between characters."""
    if not boxes:
        return []
    heights = sorted(b[4] - b[2] for b in boxes if b[0] not in HYPHENS)
    x_height = heights[len(heights) // 2] if heights else interline
    # Letters of one syllable nearly touch; syllables are separated by a visible
    # space. Use the typical letter gap of this line as the reference.
    gaps = sorted(
        b2[1] - b1[3] for b1, b2 in zip(boxes, boxes[1:])
        if not b1[0].isspace() and not b2[0].isspace()
        and b1[0] not in HYPHENS and b2[0] not in HYPHENS and b2[1] - b1[3] >= 0
    )
    letter_gap = gaps[len(gaps) // 3] if gaps else 2
    split_gap = max(letter_gap + 3.5, 0.16 * x_height)

    groups: List[List[tuple]] = []
    prev_x1 = None
    for b in boxes:
        ch = b[0]
        if ch.isspace():
            prev_x1 = None
            groups.append([])
            continue
        if prev_x1 is not None and b[1] - prev_x1 > split_gap:
            groups.append([])
        if not groups:
            groups.append([])
        groups[-1].append(b)
        prev_x1 = b[3]

    syls: List[OcrSyllable] = []
    pending_hyphen = False
    for g in groups:
        if not g:
            continue
        text = _clean("".join(b[0] for b in g))
        text = text.strip("~|`^°*")
        core = text.strip(HYPHENS)
        if core and not any(ch.isalpha() for ch in core):
            continue
        if not core:
            # Standalone hyphen between syllables
            if syls:
                syls[-1].hyphen_after = True
            else:
                pending_hyphen = True
            continue
        # A syllable can carry its hyphen attached ("mys-" or "-tère")
        letters = [b for b in g if b[0] not in HYPHENS]
        s = OcrSyllable(
            text=core,
            x0=x_offset + min(b[1] for b in letters),
            x1=x_offset + max(b[3] for b in letters),
            hyphen_after=text.endswith(tuple(HYPHENS)),
            hyphen_before=text.startswith(tuple(HYPHENS)) or pending_hyphen,
            chars=[(_clean(b[0]), x_offset + b[1], x_offset + b[3]) for b in letters],
        )
        pending_hyphen = False
        # "mys-tère" read as one group: split on the inner hyphen
        if "-" in core:
            pieces = [p for p in core.split("-") if p]
            if len(pieces) > 1:
                span = s.x1 - s.x0
                total = sum(len(p) for p in pieces)
                x = s.x0
                for i, p in enumerate(pieces):
                    w = span * len(p) / total
                    syls.append(OcrSyllable(p, x, x + w, hyphen_after=i < len(pieces) - 1 or s.hyphen_after,
                                            hyphen_before=s.hyphen_before if i == 0 else True))
                    x += w
                continue
        syls.append(s)
    return syls


def _syllabic(prev_hyphen: bool, next_hyphen: bool) -> str:
    if prev_hyphen and next_hyphen:
        return "middle"
    if prev_hyphen:
        return "end"
    if next_hyphen:
        return "begin"
    return "single"


def _split_syllable(s: OcrSyllable, at_x: float) -> Optional[Tuple[OcrSyllable, OcrSyllable]]:
    """Split a syllable between two characters, at the widest gap near at_x."""
    chars = s.chars or []
    if len(chars) < 2:
        return None
    best, best_score = None, None
    for i in range(1, len(chars)):
        gap = chars[i][1] - chars[i - 1][2]
        pos = (chars[i][1] + chars[i - 1][2]) / 2
        score = gap - 0.05 * abs(pos - at_x)
        if best is None or score > best_score:
            best, best_score = i, score
    left, right = chars[:best], chars[best:]
    lt = "".join(c[0] for c in left)
    rt = "".join(c[0] for c in right)
    if not any(ch.isalpha() for ch in lt) or not any(ch.isalpha() for ch in rt):
        return None
    a = OcrSyllable(lt, left[0][1], left[-1][2], hyphen_after=False, hyphen_before=s.hyphen_before, chars=left)
    b = OcrSyllable(rt, right[0][1], right[-1][2], hyphen_after=s.hyphen_after, hyphen_before=False, chars=right)
    return a, b


def _split_options(s: OcrSyllable) -> List[Tuple[OcrSyllable, OcrSyllable, float]]:
    """Possible two-way splits of a syllable, with the gap width at each split point."""
    chars = s.chars or []
    out = []
    for i in range(1, len(chars)):
        left, right = chars[:i], chars[i:]
        lt = "".join(c[0] for c in left)
        rt = "".join(c[0] for c in right)
        if not any(ch.isalpha() for ch in lt) or not any(ch.isalpha() for ch in rt):
            continue
        gap = right[0][1] - left[-1][2]
        a = OcrSyllable(lt, left[0][1], left[-1][2], hyphen_after=False, hyphen_before=s.hyphen_before, chars=left)
        b = OcrSyllable(rt, right[0][1], right[-1][2], hyphen_after=s.hyphen_after, hyphen_before=False, chars=right)
        out.append((a, b, gap))
    return out


def _multi_splits(s: OcrSyllable, max_pieces: int = 4) -> List[Tuple[List[OcrSyllable], float]]:
    """Ways to cut a syllable into 2..max_pieces pieces, with the summed gap width at the cuts."""
    out: List[Tuple[List[OcrSyllable], float]] = []

    def rec(cur: OcrSyllable, done: List[OcrSyllable], gaps: float):
        if len(done) + 1 >= max_pieces:
            return
        for a, b, gap in _split_options(cur):
            out.append((done + [a, b], gaps + gap))
            rec(b, done + [a], gaps + gap)

    rec(s, [], 0.0)
    return out


def assign_to_chords(syls: List[OcrSyllable], chords: List[Chord], interline: float) -> Dict[str, Tuple[str, str]]:
    """
    Align syllables to chords left to right (both keep their order).

    Dynamic programming over (syllable i, chord j) with three moves:
      - syllable i sung on chord j (cost: horizontal distance)
      - chord j has no syllable (melisma; small fixed cost)
      - syllable i was merged by the OCR from two syllables: split it over
        chords j and j+1 (cost: distances plus a penalty, reduced when the
        split falls on a visible gap)
    A syllable that fits no chord is dropped (large cost).
    Returns chord id -> (text, syllabic).
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not syls or not chords:
        return out
    chords = sorted(chords, key=lambda c: c.cx)
    n, m = len(syls), len(chords)
    skip_chord = 1.2 * interline
    drop_syl = 6.0 * interline
    split_pen = 1.0 * interline

    def dist(sy: OcrSyllable, c: Chord) -> float:
        return min(abs(c.cx - sy.cx), abs(c.x0 - sy.x0))

    splits = [_multi_splits(sy) for sy in syls]
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = cost[i][j]
            if cur == INF:
                continue
            if j < m and cur + skip_chord < cost[i][j + 1]:
                cost[i][j + 1] = cur + skip_chord
                back[i][j + 1] = ("skip", None)
            if i < n and cur + drop_syl < cost[i + 1][j]:
                cost[i + 1][j] = cur + drop_syl
                back[i + 1][j] = ("drop", None)
            if i < n and j < m:
                d = dist(syls[i], chords[j])
                if d < 4 * interline and cur + d < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1] = cur + d
                    back[i + 1][j + 1] = ("match", None)
            if i < n and j + 1 < m:
                for pieces, gap_sum in splits[i]:
                    k = len(pieces)
                    if j + k > m:
                        continue
                    d = sum(dist(p, chords[j + t]) for t, p in enumerate(pieces))
                    c2 = cur + d + (k - 1) * split_pen - min(gap_sum, 0.5 * (k - 1) * split_pen)
                    if d < 3 * k * interline and c2 < cost[i + 1][j + k]:
                        cost[i + 1][j + k] = c2
                        back[i + 1][j + k] = ("split", pieces)

    # Walk back
    i, j = n, m
    pairs: List[Tuple[int, OcrSyllable]] = []
    while i > 0 or j > 0:
        move, data = back[i][j]
        if move == "skip":
            j -= 1
        elif move == "drop":
            i -= 1
        elif move == "match":
            pairs.append((j - 1, syls[i - 1]))
            i, j = i - 1, j - 1
        else:
            for t, p in enumerate(reversed(data)):
                pairs.append((j - 1 - t, p))
            i, j = i - 1, j - len(data)
    pairs.reverse()

    for k, (cj, sy) in enumerate(pairs):
        prev = pairs[k - 1][1] if k > 0 else None
        nxt = pairs[k + 1][1] if k + 1 < len(pairs) else None
        prev_h = sy.hyphen_before or (prev is not None and prev.hyphen_after)
        next_h = sy.hyphen_after or (nxt is not None and nxt.hyphen_before)
        out[chords[cj].id] = (sy.text, _syllabic(prev_h, next_h))
    return out


def _lyric_line_offsets(sheets: List[SheetInfo]) -> List[Tuple[float, float]]:
    """Typical (top, bottom) of each verse line below the staff bottom, in interlines."""
    per_verse: Dict[int, List[Tuple[float, float]]] = {}
    for sheet in sheets:
        for system in sheet.systems:
            for staff in system.staves:
                il = staff.interline or 1.0
                for k, ll in enumerate(sorted(staff.lyric_lines, key=lambda l: l.y0)):
                    top, bot = (ll.y0 - staff.bottom) / il, (ll.y1 - staff.bottom) / il
                    if 0 < top < 8:
                        per_verse.setdefault(k, []).append((top, bot))
    out = []
    for k in sorted(per_verse):
        v = per_verse[k]
        if len(v) < 2:
            break
        tops = sorted(t for t, _ in v)
        bots = sorted(b for _, b in v)
        out.append((tops[len(tops) // 2], bots[len(bots) // 2]))
    return out


def _find_missing_lyric_lines(img: Image.Image, staff: StaffInSystem, offsets) -> List[LyricLine]:
    """
    Lyric lines under a staff where the engine found none: a band at the usual
    offset with enough dark text pixels, and clear of the next staff.
    """
    import numpy as np

    il = staff.interline
    x0 = int(staff.left or (staff.measures[0].left if staff.measures else 0))
    x1 = int(staff.measures[-1].right) if staff.measures else img.width
    if x1 - x0 < 4 * il:
        return []
    lines = []
    for k, (t, b) in enumerate(offsets):
        y0 = int(staff.bottom + t * il)
        y1 = int(staff.bottom + b * il)
        if y1 <= y0 or y1 >= img.height:
            break
        band = np.asarray(img.crop((x0, y0, x1, y1)).convert("L"))
        ink = (band < 128).mean()
        # Text lines are a few percent ink; empty paper is near zero
        if ink < 0.01:
            break
        lines.append(LyricLine(f"auto{staff.staff_id}-{k}", k + 1, x0, y0, x1, y1))
    return lines


def relyric(
    xml_content: str,
    sheets: List[SheetInfo],
    pdf_path: str,
    languages: str = "fra+eng",
    vision=None,
) -> str:
    """
    Replace lyrics in the OMR MusicXML with a fresh LSTM reading.
    With a VisionClient, each line is then proofread against its image.
    """
    tessdata = _lstm_tessdata()
    if not tessdata or shutil.which("tesseract") is None:
        return xml_content
    langs = _languages(tessdata, languages)

    root = etree.fromstring(xml_content.encode("utf-8"))
    parts = {p.get("id"): p.findall("measure") for p in root.findall("part")}

    doc = pymupdf.open(pdf_path)
    # Lyrics per part and global measure index: list of (verse, [(text, syllabic) per sung chord in order])
    new_lyrics: Dict[Tuple[str, int], Dict[int, Dict[str, Tuple[str, str]]]] = {}
    touched_parts = set()
    proofread_jobs = []
    placed = []

    for sheet in sheets:
        if sheet.number - 1 >= len(doc):
            continue
        page = doc[sheet.number - 1]
        zoom = sheet.width / page.rect.width
        img = _page_image(page, zoom)

        # Where lyric lines sit relative to their staff, from the staves the engine
        # did find them on: used to look for lines it missed under other staves
        offsets = _lyric_line_offsets(sheets)

        for system in sheet.systems:
            for staff in system.staves:
                if not staff.lyric_lines and offsets:
                    staff.lyric_lines = _find_missing_lyric_lines(img, staff, offsets)
                if not staff.lyric_lines:
                    continue
                chords = [c for m in staff.measures for c in m.chords]
                pad_y = int(0.5 * staff.interline)
                # Lyric lines form one block right under the staff; a big jump means
                # text below the music (extra stanzas), which is not sung on these notes
                block = []
                prev_bottom = staff.bottom
                for ll in sorted(staff.lyric_lines, key=lambda l: l.y0):
                    limit = 6 * staff.interline if not block else 1.5 * staff.interline
                    if ll.y0 - prev_bottom > limit:
                        break
                    block.append(ll)
                    prev_bottom = ll.y1
                # Every verse number NoteLite produced on this staff gets replaced:
                # numbers beyond the kept block become empty (stanza text removed)
                for m in staff.measures:
                    key = (staff.part_id, m.measure_index)
                    for v in range(1, len(staff.lyric_lines) + 1):
                        new_lyrics.setdefault(key, {}).setdefault(v, {})
                for v_idx, ll in enumerate(block):
                    ll.verse = v_idx + 1
                    x0 = max(0, int(ll.x0 - 2 * staff.interline))
                    x1 = min(img.width, int(ll.x1 + 2 * staff.interline))
                    crop = img.crop((x0, max(0, int(ll.y0) - pad_y), x1, min(img.height, int(ll.y1) + pad_y)))
                    syls = syllables_from_boxes(_ocr_boxes(crop, langs, tessdata), x0, staff.interline)
                    mapping = assign_to_chords(syls, chords, staff.interline)
                    if not mapping:
                        continue
                    if vision is not None:
                        proofread_jobs.append((crop, chords, mapping))
                    touched_parts.add(staff.part_id)
                    placed.append((staff, ll.verse, mapping))

    if proofread_jobs:
        _proofread(vision, proofread_jobs)

    for staff, verse, mapping in placed:
        for m in staff.measures:
            slot = new_lyrics.setdefault((staff.part_id, m.measure_index), {}).setdefault(verse, {})
            for c in m.chords:
                if c.id in mapping:
                    slot[c.id] = mapping[c.id]

    # Chord order per (part, measure) to map onto MusicXML notes
    chord_order: Dict[Tuple[str, int], List[str]] = {}
    for sheet in sheets:
        for system in sheet.systems:
            for staff in system.staves:
                for m in staff.measures:
                    chord_order[(staff.part_id, m.measure_index)] = [c.id for c in m.chords]

    for (pid, m_idx), verses in new_lyrics.items():
        measures = parts.get(pid)
        if measures is None or m_idx >= len(measures):
            continue
        notes = [
            n for n in measures[m_idx].findall("note")
            if n.find("rest") is None and n.find("chord") is None and n.find("grace") is None
        ]
        ids = chord_order.get((pid, m_idx), [])
        if len(ids) != len(notes):
            continue
        for verse, mapping in verses.items():
            for note in notes:
                for old in note.findall("lyric"):
                    if old.get("number", "1") == str(verse):
                        note.remove(old)
            for cid, note in zip(ids, notes):
                if cid not in mapping:
                    continue
                text, syl = mapping[cid]
                lyr = etree.Element("lyric", number=str(verse))
                etree.SubElement(lyr, "syllabic").text = syl
                etree.SubElement(lyr, "text").text = text
                _insert_lyric(note, lyr)

    return etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")


def _proofread(vision, jobs) -> None:
    """Correct syllable texts in place with the vision model, several lines at once."""
    from concurrent.futures import ThreadPoolExecutor

    def run(job):
        crop, chords, mapping = job
        order = [c.id for c in sorted(chords, key=lambda c: c.cx) if c.id in mapping]
        texts = [mapping[cid][0] for cid in order]
        fixed = vision.correct_lyric_line(_png_bytes(crop), texts)
        if fixed:
            for cid, t in zip(order, fixed):
                mapping[cid] = (t, mapping[cid][1])

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(run, jobs))


_AFTER_LYRIC = ("play", "listen")


def _insert_lyric(note, lyr) -> None:
    """Lyrics go after notations and before play/listen; keep verse order."""
    anchor = None
    for c in note:
        if c.tag in _AFTER_LYRIC:
            break
        if c.tag == "lyric" and int(c.get("number", "1")) > int(lyr.get("number")):
            break
        anchor = c
    if anchor is None:
        note.insert(0, lyr)
    else:
        anchor.addnext(lyr)
