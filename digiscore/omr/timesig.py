"""
Read the time signature of a scanned score from the page image.

The OMR engine often misses time signatures on scans, and it cannot tell the
common-time C from the cut-time C with a stroke: 4/4 and 2/2 have the same bar
length, so the rhythm check cannot either.

Each candidate (C, cut C, stacked digit pairs) is drawn with SMuFL music fonts
on a five-line staff at the scanned staff's own size, blurred like a scan, and
correlated with the page around the staff header. The best match wins.
"""

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts")
FONTS = ["bravura.otf", "leland.otf"]

# SMuFL codepoints
_COMMON, _CUT = "\ue08a", "\ue08b"
_DIGITS = {str(d): chr(0xE080 + d) for d in range(10)}

CANDIDATES = ["C", "C|", "2/2", "3/2", "4/2", "2/4", "3/4", "4/4", "5/4", "6/4",
              "2/8", "3/8", "5/8", "6/8", "7/8", "9/8", "12/8"]

MIN_SCORE = 0.6


@lru_cache(maxsize=1024)
def _staff_template(cand: str, il: int, font_file: str, blur: float) -> Optional[np.ndarray]:
    """
    Grayscale image (ink dark) of a time signature on a five-line staff, the top
    line at row 2*il, cropped to the signature's columns plus a small margin.
    """
    import cv2

    path = os.path.join(FONT_DIR, font_file)
    if not os.path.isfile(path):
        return None
    font = ImageFont.truetype(path, 4 * il)  # SMuFL: the em is four staff spaces
    w, h = 6 * il, 8 * il
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    top = 2 * il
    lw = max(1, int(round(0.12 * il)))
    for k in range(5):
        y = top + k * il
        d.rectangle((0, y - lw // 2, w, y - lw // 2 + lw - 1), fill=0)
    glyph = Image.new("L", (w, h), 0)
    gd = ImageDraw.Draw(glyph)
    if cand in ("C", "C|"):
        gd.text((il, top + 2 * il), _COMMON if cand == "C" else _CUT, font=font, fill=255, anchor="ls")
    else:
        a, b = cand.split("/")
        ta = "".join(_DIGITS[c] for c in a)
        tb = "".join(_DIGITS[c] for c in b)
        wa, wb = gd.textlength(ta, font=font), gd.textlength(tb, font=font)
        wm = max(wa, wb)
        # SMuFL time-signature digits are centred on their baseline
        gd.text((il + (wm - wa) / 2, top + il), ta, font=font, fill=255, anchor="ls")
        gd.text((il + (wm - wb) / 2, top + 3 * il), tb, font=font, fill=255, anchor="ls")
    g = np.asarray(glyph)
    cols = np.where((g > 127).any(axis=0))[0]
    if not len(cols):
        return None
    img.paste(0, mask=glyph)
    a = np.asarray(img).astype(np.float32)
    if blur > 0:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    x0, x1 = max(0, cols[0] - il // 3), min(w, cols[-1] + 1 + il // 3)
    return a[:, x0:x1]


def _normalise(gray: np.ndarray, interline: float, top_line_y: float) -> Tuple[np.ndarray, int]:
    """Crop rescaled so the interline is an integer il and the top line sits at row 2*il."""
    import cv2

    il = max(6, int(round(interline)))
    scale = il / interline
    g = cv2.resize(gray.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    shift = int(round(top_line_y * scale)) - 2 * il
    if shift > 0:
        g = g[shift:]
    elif shift < 0:
        g = np.vstack([np.full((-shift, g.shape[1]), 255, np.float32), g])
    g = g[:8 * il]
    if g.shape[0] < 8 * il:
        g = np.vstack([g, np.full((8 * il - g.shape[0], g.shape[1]), 255, np.float32)])
    return g, il


def classify_gray(gray: np.ndarray, interline: float, top_line_y: float) -> Dict[str, Tuple[float, float]]:
    """
    Best normalised correlation of each candidate within the crop.
    Returns {candidate: (score, x of the match in crop pixels)}.
    """
    import cv2

    g, il = _normalise(gray, interline, top_line_y)
    band = g[il:7 * il]
    blur = 0.08 * il
    dys = range(-max(1, il // 5), max(1, il // 5) + 1)
    out: Dict[str, Tuple[float, float]] = {}
    for cand in CANDIDATES:
        best, best_x = -1.0, 0.0
        for f in FONTS:
            t = _staff_template(cand, il, f, blur)
            if t is None or t.shape[1] > g.shape[1]:
                continue
            tt = t[il:7 * il]
            for dy in dys:
                r = cv2.matchTemplate(np.roll(band, dy, axis=0), tt, cv2.TM_CCOEFF_NORMED)[0]
                x = int(np.argmax(r))
                if r[x] > best:
                    best, best_x = float(r[x]), x * interline / il
        out[cand] = (best, best_x)
    return out


def _local_staff_top(img: Image.Image, staff_top: float, interline: float, x0: float, x1: float) -> float:
    """
    The staff's top line at this x range: on a slightly rotated scan the line's
    height drifts along the page. Fit five evenly spaced lines to the row
    darkness profile within one space of the given top.
    """
    y0 = int(staff_top - 1.5 * interline)
    y1 = int(staff_top + 5.5 * interline)
    a = 255.0 - np.asarray(img.crop((int(x0), max(0, y0), int(x1), y1)).convert("L"), dtype=np.float32)
    if a.size == 0:
        return staff_top
    prof = a.mean(axis=1)
    best, best_t = -1.0, staff_top
    for t in np.arange(-1.0 * interline, 1.0 * interline + 0.5, 0.5):
        rows = [int(round(staff_top + t + k * interline)) - max(0, y0) for k in range(5)]
        if min(rows) < 0 or max(rows) >= len(prof):
            continue
        v = sum(prof[r] for r in rows)
        if v > best:
            best, best_t = v, staff_top + t
    return float(best_t)


def detect_from_image(img: Image.Image, staff_top: float, interline: float, x0: float, x1: float
                      ) -> Tuple[Optional[str], Dict[str, float]]:
    """Best candidate for the staff whose top line is at staff_top, between x0 and x1 (pixels)."""
    staff_top = _local_staff_top(img, staff_top, interline, x0, x1)
    y0 = int(staff_top - 2 * interline)
    y1 = int(staff_top + 6 * interline)
    crop = img.crop((int(x0), max(0, y0), int(x1), y1)).convert("L")
    gray = np.asarray(crop)
    if y0 < 0:
        gray = np.vstack([np.full((-y0, gray.shape[1]), 255, np.uint8), gray])
    if gray.size == 0 or gray.shape[1] < 2 * interline:
        return None, {}
    res = classify_gray(gray, interline, staff_top - y0)
    scores = {k: v[0] for k, v in res.items()}
    if not scores:
        return None, {}
    best = max(scores, key=scores.get)
    detect_from_image.last_span = _span(best, res[best][1], interline, x0)
    return best, scores


def _span(cand: str, x_in_crop: float, interline: float, x0: float) -> Tuple[float, float]:
    """Page x range covered by the matched template."""
    il = max(6, int(round(interline)))
    t = _staff_template(cand, il, FONTS[0], 0.08 * il)
    w = (t.shape[1] if t is not None else 2 * il) * interline / il
    margin = il // 3 * interline / il
    return x0 + x_in_crop + margin, x0 + x_in_crop + w - margin


def to_rational(label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """('4/4', 'common') for C, ('2/2', 'cut') for C|, ('3/4', None) for digits."""
    if label == "C":
        return "4/4", "common"
    if label == "C|":
        return "2/2", "cut"
    return label, None


def header_window(staff) -> Optional[Tuple[float, float]]:
    """x range to search on a staff of the .omr book: after the clef, up to the first note."""
    il = staff.interline
    start = staff.clef_box[1] if staff.clef_box else staff.header_start + 2.5 * il
    key_end = staff.key_box[1] if staff.key_box else start
    end = max(start, key_end) + 6 * il
    # The engine sometimes takes time-signature digits for a chord: only a chord
    # leaving room for a signature after the key bounds the search
    firsts = [c.x0 for m in staff.measures[:1] for c in m.chords if c.x0 > key_end + 3.5 * il]
    if firsts:
        end = min(end, min(firsts) - 0.1 * il)
    if end - start < 2 * il:
        return None
    return start, end


# staff id -> page x range of the time signature found by the last detection
spans: Dict[str, Tuple[float, float]] = {}


def detect_time_signature_in_book(sheets: List, *pdf_paths: str
                                  ) -> Tuple[Optional[str], Optional[str], Dict[str, float]]:
    """
    Time signature of the first system, averaged over its staves. Several PDFs
    of the same pages (the original scan, a cleaned-up copy) may be given: each
    candidate keeps its best score among them.
    Returns (rational, symbol, mean scores) or (None, None, scores) when unsure.
    """
    import pymupdf

    if not sheets or not sheets[0].systems:
        return None, None, {}
    sheet = sheets[0]
    images = []
    for path in dict.fromkeys(p for p in pdf_paths if p):
        page = pymupdf.open(path)[sheet.number - 1]
        zoom = sheet.width / page.rect.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
        images.append(Image.frombytes("L", (pix.width, pix.height), pix.samples))
    total: Dict[str, float] = {}
    n = 0
    spans.clear()
    for staff in sheet.systems[0].staves:
        win = header_window(staff)
        if win is None:
            continue
        best: Dict[str, float] = {}
        top_label, top_score = None, -1.0
        for img in images:
            label, scores = detect_from_image(img, staff.top, staff.interline, *win)
            for k, v in scores.items():
                best[k] = max(best.get(k, -1.0), v)
            if label and scores[label] > top_score:
                top_label, top_score = label, scores[label]
                spans[staff.staff_id] = detect_from_image.last_span
        if not best:
            continue
        n += 1
        for k, v in best.items():
            total[k] = total.get(k, 0.0) + v
    if not n:
        return None, None, {}
    mean = {k: v / n for k, v in total.items()}
    label = max(mean, key=mean.get)
    if mean[label] < MIN_SCORE:
        return None, None, mean
    rational, symbol = to_rational(label)
    return rational, symbol, mean


def verify_book_time_pairs(omr_path: str, *pdf_paths: str, min_gain: float = 0.025
                           ) -> Dict[Tuple[str, str], str]:
    """
    Check every time signature the engine found against the image. Returns
    {(sheet entry name, inter id): corrected rational} for those whose image
    shows another signature (C/cut map to 4/4 and 2/2). All the signatures of
    a system must agree on the same correction.
    """
    import re
    import zipfile

    import pymupdf
    from lxml import etree

    fixes: Dict[Tuple[str, str], str] = {}
    with zipfile.ZipFile(omr_path) as z:
        names = [n for n in z.namelist() if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", n)]
        sheets = {n: etree.fromstring(z.read(n)) for n in names}
    docs = [pymupdf.open(p) for p in dict.fromkeys(p for p in pdf_paths if p)]
    for name, root in sheets.items():
        num = int(re.search(r"#(\d+)/", name).group(1))
        pic = root.find("picture")
        il = float(root.find("scale/interline").get("main"))
        images = []
        for doc in docs:
            if num - 1 >= len(doc):
                continue
            page = doc[num - 1]
            zoom = float(pic.get("width")) / page.rect.width
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
            images.append(Image.frombytes("L", (pix.width, pix.height), pix.samples))
        for sysm in root.iter("system"):
            staff_top = {}
            for st in sysm.iter("staff"):
                ys = [float(q.get("y")) for q in st.iter("point")]
                if ys:
                    staff_top[st.get("id")] = min(ys)
            sys_fix: Dict[str, str] = {}
            sys_pairs = 0
            for c in sysm.findall("sig/inters"):
                for e in c:
                    if e.tag != "time-pair" or e.get("staff") not in staff_top:
                        continue
                    sys_pairs += 1
                    b = e.find("bounds")
                    x = float(b.get("x"))
                    w = float(b.get("w"))
                    best: Dict[str, float] = {}
                    for img in images:
                        _, sc = detect_from_image(img, staff_top[e.get("staff")], il,
                                                  x - 1.5 * il, x + w + 1.5 * il)
                        for k, v in sc.items():
                            best[k] = max(best.get(k, -1.0), v)
                    if not best:
                        continue
                    label = max(best, key=best.get)
                    rational, _ = to_rational(label)
                    have = e.get("time-rational")
                    if rational and rational != have and best[label] >= MIN_SCORE:
                        # The engine's own reading must score clearly lower
                        own = max(best.get(have, -1.0),
                                  best.get("C", -1.0) if have == "4/4" else -1.0,
                                  best.get("C|", -1.0) if have == "2/2" else -1.0)
                        if best[label] - own >= min_gain:
                            sys_fix[e.get("id")] = rational
            if sys_fix and len(sys_fix) == sys_pairs and len(set(sys_fix.values())) == 1:
                for pid, rational in sys_fix.items():
                    fixes[(name, pid)] = rational
    return fixes


def apply_time_pair_fixes(omr_path: str, fixes: Dict[Tuple[str, str], str]) -> bool:
    """Rewrite the time-rational of the given time-pairs and make the engine redo the rhythm."""
    import re
    import zipfile

    from lxml import etree

    if not fixes:
        return False
    with zipfile.ZipFile(omr_path) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    for name in {n for n, _ in fixes}:
        root = etree.fromstring(entries[name])
        for e in root.iter("time-pair"):
            new = fixes.get((name, e.get("id")))
            if new:
                e.set("time-rational", new)
                e.set("frozen", "true")
        # The pair's digit inters would contradict the new value: drop them
        for sysm in root.iter("system"):
            pairs = [e for e in sysm.iter("time-pair") if (name, e.get("id")) in fixes]
            boxes = [p.find("bounds") for p in pairs]
            gone = set()
            for c in sysm.findall("sig/inters"):
                for e in list(c):
                    if e.tag == "time-number" and any(_inside_box(e.find("bounds"), bb) for bb in boxes):
                        gone.add(e.get("id"))
                        c.remove(e)
            for rel in list(sysm.iter("relation")):
                if rel.get("source") in gone or rel.get("target") in gone:
                    rel.getparent().remove(rel)
        entries[name] = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    bk = entries["book.xml"].decode("utf-8")
    bk = re.sub(r" RHYTHMS PAGE</steps>", "</steps>", bk)
    bk = re.sub(r" RHYTHMS</steps>", "</steps>", bk)
    bk = re.sub(r" PAGE</steps>", "</steps>", bk)
    entries["book.xml"] = bk.encode("utf-8")
    with zipfile.ZipFile(omr_path, "w", zipfile.ZIP_STORED) as zw:
        for n, data in entries.items():
            zw.writestr(n, data)
    return True


def _inside_box(b, outer) -> bool:
    if b is None or outer is None:
        return False
    x, y, w, h = (float(b.get(k)) for k in ("x", "y", "w", "h"))
    X, Y, W, H = (float(outer.get(k)) for k in ("x", "y", "w", "h"))
    return X - 2 <= x and Y - 2 <= y and x + w <= X + W + 2 and y + h <= Y + H + 2
