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


def detect_from_image(img: Image.Image, staff_top: float, interline: float, x0: float, x1: float
                      ) -> Tuple[Optional[str], Dict[str, float]]:
    """Best candidate for the staff whose top line is at staff_top, between x0 and x1 (pixels)."""
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
    return max(scores, key=scores.get), scores


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
    end = start + 9 * il
    firsts = [c.x0 for m in staff.measures[:1] for c in m.chords]
    if firsts:
        end = min(end, min(firsts) - 0.1 * il)
    if end - start < 2 * il:
        return None
    return start, end


def detect_time_signature_in_book(sheets: List, pdf_path: str
                                  ) -> Tuple[Optional[str], Optional[str], Dict[str, float]]:
    """
    Time signature of the first system, summed over its staves.
    Returns (rational, symbol, summed scores) or (None, None, {}) when unsure.
    """
    import pymupdf

    if not sheets or not sheets[0].systems:
        return None, None, {}
    sheet = sheets[0]
    page = pymupdf.open(pdf_path)[sheet.number - 1]
    zoom = sheet.width / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    total: Dict[str, float] = {}
    n = 0
    for staff in sheet.systems[0].staves:
        win = header_window(staff)
        if win is None:
            continue
        _, scores = detect_from_image(img, staff.top, staff.interline, *win)
        if not scores:
            continue
        n += 1
        for k, v in scores.items():
            total[k] = total.get(k, 0.0) + v
    if not n:
        return None, None, {}
    mean = {k: v / n for k, v in total.items()}
    label = max(mean, key=mean.get)
    if mean[label] < MIN_SCORE:
        return None, None, mean
    rational, symbol = to_rational(label)
    return rational, symbol, mean
