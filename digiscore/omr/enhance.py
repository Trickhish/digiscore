"""
Image clean-up for low-resolution scans before OMR.

Scans around 100-200 DPI lose thin details (stems, beams, flags, barlines) when
the OMR engine binarises them at their native resolution. Upscaling them first
with a smooth interpolation, removing JPEG noise and binarising with a local
threshold gives the engine sharp, solid strokes at the resolution it expects.
"""

import io
import os
from typing import List, Optional, Tuple

import numpy as np
import pymupdf
from PIL import Image

TARGET_DPI = 400


def page_image_dpi(page) -> Optional[float]:
    """Effective DPI of the largest image on a page, or None for a vector page."""
    best = None
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for r in rects:
            if r.width <= 0:
                continue
            dpi = info[2] / (r.width / 72.0)
            area = r.width * r.height
            if best is None or area > best[1]:
                best = (dpi, area)
    return best[0] if best else None


def staff_interline(gray: np.ndarray) -> Optional[int]:
    """
    Distance between staff lines in pixels: the most common spacing of ink
    runs down the page's columns.
    """
    ink = gray < 128
    h, w = ink.shape
    counts: dict = {}
    for c in ink[:, w // 4: 3 * w // 4: 7].T:
        d = np.diff(np.concatenate([[0], c.astype(np.int8), [0]]))
        starts = np.nonzero(d == 1)[0]
        for a in np.diff(starts):
            if 3 <= a < 80:
                counts[int(a)] = counts.get(int(a), 0) + 1
    return max(counts, key=counts.get) if counts else None


# NoteLite reads best with about 20 pixels between staff lines
TARGET_INTERLINE = 20


def native_interline(page) -> Optional[int]:
    dpi = page_image_dpi(page)
    if not dpi:
        return None
    return staff_interline(page_gray(page, int(round(min(dpi, 600)))))


def needs_enhancement(pdf_path: str, max_dpi: float = 250) -> bool:
    """
    A low-resolution scan: few pixels per inch, or small staves (a reduced page
    keeps its DPI but packs the staff in a few pixels).
    """
    doc = pymupdf.open(pdf_path)
    dpis = [d for d in (page_image_dpi(p) for p in doc) if d]
    if not dpis:
        return False
    if float(np.median(dpis)) < max_dpi:
        return True
    il = native_interline(doc[0])
    return bool(il) and il < 0.7 * TARGET_INTERLINE


MODE = os.environ.get("DIGISCORE_ENHANCE_MODE", "otsu")


def _enhance(gray: np.ndarray, scale: float, mode: str = None) -> np.ndarray:
    import cv2

    mode = mode or MODE
    if mode == "denoise":
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    interp = cv2.INTER_LANCZOS4 if mode != "cubic" else cv2.INTER_CUBIC
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    if mode == "otsu":
        # Strokes keep their real width: one global threshold between ink and paper
        _, up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "sharpen":
        blur = cv2.GaussianBlur(up, (0, 0), sigmaX=scale)
        up = cv2.addWeighted(up, 1.6, blur, -0.6, 0)
    # Otherwise keep grayscale: NoteLite binarises it with its own adaptive filter
    return up


def page_gray(page, dpi: int) -> np.ndarray:
    """The page rendered in grey at dpi."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _paper_unevenness(gray: np.ndarray) -> Tuple[float, float]:
    """
    How far the page is from flat white paper: the paper's median brightness,
    and how much it varies across the page (light falloff, shadows of a photo).
    """
    import cv2
    small = cv2.resize(gray, (max(1, gray.shape[1] // 8), max(1, gray.shape[0] // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return float(np.median(bg)), float(np.percentile(bg, 95) - np.percentile(bg, 5))


def is_photo(gray: np.ndarray) -> bool:
    """A photographed (or badly lit) page: grey or uneven paper."""
    level, spread = _paper_unevenness(gray)
    return level < 215 or spread > 40


def flatten(gray: np.ndarray) -> np.ndarray:
    """
    Even out the lighting of a photographed page and binarise it: divide by the
    paper's own brightness (estimated by closing away the ink), then one global
    threshold between ink and paper.
    """
    import cv2
    h, w = gray.shape
    k = max(15, int(round(min(h, w) / 60)) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    flat = np.clip(gray.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    _, out = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def _line_tracks(ink: np.ndarray, step: int) -> List[List[Tuple[int, float]]]:
    """
    Staff lines followed across the page: in vertical slabs, rows that are
    mostly ink; linked slab to slab from the middle outwards.
    """
    import cv2
    h, w = ink.shape
    long = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, step), 1)))
    slabs = {}
    for x0 in range(0, w - step, step):
        prof = long[:, x0:x0 + step].mean(axis=1)
        rows = np.nonzero(prof > 0.5)[0]
        ys, cur = [], []
        for r in rows:
            if cur and r - cur[-1] > 1:
                ys.append(float(np.mean(cur)))
                cur = []
            cur.append(r)
        if cur:
            ys.append(float(np.mean(cur)))
        slabs[x0 + step // 2] = ys
    xs = sorted(slabs)
    if not xs:
        return []
    mid = max(xs[len(xs) // 4: 3 * len(xs) // 4] or xs, key=lambda x: len(slabs[x]))
    tracks = [[(mid, y)] for y in slabs[mid]]
    if len(tracks) < 2:
        return []
    gaps = np.diff(sorted(slabs[mid]))
    tol = max(2.0, 0.35 * float(np.median(gaps[gaps > 2])) if (gaps > 2).any() else 3.0)
    for direction in (1, -1):
        last = {i: t[0][1] for i, t in enumerate(tracks)}
        order = [x for x in xs if (x - mid) * direction > 0]
        order.sort(key=lambda x: abs(x - mid))
        for x in order:
            ys = slabs[x]
            if not ys:
                continue
            for i, y0 in last.items():
                j = int(np.argmin([abs(y - y0) for y in ys]))
                if abs(ys[j] - y0) < tol:
                    tracks[i].append((x, ys[j]))
                    last[i] = ys[j]
    return [sorted(t) for t in tracks if len(t) >= max(5, len(xs) // 3)]


def dewarp(gray: np.ndarray) -> np.ndarray:
    """
    Straighten bowed staff lines (a photographed page that was not flat): each
    column is shifted vertically so the staff lines run level, the shift
    interpolated between lines.
    """
    import cv2
    h, w = gray.shape
    ink = (gray < 128).astype(np.uint8)
    step = max(8, w // 120)
    tracks = _line_tracks(ink, step)
    if len(tracks) < 5:
        return gray
    grid_x = np.arange(w, dtype=np.float32)
    # Per line: its vertical offset at every x, relative to its mean height
    refs, offs = [], []
    for t in tracks:
        tx = np.array([p[0] for p in t], dtype=np.float64)
        ty = np.array([p[1] for p in t], dtype=np.float64)
        # A smooth page curve: a low-order fit, points that stray (a beam, a
        # jump to the next line) left out
        keep = np.ones(len(tx), bool)
        coef = None
        for _ in range(4):
            if keep.sum() < 5:
                break
            coef = np.polyfit(tx[keep] / w, ty[keep], 3)
            res = np.abs(np.polyval(coef, tx / w) - ty)
            keep = res < max(1.5, 2.5 * float(np.median(res[keep])))
        if coef is None or keep.sum() < max(5, 0.6 * len(tx)):
            continue
        span = (tx[keep].min(), tx[keep].max())
        if span[1] - span[0] < 0.5 * w:
            continue
        fit = np.polyval(coef, np.clip(grid_x, span[0], span[1]) / w)
        ref = float(np.median(ty[keep]))
        refs.append(ref)
        offs.append((fit - ref).astype(np.float32))
    if len(refs) < 5:
        return gray
    order = np.argsort(refs)
    refs = np.array(refs, dtype=np.float32)[order]
    offs = np.array(offs, dtype=np.float32)[order]
    # Neighbouring lines bend together: the median of each line's neighbours
    offs = np.stack([np.median(offs[max(0, i - 2):i + 3], axis=0) for i in range(len(offs))])
    if float(np.abs(offs).max()) < 1.5:
        return gray  # already straight
    # Offset at every pixel: interpolated between lines along y
    ys = np.arange(h, dtype=np.float32)
    coarse = 4
    cx = grid_x[::coarse]
    field = np.empty((h, len(cx)), dtype=np.float32)
    for j, x in enumerate(cx):
        field[:, j] = np.interp(ys, refs, offs[:, int(x)])
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
    map_x = np.tile(grid_x, (h, 1))
    map_y = (ys[:, None] + field).astype(np.float32)
    return cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)


def needs_flattening(pdf_path: str) -> bool:
    doc = pymupdf.open(pdf_path)
    pages = [p for p in doc if page_image_dpi(p)]
    return bool(pages) and sum(is_photo(page_gray(p, 72)) for p in pages) * 2 > len(pages)


def flatten_pdf(pdf_path: str, out_path: str, dpi: int = 300) -> str:
    """A copy of the PDF with each page's lighting evened out and binarised."""
    src = pymupdf.open(pdf_path)
    out = pymupdf.open()
    for page in src:
        native = min(page_image_dpi(page) or dpi, 600)
        il = native_interline(page)
        if il and il < TARGET_INTERLINE:
            dpi = int(min(1200, native * TARGET_INTERLINE / il))
        render = max(dpi, int(round(native)))
        img = dewarp(flatten(page_gray(page, render)))
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG", optimize=True)
        p = out.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=buf.getvalue())
    out.save(out_path, deflate=True)
    return out_path


def enhance_pdf(pdf_path: str, out_path: str, target_dpi: int = TARGET_DPI, mode: str = None) -> str:
    """Write a copy of the PDF with each page re-rendered, upscaled and binarised."""
    src = pymupdf.open(pdf_path)
    out = pymupdf.open()
    for page in src:
        native = page_image_dpi(page) or 150.0
        il = native_interline(page)
        if il and native >= target_dpi * 0.6:
            # Small staves at a decent DPI: upscale until the staff is the size
            # the engine expects (within reason)
            target_dpi = int(min(native * 4, max(target_dpi, native * TARGET_INTERLINE / il)))
        native = min(native, target_dpi)
        # Render at the native resolution (no resampling yet), then upscale ourselves
        pix = page.get_pixmap(dpi=int(round(native)), colorspace=pymupdf.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        img = _enhance(gray, target_dpi / native, mode)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG", optimize=True)
        p = out.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=buf.getvalue())
    out.save(out_path, deflate=True)
    return out_path
