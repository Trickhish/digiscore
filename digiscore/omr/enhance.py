"""
Image clean-up for low-resolution scans before OMR.

Scans around 100-200 DPI lose thin details (stems, beams, flags, barlines) when
the OMR engine binarises them at their native resolution. Upscaling them first
with a smooth interpolation, removing JPEG noise and binarising with a local
threshold gives the engine sharp, solid strokes at the resolution it expects.
"""

import io
import os
from typing import List, Optional

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


def needs_enhancement(pdf_path: str, max_dpi: float = 250) -> bool:
    doc = pymupdf.open(pdf_path)
    dpis = [d for d in (page_image_dpi(p) for p in doc) if d]
    return bool(dpis) and float(np.median(dpis)) < max_dpi


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


def enhance_pdf(pdf_path: str, out_path: str, target_dpi: int = TARGET_DPI, mode: str = None) -> str:
    """Write a copy of the PDF with each page re-rendered, upscaled and binarised."""
    src = pymupdf.open(pdf_path)
    out = pymupdf.open()
    for page in src:
        native = page_image_dpi(page) or 150.0
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
