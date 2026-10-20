"""
Score digiScore against the synthetic ground truth.

    python tools/bench_synthetic.py [--dir examples/synthetic] [--scan DPI] [--engine auto]

--scan renders each PDF to a degraded raster copy at that DPI first, to test the
scanned-score path on the same material.
"""

import argparse
import difflib
import glob
import io
import os
import random
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import music21  # noqa: E402
from lxml import etree  # noqa: E402

from digiscore.pipeline import convert_score  # noqa: E402


def tokens(part):
    out = []
    for n in part.flatten().notesAndRests:
        if n.isRest:
            p = "R"
        elif n.isChord:
            p = ".".join(sorted(x.nameWithOctave for x in n.pitches))
        else:
            p = n.pitch.nameWithOctave
        lyr = tuple(l.text for l in n.lyrics) if not n.isRest else ()
        out.append((p, float(n.quarterLength), lyr))
    return out


def ratio(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(b))


def time_sigs(path):
    """[(measure index, 'beats/type', symbol)] of the first part, from the raw XML."""
    s = music21.converter.parse(path)
    out = []
    for i, m in enumerate(s.parts[0].getElementsByClass("Measure")):
        for t in m.getElementsByClass("TimeSignature"):
            sym = t.symbol if t.symbol in ("common", "cut") else ""
            out.append((i, t.ratioString, sym))
    return out


def fake_scan(pdf, out, dpi, seed=1):
    import pymupdf
    from PIL import Image, ImageFilter
    rnd = random.Random(seed)
    src, dst = pymupdf.open(pdf), pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        im = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        im = im.rotate(0.3, resample=Image.BICUBIC, fillcolor=255).filter(ImageFilter.GaussianBlur(0.5))
        im = im.point(lambda v: max(0, min(255, v + rnd.randint(-10, 10))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
        p = dst.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=buf.getvalue())
    dst.save(out)
    return out


def compare(pred_path, gt_path):
    pred, gt = music21.converter.parse(pred_path), music21.converter.parse(gt_path)
    res = {"parts": f"{len(pred.parts)}/{len(gt.parts)}"}
    notes, lyrics = [], []
    for i, gp in enumerate(gt.parts):
        if i >= len(pred.parts):
            notes.append(0.0)
            continue
        pt, gtk = tokens(pred.parts[i]), tokens(gp)
        notes.append(ratio([t[:2] for t in pt], [t[:2] for t in gtk]))
        lg = [x for t in gtk for x in t[2]]
        lp = [x for t in pt for x in t[2]]
        if lg:
            lyrics.append(ratio(lp, lg))
    res["notes"] = round(min(notes), 3)
    def ties(sc):
        return [(pi, round(float(n.offset + n.getContextByClass("Measure").offset), 3), n.tie.type)
                for pi, p in enumerate(sc.parts) for n in p.recurse().notes
                if n.isNote and n.tie is not None and n.tie.type in ("start", "stop")]
    tg_, tp_ = set(ties(gt)), set(ties(pred))
    res["ties"] = f"{len(tg_ & tp_)}/{len(tg_)}" + (f" +{len(tp_ - tg_)}" if tp_ - tg_ else "")
    res["lyrics"] = round(min(lyrics), 3) if lyrics else None
    tg, tp = time_sigs(gt_path), time_sigs(pred_path)
    res["time"] = "ok" if [x[1:] for x in tg] == [x[1:] for x in tp] else f"{tp} != {tg}"
    res["valid"] = None
    xsd = os.environ.get("MUSICXML_XSD", "/tmp/opencode/xsd/musicxml.xsd")
    if os.path.isfile(xsd):
        res["valid"] = etree.XMLSchema(etree.parse(xsd)).validate(etree.parse(pred_path))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="examples/synthetic")
    ap.add_argument("--scan", type=int)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--only")
    ap.add_argument("--ai", action="store_true")
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
        for gt in sorted(glob.glob(os.path.join(a.dir, "*.mxl"))):
            stem = os.path.basename(gt)[:-4]
            if a.only and a.only not in stem:
                continue
            pdf = gt[:-4] + ".pdf"
            if a.scan:
                pdf = fake_scan(pdf, os.path.join(tmp, f"{stem}.pdf"), a.scan)
            out = os.path.join(tmp, f"{stem}.musicxml")
            try:
                convert_score(pdf, out, engine=a.engine, use_ai=a.ai)
                r = compare(out, gt)
            except Exception as e:
                r = {"error": str(e)[:120]}
            rows.append((stem, r))
            print(stem, r, flush=True)
    ok = [r for _, r in rows if "notes" in r]
    if ok:
        print(f"\n{len(ok)}/{len(rows)} converted | mean notes {sum(r['notes'] for r in ok)/len(ok):.3f}"
              f" | perfect notes {sum(r['notes'] == 1 for r in ok)}"
              f" | time ok {sum(r['time'] == 'ok' for r in ok)}"
              f" | valid {sum(bool(r['valid']) for r in ok)}")


if __name__ == "__main__":
    main()
