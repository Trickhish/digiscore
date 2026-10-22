"""
Score digiScore against the synthetic ground truth.

    python tools/bench_synthetic.py [--dir examples/synthetic] [--scan DPI] [--engine auto]

--scan renders each PDF to a degraded raster copy at that DPI first, to test the
scanned-score path on the same material.
"""

import argparse
import re
import zipfile
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
        if n.isRest and n.style.hideObjectOnPrint:
            continue  # invisible voice filler: nothing on the page to read
        if n.isRest:
            p = "R"
        elif n.isChord:
            p = ".".join(sorted(x.nameWithOctave for x in n.pitches))
        else:
            p = n.pitch.nameWithOctave
        lyr = tuple(l.text for l in n.lyrics) if not n.isRest else ()
        out.append((p, float(n.quarterLength), lyr))
    return out


def onset_ratio(pred_part, gt_part, shift=0) -> float:
    """
    Notes per onset, the same however they are split into chords and voices:
    each part becomes a sequence of (pitch, duration) ordered by bar, onset in
    the bar and pitch, and the sequences are matched like the main score.
    """
    def seq(part, sh):
        out = []
        for m in part.getElementsByClass("Measure"):
            here = []
            for n in m.recurse().notes:
                on = round(float(n.getOffsetInHierarchy(m)), 3)
                for pp in n.pitches:
                    here.append((on, pp.midi + sh, round(float(n.quarterLength), 3)))
            out.extend((q[1], q[2]) for q in sorted(here))
        return out
    return ratio(seq(pred_part, 0), seq(gt_part, shift))


def _transpose(name: str, semitones: int) -> str:
    if name == "R":
        return name
    out = []
    for n in name.split("."):
        p = music21.pitch.Pitch(n)
        p.octave += semitones // 12
        out.append(p.nameWithOctave)
    return ".".join(out)


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


def _sung_only(score, n_pred):
    """
    The reference cut to its sung parts (choir mode), when the prediction has
    fewer parts: the parts without any words go, starting from the last.
    """
    if len(score.parts) <= n_pred:
        return score
    wordless = []
    for p in score.parts:
        notes = list(p.recurse().notes)
        if not any(any(l.text and any(c.isalpha() for c in l.text) for l in n.lyrics) for n in notes):
            wordless.append(p)
    drop = wordless[-(len(score.parts) - n_pred):] if len(wordless) >= len(score.parts) - n_pred else wordless
    keep = [p for p in score.parts if p not in drop]
    if keep and len(keep) < len(score.parts):
        for p in list(score.parts):
            if p not in keep:
                score.remove(p)
    return score


def structure(path):
    """
    The repeat structure of a score's first part, in bar order: forward and
    backward repeats, and the start of each volta ending with its numbers.
    Bars carrying nothing are left out, so the list compares across a shifted
    bar numbering.
    """
    tree = etree.parse(path) if not zipfile.is_zipfile(path) else None
    if tree is None:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist()
                        if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF"))
            tree = etree.fromstring(z.read(name))
    root = tree if isinstance(tree, etree._Element) else tree.getroot()
    part = root.find("part")
    out = []
    if part is None:
        return out
    for k, m in enumerate(part.findall("measure")):
        marks = []
        for bl in m.findall("barline"):
            r = bl.find("repeat")
            if r is not None:
                marks.append("|:" if r.get("direction") == "forward" else ":|")
            e = bl.find("ending")
            if e is not None and e.get("type") == "start":
                marks.append("[" + re.sub(r"\s", "", e.get("number", "")))
        for mk in sorted(set(marks)):
            out.append((k, mk))
    return out


def structure_score(pred_path, gt_path) -> str:
    """'found/expected' repeat marks (+extra), matched by kind within a few bars."""
    g, p = structure(gt_path), structure(pred_path)
    used, hit = set(), 0
    def kind(mk):
        # A volta counts by where it is; its label ("1, 2" or "1") is the printed one
        return "[" if mk.startswith("[") else mk
    for k, mk in g:
        j = next((j for j, (k2, m2) in enumerate(p) if j not in used and kind(m2) == kind(mk) and abs(k2 - k) <= 3), None)
        if j is not None:
            used.add(j)
            hit += 1
    extra = len(p) - len(used)
    return f"{hit}/{len(g)}" + (f" +{extra}" if extra else "")


def _sum_frac(xs) -> str:
    hit = tot = extra = 0
    for x in xs:
        m = re.match(r"(\d+)/(\d+)(?: \+(\d+))?", x)
        if m:
            hit += int(m[1]); tot += int(m[2]); extra += int(m[3] or 0)
    return f"{hit}/{tot}" + (f" +{extra}" if extra else "")


def compare(pred_path, gt_path, choir=False):
    pred, gt = music21.converter.parse(pred_path), music21.converter.parse(gt_path)
    if choir:
        gt = _sung_only(gt, len(pred.parts))
    res = {"parts": f"{len(pred.parts)}/{len(gt.parts)}"}
    notes, lyrics, per_part, octave_parts, onsets = [], [], [], [], []
    for i, gp in enumerate(gt.parts):
        if i >= len(pred.parts):
            notes.append(0.0)
            onsets.append(0.0)
            continue
        pt, gtk = tokens(pred.parts[i]), tokens(gp)
        r = ratio([t[:2] for t in pt], [t[:2] for t in gtk])
        # Reference files sometimes write a tenor's octave-treble clef as a plain
        # treble clef: a whole part one octave off is that, not a reading error
        for shift in (12, -12):
            shifted = [(_transpose(t[0], shift), t[1]) for t in gtk]
            r2 = ratio([t[:2] for t in pt], shifted)
            if r2 > max(r, 0.5) + 0.2:
                r = r2
                octave_parts.append(gp.partName or f"P{i + 1}")
        notes.append(r)
        if any(True for _ in gp.recurse().notes):
            onsets.append(max(onset_ratio(pred.parts[i], gp, sh) for sh in (0, 12, -12)))
        # Entries without letters ("-" typed as a lyric) are not lyrics; verse
        # numbers typed into the first syllable ("2. Swift") are not part of it
        def _clean(x):
            return re.sub(r"^\d+[.)]\s*", "", x.replace("\xa0", " ")).strip()
        lg = [_clean(x) for t in gtk for x in t[2] if x and any(c.isalpha() for c in x)]
        lp = [_clean(x) for t in pt for x in t[2] if x and any(c.isalpha() for c in x)]
        if lg:
            lyrics.append(ratio(lp, lg))
        per_part.append((gp.partName or f"P{i + 1}", round(notes[-1], 3), round(ratio(lp, lg), 3) if lg else None))
    res["notes"] = round(min(notes), 3)
    # Same notes however split into chords and voices (worst part)
    res["onsets"] = round(min(onsets), 3) if onsets else 0.0
    def ties(sc):
        # Matched by part, bar, beat in the bar and pitch: a bar's absolute time
        # shifts when an earlier bar is off, the tie itself is still right
        return [(pi, n.getContextByClass("Measure").number, round(float(n.offset), 3),
                 n.pitch.nameWithOctave, n.tie.type)
                for pi, p in enumerate(sc.parts) for n in p.recurse().notes
                if n.isNote and n.tie is not None and n.tie.type in ("start", "stop")]
    def paired(ts):
        # A start without a stop on the next note of that pitch is no tie that
        # can be seen on the page (some editors leave dangling tie starts)
        out = set()
        stops = {(t[0], t[3]) for t in ts if t[4] == "stop"}
        for t in ts:
            if t[4] == "stop" or (t[0], t[3]) in stops:
                out.add(t)
        return out
    tg_, tp_ = paired(ties(gt)), paired(ties(pred))
    # Bar numbers may run a few off from the reference (an incipit or pickup
    # numbered differently): take the shift that matches most ties
    best = tp_
    for shift in (-3, -2, -1, 1, 2, 3):
        moved = {(t[0], t[1] + shift) + t[2:] for t in tp_}
        if len(moved & tg_) > len(best & tg_):
            best = moved
    tp_ = best
    res["ties"] = f"{len(tg_ & tp_)}/{len(tg_)}" + (f" +{len(tp_ - tg_)}" if tp_ - tg_ else "")
    res["lyrics"] = round(min(lyrics), 3) if lyrics else None
    res["repeats"] = structure_score(pred_path, gt_path)
    if octave_parts:
        res["octave_off_in_reference"] = octave_parts
    if os.environ.get("BENCH_PER_PART"):
        res["per_part"] = per_part
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
    ap.add_argument("--keep", help="directory to keep the converted files in")
    ap.add_argument("--choir", action="store_true", help="choir mode (references cut to their sung parts)")
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
            out_dir = a.keep or tmp
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, f"{stem}.musicxml")
            try:
                convert_score(pdf, out, engine=a.engine, use_ai=a.ai, choir=a.choir)
                r = compare(out, gt, choir=a.choir)
            except Exception as e:
                r = {"error": str(e)[:120]}
            rows.append((stem, r))
            print(stem, r, flush=True)
    ok = [r for _, r in rows if "notes" in r]
    if ok:
        print(f"\n{len(ok)}/{len(rows)} converted | mean notes {sum(r['notes'] for r in ok)/len(ok):.3f}"
              f" | mean onsets {sum(r.get('onsets', 0) for r in ok)/len(ok):.3f}"
              f" | perfect notes {sum(r['notes'] == 1 for r in ok)}"
              f" | time ok {sum(r['time'] == 'ok' for r in ok)}"
              f" | valid {sum(bool(r['valid']) for r in ok)}"
              f" | repeats {_sum_frac([r.get('repeats', '0/0') for r in ok])}")


if __name__ == "__main__":
    main()
