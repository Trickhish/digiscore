"""
Inject a time signature into an .omr project when the OMR engine did not find one.

The common and cut-time symbols are often missed by Audiveris/NoteLite on scans,
but the AI can read them from the image. This module:
  1. reads the .omr project,
  2. asks the vision model for the time signature of the first system,
  3. writes a <time-pair> inter into every staff header that has no time, and
  4. strips the RHYTHMS+PAGE steps so the engine re-does them on the next run.
"""

import io
import os
import re
import zipfile

from lxml import etree


def detect_time_signature(pdf_path: str, vision) -> str | None:
    """Ask the AI for the time signature; returns 'beats/beat-type' or None."""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    if not doc:
        return None
    page = doc[0]
    # Crop a strip across the top-left showing the first system header
    clip = pymupdf.Rect(page.rect.width * 0.04, page.rect.height * 0.06,
                        page.rect.width * 0.45, page.rect.height * 0.35)
    png = page.get_pixmap(dpi=300, clip=clip).tobytes("png")
    import base64
    prompt = (
        "This is the first system of a music score. What is the time signature? "
        "Answer with ONLY the fraction, e.g. 4/4, 3/4, 6/8, 2/2, 2/4. "
        "If it is a common-time C symbol answer 4/4; if it is a cut-time (¢) symbol answer 2/2."
    )
    try:
        raw = vision._chat_completion(
            [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            ]}],
            max_tokens=20,
        )
    except Exception:
        return None
    m = re.search(r"(\d+)/(\d+)", raw)
    return m.group(0) if m else None


def guess_time_from_durations(xml_content: str) -> str | None:
    """
    The bar length that most measures already fill, as a time signature.
    Without a time signature the OMR engine still assigns durations; most bars
    come out right, so the most common length is the real one.
    """
    from collections import Counter
    from fractions import Fraction

    root = etree.fromstring(xml_content.encode("utf-8"))
    counts: Counter = Counter()
    for part in root.findall("part"):
        div = 1
        for m in part.findall("measure"):
            d = m.findtext("attributes/divisions")
            if d:
                div = int(d)
            voices: dict = {}
            for n in m.findall("note"):
                if n.find("chord") is not None or n.find("grace") is not None:
                    continue
                if n.find("rest") is not None and n.find("rest").get("measure") == "yes":
                    voices = {}
                    break
                v = n.findtext("voice") or "1"
                voices[v] = voices.get(v, Fraction(0)) + Fraction(int(n.findtext("duration") or 0), div)
            if voices:
                counts[max(voices.values())] += 1
    if not counts:
        return None
    length, n = counts.most_common(1)[0]
    if n < 0.3 * sum(counts.values()):
        return None
    # Same bar length, different beaming meaning (2/2 vs 4/4, 6/8 vs 3/4) cannot
    # be told apart from durations; the length is what the rhythm check needs
    table = {
        Fraction(4): "4/4", Fraction(3): "3/4", Fraction(2): "2/4",
        Fraction(3, 2): "3/8", Fraction(6): "6/4", Fraction(5): "5/4",
    }
    return table.get(length)


def omr_time_signatures(omr_path: str) -> set:
    """The time-rational values of the time signatures in the first system's headers."""
    found = set()
    with zipfile.ZipFile(omr_path) as z:
        names = sorted((n for n in z.namelist() if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", n)),
                       key=lambda n: int(re.search(r"#(\d+)/", n).group(1)))
        if not names:
            return found
        root = etree.fromstring(z.read(names[0]))
    sysm = next(root.iter("system"), None)
    if sysm is None:
        return found
    inters = {e.get("id"): e for c in sysm.findall("sig/inters") for e in c}
    for st in sysm.iter("staff"):
        h = st.find("header")
        ref = h.findtext("time") if h is not None else None
        e = inters.get(ref) if ref else None
        if e is not None and e.get("time-rational"):
            found.add(e.get("time-rational"))
    return found


def _inside(b, outer) -> bool:
    if b is None or outer is None:
        return False
    x, y, w, h = (float(b.get(k)) for k in ("x", "y", "w", "h"))
    X, Y, W, H = (float(outer.get(k)) for k in ("x", "y", "w", "h"))
    return X - 2 <= x and Y - 2 <= y and x + w <= X + W + 2 and y + h <= Y + H + 2


def replace_header_times(omr_path: str, time_sig: str, spans: dict | None = None) -> bool:
    """
    Remove the header time signatures (and their references) from every staff,
    then inject time_sig: used when the engine misread the signature.
    """
    with zipfile.ZipFile(omr_path) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    for name in [n for n in entries if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", n)]:
        root = etree.fromstring(entries[name])
        for sysm in root.iter("system"):
            gone = set()
            for st in sysm.iter("staff"):
                h = st.find("header")
                t = h.find("time") if h is not None else None
                if t is not None:
                    gone.add(t.text)
                    h.remove(t)
            if not gone:
                continue
            for c in sysm.findall("sig/inters"):
                # The pair's digits (time-number) are separate inters of the same box
                boxes = [e.find("bounds") for e in c if e.get("id") in gone and e.find("bounds") is not None]
                for e in list(c):
                    if e.get("id") in gone:
                        c.remove(e)
                    elif e.tag == "time-number" and any(_inside(e.find("bounds"), b) for b in boxes):
                        gone.add(e.get("id"))
                        c.remove(e)
            for e in list(sysm.iter("times")):
                if e.text in gone:
                    e.getparent().remove(e)
            # Relations pointing at removed inters
            for rel in list(sysm.iter("relation")):
                if rel.get("source") in gone or rel.get("target") in gone:
                    rel.getparent().remove(rel)
        entries[name] = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    with zipfile.ZipFile(omr_path, "w", zipfile.ZIP_STORED) as zw:
        for n, data in entries.items():
            zw.writestr(n, data)
    return inject_time_into_omr(omr_path, time_sig, spans)


def _drop_chords_in_spans(sysm, spans: dict) -> None:
    """Remove head-chords (and their heads and stems) lying inside a signature's span."""
    inters = {e.get("id"): e for c in sysm.findall("sig/inters") for e in c}
    gone = set()
    for e in inters.values():
        if e.tag != "head-chord":
            continue
        span = spans.get(e.get("staff"))
        b = e.find("bounds")
        if span is None or b is None:
            continue
        x, w = float(b.get("x")), float(b.get("w"))
        if span[0] - 5 <= x and x + w <= span[1] + 10:
            gone.add(e.get("id"))
    if not gone:
        return
    # Members of the chords: heads and stems in the same span
    for e in inters.values():
        if e.tag in ("head", "stem"):
            b = e.find("bounds")
            span = spans.get(e.get("staff"))
            if b is not None and span and span[0] - 5 <= float(b.get("x")) and \
                    float(b.get("x")) + float(b.get("w")) <= span[1] + 10:
                gone.add(e.get("id"))
    for c in sysm.findall("sig/inters"):
        for e in list(c):
            if e.get("id") in gone:
                c.remove(e)
    for rel in list(sysm.iter("relation")):
        if rel.get("source") in gone or rel.get("target") in gone:
            rel.getparent().remove(rel)
    for tag in ("head-chords", "stems"):
        for e in sysm.iter(tag):
            ids = [i for i in (e.text or "").split() if i not in gone]
            e.text = " ".join(ids)
    # Voice slots that pointed at a removed chord
    for entry in list(sysm.iter("entry")):
        v = entry.find("value")
        if v is not None and v.get("chord") in gone:
            entry.getparent().remove(entry)


def inject_time_into_omr(omr_path: str, time_sig: str, spans: dict | None = None) -> bool:
    """
    Patch every staff header without a time with a time-pair inter. Returns True if patched.
    spans: {staff id: (x0, x1)} of the signature on the first sheet's first
    system; chords the engine built from its digits there are removed.
    """
    z = zipfile.ZipFile(omr_path, "r")
    names = sorted(n for n in z.namelist() if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", n))
    book_xml = z.read("book.xml")
    if not names:
        z.close()
        return False

    changed = False
    out_entries: dict[str, bytes] = {}
    for name in z.namelist():
        out_entries[name] = z.read(name)

    first_sheet = min(names, key=lambda n: int(re.search(r"#(\d+)/", n).group(1)))
    for name in names:
        root = etree.fromstring(out_entries[name])
        nid = int(root.get("last-persistent-id"))
        for s_idx, sysm in enumerate(root.iter("system")):
            inters = sysm.find("sig/inters")
            if inters is None:
                continue
            if spans and name == first_sheet and s_idx == 0:
                _drop_chords_in_spans(sysm, spans)
            for part in sysm.findall("part"):
                for st in part.findall("staff"):
                    h = st.find("header")
                    if h is None or h.find("time") is not None:
                        continue
                    ys = [float(q.get("y")) for q in st.iter("point")]
                    if not ys:
                        continue
                    stop = int(h.get("stop"))
                    span = spans.get(st.get("id")) if spans and name == first_sheet and s_idx == 0 else None
                    x_at, w_at = (int(span[0]), int(span[1] - span[0])) if span else (stop + 5, 40)
                    nid += 1
                    e = etree.SubElement(
                        inters, "time-pair",
                        **{"time-rational": time_sig},
                        grade="0.9", frozen="true",
                        staff=st.get("id"), id=str(nid),
                    )
                    e.set("ctx-grade", "0.9")
                    etree.SubElement(e, "bounds", x=str(x_at),
                                     y=str(int(min(ys))), w=str(w_at),
                                     h=str(int(max(ys) - min(ys))))
                    te = etree.SubElement(h, "time")
                    te.text = str(nid)
                    h.set("stop", str(max(stop, x_at + w_at) + 10))
                    m = part.find("measure")
                    if m is not None:
                        tm = etree.Element("times")
                        tm.text = str(nid)
                        anchor = None
                        for tag in ("keys", "clefs", "right-barline", "mid-barline"):
                            if m.find(tag) is not None:
                                anchor = m.find(tag)
                                break
                        if anchor is not None:
                            anchor.addnext(tm)
                        else:
                            m.insert(0, tm)
                    changed = True
        root.set("last-persistent-id", str(nid))
        out_entries[name] = etree.tostring(root, xml_declaration=True, encoding="UTF-8")

    if changed:
        # Strip RHYTHMS PAGE from the step list so NoteLite re-runs them
        bk = book_xml.decode("utf-8")
        bk = re.sub(r" RHYTHMS PAGE</steps>", "</steps>", bk)
        bk = re.sub(r" RHYTHMS</steps>", "</steps>", bk)
        bk = re.sub(r" PAGE</steps>", "</steps>", bk)
        out_entries["book.xml"] = bk.encode("utf-8")

        z.close()
        with zipfile.ZipFile(omr_path, "w", compression=zipfile.ZIP_STORED) as zw:
            for n, data in out_entries.items():
                zw.writestr(n, data)
    else:
        z.close()
    return changed


def apply_time_signature(xml_content: str, time_sig: str, symbol: str | None = None) -> str:
    """
    Write the time signature read from the image into the first measure of every
    part, replacing the engine's when it has the same bar length (4/4 vs 2/2,
    3/4 vs 6/8) or adding it when there is none and most bars fill it. A
    different bar length means one of the two readings is wrong: the engine's
    stays, since its rhythm was built on it.
    """
    from fractions import Fraction

    beats, beat_type = (int(x) for x in time_sig.split("/"))
    want = Fraction(beats, beat_type)
    root = etree.fromstring(xml_content.encode("utf-8"))
    changed = False
    for part in root.findall("part"):
        m = part.find("measure")
        if m is None:
            continue
        # Later restatements with the same bar length (the engine repeats the
        # injected signature at system starts) carry no information: drop them
        prev = None
        for mm in part.findall("measure")[1:]:
            t = mm.find("attributes/time")
            if t is None:
                continue
            try:
                f = Fraction(int(t.findtext("beats")), int(t.findtext("beat-type")))
            except (TypeError, ValueError):
                continue
            if f == (prev if prev is not None else want):
                t.getparent().remove(t)
                changed = True
            else:
                prev = f
        attrs = m.find("attributes")
        cur = attrs.find("time") if attrs is not None else None
        if cur is not None:
            try:
                have = Fraction(int(cur.findtext("beats")), int(cur.findtext("beat-type")))
            except (TypeError, ValueError):
                continue
            if have != want:
                continue
        else:
            guess = guess_time_from_durations(xml_content)
            if not guess or Fraction(*map(int, guess.split("/"))) != want:
                continue
            if attrs is None:
                attrs = etree.Element("attributes")
                m.insert(0, attrs)
                etree.SubElement(attrs, "divisions").text = "1"
            cur = etree.Element("time")
            # <time> goes after <divisions> and <key>
            anchor = attrs.find("key") if attrs.find("key") is not None else attrs.find("divisions")
            if anchor is not None:
                anchor.addnext(cur)
            else:
                attrs.insert(0, cur)
        for c in list(cur):
            cur.remove(c)
        etree.SubElement(cur, "beats").text = str(beats)
        etree.SubElement(cur, "beat-type").text = str(beat_type)
        if symbol in ("common", "cut"):
            cur.set("symbol", symbol)
        elif "symbol" in cur.attrib:
            del cur.attrib["symbol"]
        changed = True
    if not changed:
        return xml_content
    return etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True,
                          encoding="UTF-8").decode("utf-8")
