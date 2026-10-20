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


def inject_time_into_omr(omr_path: str, time_sig: str) -> bool:
    """Patch every staff header without a time with a time-pair inter. Returns True if patched."""
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

    for name in names:
        root = etree.fromstring(out_entries[name])
        nid = int(root.get("last-persistent-id"))
        for sysm in root.iter("system"):
            inters = sysm.find("sig/inters")
            if inters is None:
                continue
            for part in sysm.findall("part"):
                for st in part.findall("staff"):
                    h = st.find("header")
                    if h is None or h.find("time") is not None:
                        continue
                    ys = [float(q.get("y")) for q in st.iter("point")]
                    if not ys:
                        continue
                    stop = int(h.get("stop"))
                    nid += 1
                    e = etree.SubElement(
                        inters, "time-pair",
                        **{"time-rational": time_sig},
                        grade="0.9", frozen="true",
                        staff=st.get("id"), id=str(nid),
                    )
                    e.set("ctx-grade", "0.9")
                    etree.SubElement(e, "bounds", x=str(stop + 5),
                                     y=str(int(min(ys))), w="40",
                                     h=str(int(max(ys) - min(ys))))
                    te = etree.SubElement(h, "time")
                    te.text = str(nid)
                    h.set("stop", str(stop + 50))
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
