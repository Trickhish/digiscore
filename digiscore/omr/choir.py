"""
Choir mode for OMR output: drop the accompaniment parts from a MusicXML string.

The OMR engine names parts only rarely, so a part is dropped when its name is
an instrument, or, in a score whose other parts carry words, when it has none.
"""

from lxml import etree

from digiscore.vector.choir import _INSTRUMENT, _VOICE


def drop_accompaniment(xml: str) -> str:
    try:
        root = etree.fromstring(xml.encode("utf-8"))
    except etree.XMLSyntaxError:
        return xml
    score_parts = root.findall("part-list/score-part")
    parts = {p.get("id"): p for p in root.findall("part")}
    if len(score_parts) < 2:
        return xml

    info = []
    for sp in score_parts:
        pid = sp.get("id")
        name = " ".join(t.strip() for t in sp.xpath("part-name/text()|part-abbreviation/text()"))
        notes = parts[pid].findall(".//note") if pid in parts else []
        pitched = [n for n in notes if n.find("rest") is None]
        words = sum(1 for n in pitched if n.find("lyric/text") is not None)
        share = words / len(pitched) if pitched else 0.0
        info.append((pid, name, share))

    any_words = max(s for _, _, s in info) >= 0.2
    drop = set()
    for pid, name, share in info:
        if name and _INSTRUMENT.search(name) and not _VOICE.search(name):
            drop.add(pid)
        # OCR of the page text picks up stray words: a part with far fewer
        # words than the sung ones is accompaniment
        elif any_words and share < 0.06 and not (name and _VOICE.search(name)):
            drop.add(pid)
    if not drop or len(drop) == len(info):
        return xml

    plist = root.find("part-list")
    for sp in score_parts:
        if sp.get("id") in drop:
            plist.remove(sp)
    for pid in drop:
        if pid in parts:
            root.remove(parts[pid])
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          doctype=root.getroottree().docinfo.doctype or None).decode("utf-8")
