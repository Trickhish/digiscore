"""
Choir mode for OMR output: drop the accompaniment parts from a MusicXML string.

The OMR engine names parts only rarely, so a part is dropped when its name is
an instrument, or, in a score whose other parts carry words, when it has none.
"""

from lxml import etree

from digiscore.vector.choir import _INSTRUMENT, _VOICE, _VOICE_PAIR


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
        texts = [t.strip() for t in sp.xpath("part-name/text()|part-abbreviation/text()") if t.strip()]
        name = texts[0] if texts else ""
        if any(_VOICE_PAIR.match(t) for t in texts):
            name = next(t for t in texts if _VOICE_PAIR.match(t))
        elif len(texts) > 1 and not _VOICE.search(name):
            name = " ".join(texts)
        notes = parts[pid].findall(".//note") if pid in parts else []
        pitched = [n for n in notes if n.find("rest") is None]
        words = sum(1 for n in pitched if n.find("lyric/text") is not None)
        share = words / len(pitched) if pitched else 0.0
        info.append((pid, name, share))

    any_words = max(s for _, _, s in info) >= 0.2
    drop = set()
    def voiced(name):
        return bool(name) and (_VOICE.search(name) or _VOICE_PAIR.match(name))

    for pid, name, share in info:
        if voiced(name):
            continue
        if name and _INSTRUMENT.search(name):
            drop.add(pid)
        # OCR of the page text picks up stray words: a part with far fewer
        # words than the sung ones is accompaniment
        elif any_words and share < 0.06:
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
