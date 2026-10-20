"""Fast tests for OMR post-processing that do not need NoteLite."""

from lxml import etree

from digiscore.omr.notelite import join_movements
from digiscore.omr.postprocess import merge_split_parts


def _score(parts):
    """parts: list of (name, [bar contents]) where a bar is 'N' (note) or 'R' (rest)."""
    pl = "".join(
        f'<score-part id="P{i + 1}"><part-name>{name}</part-name></score-part>'
        for i, (name, _) in enumerate(parts)
    )
    body = ""
    for i, (_, bars) in enumerate(parts):
        ms = ""
        for k, b in enumerate(bars):
            if b == "N":
                note = "<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note>"
            else:
                note = '<note><rest measure="yes"/><duration>4</duration><voice>1</voice></note>'
            attrs = "<attributes><divisions>1</divisions></attributes>" if k == 0 else ""
            ms += f'<measure number="{k + 1}">{attrs}{note}</measure>'
        body += f'<part id="P{i + 1}">{ms}</part>'
    return f'<?xml version="1.0" encoding="UTF-8"?><score-partwise version="4.0"><part-list>{pl}</part-list>{body}</score-partwise>'


def _usage(xml):
    root = etree.fromstring(xml.encode())
    return [
        "".join("N" if any(n.find("rest") is None for n in m.findall("note")) else "R" for m in p.findall("measure"))
        for p in root.findall("part")
    ]


def test_sparse_parts_merge_into_main_parts():
    # 3 sparse parts stand in for the 3 main parts in bars 3-4 only
    xml = _score([
        ("Voice", "RRNN" + "R" * 6), ("Voice", "RRNN" + "R" * 6), ("Voice", "RRNN" + "R" * 6),
        ("Voice", "NNRRNNNNNN"), ("Voice", "NNRRNNNNNN"), ("Voice", "NNRRNNNNNN"),
    ])
    out = merge_split_parts(xml)
    assert _usage(out) == ["NNNNNNNNNN"] * 3


def test_join_movements_keeps_parts_apart_with_identical_names():
    a = _score([("Voice", "NN"), ("Voice", "NR")])
    b = _score([("Voice", "RN"), ("Voice", "NN")])
    out = join_movements([a, b])
    assert _usage(out) == ["NNRN", "NRNN"]
    root = etree.fromstring(out.encode())
    assert [m.get("number") for m in root.find("part").findall("measure")] == ["1", "2", "3", "4"]


def test_cross_staff_realign_extends_short_note():
    from digiscore.omr.align import realign_from_other_staves
    from digiscore.omr.book import Chord, SheetInfo, StaffInSystem, StaffMeasure, SystemInfo

    def note(step, dur, typ):
        return (f"<note><pitch><step>{step}</step><octave>4</octave></pitch>"
                f"<duration>{dur}</duration><voice>1</voice><type>{typ}</type></note>")

    attrs = "<attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
    # Staff 1 is right (4 quarters); staff 2 read its first half note as a quarter
    p1 = attrs + "".join(note(s, 2, "quarter") for s in "CDEF")
    p2 = attrs + note("A", 2, "quarter") + note("B", 2, "quarter") + note("C", 2, "quarter")
    xml = ('<?xml version="1.0" encoding="UTF-8"?><score-partwise version="4.0"><part-list>'
           '<score-part id="P1"><part-name>S</part-name></score-part>'
           '<score-part id="P2"><part-name>A</part-name></score-part></part-list>'
           f'<part id="P1"><measure number="1">{p1}</measure></part>'
           f'<part id="P2"><measure number="1">{p2}</measure></part></score-partwise>')

    def staff(pid, xs):
        chords = [Chord(f"{pid}{i}", x - 5, x + 5, 0, 10, voice=1, slot=i + 1) for i, x in enumerate(xs)]
        return StaffInSystem("1", pid, 0, 100, 20.0, [StaffMeasure(0, 0, 500, chords)])

    sheets = [SheetInfo(1, 1000, 1000, [SystemInfo([staff("P1", [100, 200, 300, 400]), staff("P2", [100, 300, 400])])])]
    out, n = realign_from_other_staves(xml, sheets)
    assert n == 1
    alto = etree.fromstring(out.encode()).findall("part")[1]
    assert [x.findtext("duration") for x in alto.iter("note")] == ["4", "2", "2"]
    assert alto.find(".//note").findtext("type") == "half"
