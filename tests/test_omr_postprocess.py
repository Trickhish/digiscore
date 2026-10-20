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
