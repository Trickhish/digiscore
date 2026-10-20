"""
Reader for the NoteLite/Audiveris project file (.omr).

The .omr archive keeps the geometry the MusicXML export throws away: where each
chord and each lyric line sits on the page image. That geometry lets us redo the
lyrics with a better OCR and attach them to the right notes.
"""

import re
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lxml import etree


@dataclass
class Chord:
    id: str
    x0: float
    x1: float
    y0: float
    y1: float
    voice: int
    slot: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class LyricLine:
    id: str
    verse: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class StaffMeasure:
    """One staff's content in one measure (stack)."""
    measure_index: int          # global measure index in the score (0-based)
    left: float
    right: float
    chords: List[Chord] = field(default_factory=list)  # in time order


@dataclass
class StaffInSystem:
    staff_id: str
    part_id: str                # MusicXML part id ("P<logical-id>")
    top: float = 0.0
    bottom: float = 0.0
    interline: float = 20.0
    measures: List[StaffMeasure] = field(default_factory=list)
    lyric_lines: List[LyricLine] = field(default_factory=list)


@dataclass
class SystemInfo:
    staves: List[StaffInSystem] = field(default_factory=list)


@dataclass
class SheetInfo:
    number: int                 # 1-based sheet number = PDF page number
    width: int
    height: int
    systems: List[SystemInfo] = field(default_factory=list)


def _bounds(elem) -> Optional[Tuple[float, float, float, float]]:
    """(x0, x1, y0, y1)"""
    b = elem.find("bounds")
    if b is None:
        return None
    x, y, w, h = (float(b.get(k)) for k in ("x", "y", "w", "h"))
    return x, x + w, y, y + h


def read_book(omr_path: str) -> List[SheetInfo]:
    z = zipfile.ZipFile(omr_path)
    book = etree.fromstring(z.read("book.xml"))

    # Logical part ids per sheet / system / part position
    logical: Dict[int, List[List[str]]] = {}
    for sheet in book.findall("sheet"):
        num = int(sheet.get("number"))
        systems = []
        for page in sheet.findall("page"):
            for sysm in page.findall("system"):
                systems.append([p.get("logical-id") for p in sysm.findall("part")])
        logical[num] = systems

    sheets: List[SheetInfo] = []
    measure_base = 0
    names = sorted(
        (n for n in z.namelist() if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", n)),
        key=lambda n: int(re.search(r"#(\d+)/", n).group(1)),
    )
    for name in names:
        num = int(re.search(r"#(\d+)/", name).group(1))
        root = etree.fromstring(z.read(name))
        pic = root.find("picture")
        info = SheetInfo(number=num, width=int(pic.get("width")), height=int(pic.get("height")))
        interline = float(root.find("scale/interline").get("main"))

        for s_idx, sysm in enumerate(root.iter("system")):
            # All inters of the system by id
            inters = {}
            for container in sysm.findall("sig/inters"):
                for elem in container:
                    if elem.get("id") is not None:
                        inters[elem.get("id")] = elem

            stacks = sysm.findall("stack")
            stack_pos = {st.get("id"): i for i, st in enumerate(stacks)}
            sys_logical = logical.get(num, [])
            part_ids = sys_logical[s_idx] if s_idx < len(sys_logical) else []

            system = SystemInfo()
            for p_idx, part in enumerate(sysm.findall("part")):
                staff_elem = part.find("staff")
                if staff_elem is None:
                    continue
                lid = part_ids[p_idx] if p_idx < len(part_ids) else None
                staff = StaffInSystem(
                    staff_id=staff_elem.get("id"),
                    part_id=f"P{lid}" if lid else "",
                    interline=interline,
                )
                ys = [float(pt.get("y")) for pt in staff_elem.iter("point")]
                if ys:
                    staff.top, staff.bottom = min(ys), max(ys)

                for meas in part.findall("measure"):
                    pos = stack_pos.get(meas.get("id"))
                    if pos is None:
                        continue
                    st = stacks[pos]
                    sm = StaffMeasure(
                        measure_index=measure_base + pos,
                        left=float(st.get("left", 0)),
                        right=float(st.get("right", 0)),
                    )
                    for voice in meas.findall("voice"):
                        v_id = int(voice.get("id"))
                        for entry in voice.iter("entry"):
                            val = entry.find("value")
                            if val is None or val.get("status") != "BEGIN":
                                continue
                            hc = inters.get(val.get("chord"))
                            if hc is None or hc.tag != "head-chord":
                                continue
                            bb = _bounds(hc)
                            if bb is None:
                                continue
                            sm.chords.append(Chord(hc.get("id"), *bb, voice=v_id, slot=int(entry.findtext("key"))))
                    sm.chords.sort(key=lambda c: (c.voice, c.slot))
                    staff.measures.append(sm)

                for elem in inters.values():
                    if elem.tag == "lyric-line" and elem.get("staff") == staff.staff_id:
                        bb = _bounds(elem)
                        if bb:
                            x0, x1, y0, y1 = bb
                            staff.lyric_lines.append(
                                LyricLine(elem.get("id"), int(elem.get("number", 1)), x0, y0, x1, y1)
                            )
                staff.lyric_lines.sort(key=lambda l: (l.verse, l.y0))
                system.staves.append(staff)
            info.systems.append(system)
            measure_base += len(stacks)
        sheets.append(info)
    return sheets
