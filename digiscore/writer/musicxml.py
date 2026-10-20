"""
MusicXML 4.0 Exporter using lxml.
Produces clean, fully compliant MusicXML compatible with MuseScore, Dorico, Finale, Sibelius.
"""

from datetime import date
from typing import Dict, List, Optional
from lxml import etree


def build_musicxml(
    score_data: dict,
    divisions: int = 8,  # 8 divisions per quarter note (supports up to 32nd notes and triplets)
) -> str:
    """
    Constructs a valid MusicXML 4.0 string from the internal score representation.
    """
    root = etree.Element("score-partwise", version="4.0")

    # Work / Title
    title = score_data.get("title", "")
    if title:
        work = etree.SubElement(root, "work")
        wt = etree.SubElement(work, "work-title")
        wt.text = title

    # Identification
    ident = etree.SubElement(root, "identification")
    composer = score_data.get("composer", "")
    if composer:
        c_elem = etree.SubElement(ident, "creator", type="composer")
        c_elem.text = composer
    lyricist = score_data.get("lyricist", "")
    if lyricist:
        l_elem = etree.SubElement(ident, "creator", type="lyricist")
        l_elem.text = lyricist

    enc = etree.SubElement(ident, "encoding")
    soft = etree.SubElement(enc, "software")
    soft.text = "digiScore"
    edate = etree.SubElement(enc, "encoding-date")
    edate.text = str(date.today())

    # Part list
    part_list_elem = etree.SubElement(root, "part-list")
    parts = score_data.get("parts", [])
    if not parts:
        parts = [{"part_id": "P1", "name": "Voice", "measures": []}]

    for idx, p in enumerate(parts):
        pid = p.get("part_id", f"P{idx + 1}")
        pname = p.get("name", f"Part {idx + 1}")
        sp = etree.SubElement(part_list_elem, "score-part", id=pid)
        pn = etree.SubElement(sp, "part-name")
        pn.text = pname

    # Part contents
    for idx, p in enumerate(parts):
        pid = p.get("part_id", f"P{idx + 1}")
        part_elem = etree.SubElement(root, "part", id=pid)

        measures = p.get("measures", [])
        prev_sig = None
        for m_idx, m in enumerate(measures):
            m_num = str(m.get("number", m_idx + 1))
            meas_elem = etree.SubElement(part_elem, "measure", number=m_num)

            sig = (
                m.get("key_fifths"), m.get("time_beats"), m.get("time_beat_type"),
                m.get("clef_sign"), m.get("clef_line"), m.get("clef_octave"),
            )
            changed = sig != prev_sig
            prev_sig = sig

            # Measure attributes (key, time, clef) on first measure or when changed
            if m_idx == 0 or changed or m.get("attributes"):
                attrs = etree.SubElement(meas_elem, "attributes")
                div_elem = etree.SubElement(attrs, "divisions")
                div_elem.text = str(divisions)

                # Key signature
                fifths = m.get("key_fifths", score_data.get("key_fifths", 0))
                key_elem = etree.SubElement(attrs, "key")
                f_elem = etree.SubElement(key_elem, "fifths")
                f_elem.text = str(fifths)

                # Time signature
                beats = m.get("time_beats", score_data.get("time_beats", 4))
                beat_type = m.get("time_beat_type", score_data.get("time_beat_type", 4))
                time_elem = etree.SubElement(attrs, "time")
                b_elem = etree.SubElement(time_elem, "beats")
                b_elem.text = str(beats)
                bt_elem = etree.SubElement(time_elem, "beat-type")
                bt_elem.text = str(beat_type)

                # Clef
                clef_sign = m.get("clef_sign", p.get("clef_sign", "G"))
                clef_line = m.get("clef_line", p.get("clef_line", 2))
                clef_octave = m.get("clef_octave", p.get("clef_octave", 0))

                clef_elem = etree.SubElement(attrs, "clef")
                cs_elem = etree.SubElement(clef_elem, "sign")
                cs_elem.text = clef_sign
                cl_elem = etree.SubElement(clef_elem, "line")
                cl_elem.text = str(clef_line)
                if clef_octave != 0:
                    co_elem = etree.SubElement(clef_elem, "clef-octave-change")
                    co_elem.text = str(clef_octave)

            # Direction (Tempo mark) on first measure if available
            tempo = score_data.get("tempo")
            if m_idx == 0 and tempo:
                direction = etree.SubElement(meas_elem, "direction", placement="above")
                dir_type = etree.SubElement(direction, "direction-type")
                met = etree.SubElement(dir_type, "metronome")
                bu = etree.SubElement(met, "beat-unit")
                bu.text = "quarter"
                pm = etree.SubElement(met, "per-minute")
                pm.text = str(tempo)
                sound = etree.SubElement(direction, "sound", tempo=str(tempo))

            # Notes: voices of events (an event with several notes is a chord)
            voices = m.get("voices")
            if voices is None:
                voices = [[[n] for n in m.get("notes", [])]]
            voices = [v for v in voices if v]

            for v_idx, events in enumerate(voices):
                if v_idx > 0:
                    prev_len = sum(
                        max(1, int(round(ev[0].get("duration_quarters", 1.0) * divisions)))
                        for ev in voices[v_idx - 1]
                    )
                    backup = etree.SubElement(meas_elem, "backup")
                    etree.SubElement(backup, "duration").text = str(prev_len)
                for event in events:
                    # Chord notes are written bottom to top; lyrics go on the first note
                    ordered = sorted(event, key=lambda n: -n.get("y", 0))
                    lyr_note = next((n for n in ordered if n.get("lyrics")), ordered[0])
                    for c_idx, note in enumerate(ordered):
                        _write_note(
                            meas_elem, note, divisions, v_idx + 1,
                            is_chord=c_idx > 0,
                            lyrics=lyr_note.get("lyrics", []) if c_idx == 0 else [],
                        )

            # End barline (repeats)
            end_bl = m.get("end_barline")
            if end_bl and end_bl.get("repeat_direction") == "backward":
                bl_elem = etree.SubElement(meas_elem, "barline", location="right")
                etree.SubElement(bl_elem, "bar-style").text = "light-heavy"
                etree.SubElement(bl_elem, "repeat", direction="backward")

    xml_bytes = etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
        doctype='<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
    )
    return xml_bytes.decode("utf-8")


def _write_note(meas_elem, note: dict, divisions: int, voice: int, is_chord: bool, lyrics: list) -> None:
    note_elem = etree.SubElement(meas_elem, "note")
    if note.get("hidden"):
        note_elem.set("print-object", "no")
    is_rest = note.get("is_rest", False)
    if is_chord:
        etree.SubElement(note_elem, "chord")

    if is_rest:
        etree.SubElement(note_elem, "rest")
    else:
        pitch_elem = etree.SubElement(note_elem, "pitch")
        step_elem = etree.SubElement(pitch_elem, "step")
        step_elem.text = note.get("step", "C")
        alter = note.get("alter", 0)
        if alter != 0:
            alt_elem = etree.SubElement(pitch_elem, "alter")
            alt_elem.text = str(alter)
        oct_elem = etree.SubElement(pitch_elem, "octave")
        oct_elem.text = str(note.get("octave", 4))

    # Duration in divisions
    quarters = note.get("duration_quarters", 1.0)
    dur_divs = max(1, int(round(quarters * divisions)))
    d_elem = etree.SubElement(note_elem, "duration")
    d_elem.text = str(dur_divs)

    v_elem = etree.SubElement(note_elem, "voice")
    v_elem.text = str(voice)

    ntype = note.get("type", "quarter").replace(" (dotted)", "")
    t_elem = etree.SubElement(note_elem, "type")
    t_elem.text = ntype

    dot_count = note.get("dot_count", 0)
    for _ in range(dot_count):
        etree.SubElement(note_elem, "dot")

    acc = note.get("accidental")
    if acc:
        acc_elem = etree.SubElement(note_elem, "accidental")
        acc_elem.text = acc

    # Lyrics
    for lyr in sorted(lyrics, key=lambda l: l.get("verse", 1)):
        lyr_elem = etree.SubElement(note_elem, "lyric", number=str(lyr.get("verse", 1)))
        syl_elem = etree.SubElement(lyr_elem, "syllabic")
        syl_elem.text = lyr.get("syllabic", "single")
        txt_elem = etree.SubElement(lyr_elem, "text")
        txt_elem.text = lyr.get("text", "")
        if lyr.get("has_extender"):
            etree.SubElement(lyr_elem, "extend")
