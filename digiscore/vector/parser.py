"""
Full vector score parser.
Extracts staves, systems, clefs, notes, measures, and lyrics from vector PDF pages.
"""

from typing import Dict, List, Optional, Tuple
import pymupdf

from digiscore.smufl import get_glyph_name, is_clef, is_notehead, is_rest
from digiscore.vector.duration import resolve_note_durations
from digiscore.vector.lyrics import extract_lyrics_for_staff, match_syllables_to_notes
from digiscore.vector.measure import create_measures_for_staff, extract_barlines_for_staff
from digiscore.vector.pitch import calculate_pitch_from_y
from digiscore.vector.staff import Staff, System, detect_staves_on_page, group_staves_into_systems


def parse_vector_score(doc: pymupdf.Document) -> dict:
    """
    Parses a vector music PDF and returns the full score dictionary.
    """
    all_systems: List[System] = []
    page_data = []

    # 1. Extract staves and page elements per page
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        staves = detect_staves_on_page(page, page_idx)
        drawings = page.get_drawings()
        systems = group_staves_into_systems(staves, drawings)
        all_systems.extend(systems)

        raw = page.get_text("rawdict")
        words = page.get_text("words")

        # Extract SMuFL / Opus glyphs
        glyphs = []
        for block in raw.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        font = span.get("font", "")
                        for ch in span.get("chars", []):
                            gname = get_glyph_name(ch["c"], font)
                            if gname:
                                glyphs.append({
                                    "name": gname,
                                    "char": ch["c"],
                                    "font": font,
                                    "x": ch["origin"][0],
                                    "y": ch["origin"][1],
                                    "bbox": ch["bbox"],
                                })

        # Extract beams: drawings with height between 1.5 and 8 pt and width between 8 and 160 pt
        beams = []
        for d in drawings:
            r = d["rect"]
            if 8.0 < r.width < 160.0 and 1.5 <= r.height <= 8.5:
                beams.append({
                    "x0": r.x0,
                    "x1": r.x1,
                    "y0": r.y0,
                    "y1": r.y1,
                })

        page_data.append({
            "page_idx": page_idx,
            "staves": staves,
            "systems": systems,
            "glyphs": glyphs,
            "beams": beams,
            "words": words,
            "drawings": drawings,
        })

    if not all_systems:
        return {}

    # Determine number of parts across systems
    part_count = max(len(sys.staves) for sys in all_systems)
    
    # Standard choral part names
    default_names = ["Soprano", "Alto", "Tenor", "Bass"] if part_count == 4 else [f"Voice {i+1}" for i in range(part_count)]
    if part_count == 1:
        default_names = ["Voice"]

    parts = [{"part_id": f"P{p+1}", "name": default_names[p] if p < len(default_names) else f"Part {p+1}", "measures": []} for p in range(part_count)]

    # Global key & time sig defaults
    score_key_fifths = 0
    score_time_beats = 4
    score_time_beat_type = 4
    score_tempo = None

    # Track measure numbers across systems for each part
    part_measure_counters = [1] * part_count

    for p_info in page_data:
        p_idx = p_info["page_idx"]
        glyphs = p_info["glyphs"]
        beams = p_info["beams"]
        words = p_info["words"]
        drawings = p_info["drawings"]

        # Separate glyph types
        noteheads = [g for g in glyphs if is_notehead(g["name"])]
        rests = [g for g in glyphs if is_rest(g["name"])]
        clefs = [g for g in glyphs if is_clef(g["name"])]
        accidentals = [g for g in glyphs if "accidental" in g["name"].lower()]
        flags = [g for g in glyphs if "flag" in g["name"].lower()]
        dots = [g for g in glyphs if "dot" in g["name"].lower() and "repeat" not in g["name"].lower()]
        repeat_dots = [g for g in glyphs if "repeatdot" in g["name"].lower()]

        for sys in p_info["systems"]:
            for s_idx, staff in enumerate(sys.staves):
                p_idx_for_staff = staff.part_idx

                # Find clef for this staff (near staff start x, vertically within staff bounds +- 10 pt)
                staff_clefs = [
                    c for c in clefs
                    if staff.x_min - 15 <= c["x"] <= staff.x_min + 60
                    and staff.top_y - 15 <= c["y"] <= staff.bottom_y + 15
                ]
                clef_x = staff.x_min
                if staff_clefs:
                    c_name = staff_clefs[0]["name"]
                    clef_x = staff_clefs[0]["x"]
                    if "fClef" in c_name:
                        staff.clef_sign = "F"
                        staff.clef_line = 4
                        staff.clef_octave = 0
                    elif "gClef8vb" in c_name:
                        staff.clef_sign = "G"
                        staff.clef_line = 2
                        staff.clef_octave = -1
                    else:
                        staff.clef_sign = "G"
                        staff.clef_line = 2
                        staff.clef_octave = 0

                # Detect key signature and time signature in staff header
                h_glyphs = [
                    g for g in glyphs
                    if staff.x_min - 10 <= g["x"] <= staff.x_min + 120
                    and staff.top_y - 20 <= g["y"] <= staff.bottom_y + 20
                ]
                h_note_xs = [
                    g["x"] for g in h_glyphs
                    if ("notehead" in g["name"] or "rest" in g["name"]) and g["x"] > clef_x + 10
                ]
                first_header_note_x = min(h_note_xs) if h_note_xs else staff.x_min + 120

                # Key signature sharps/flats
                key_accs = [
                    g for g in h_glyphs
                    if clef_x < g["x"] < first_header_note_x
                    and ("Sharp" in g["name"] or "Flat" in g["name"])
                ]
                if key_accs:
                    sharps = sum(1 for g in key_accs if "Sharp" in g["name"])
                    flats = sum(1 for g in key_accs if "Flat" in g["name"])
                    staff.key_fifths = sharps - flats
                    if score_key_fifths == 0:
                        score_key_fifths = staff.key_fifths
                else:
                    staff.key_fifths = score_key_fifths

                # Time signature
                ts_glyphs = [
                    g for g in h_glyphs
                    if clef_x < g["x"] < first_header_note_x
                    and "timeSig" in g["name"]
                ]
                if ts_glyphs:
                    ts_glyphs.sort(key=lambda g: g["y"])
                    if len(ts_glyphs) >= 2:
                        try:
                            b_str = ts_glyphs[0]["name"].replace("timeSig", "")
                            bt_str = ts_glyphs[1]["name"].replace("timeSig", "")
                            staff.time_beats = int(b_str)
                            staff.time_beat_type = int(bt_str)
                            score_time_beats = staff.time_beats
                            score_time_beat_type = staff.time_beat_type
                        except Exception:
                            pass
                    elif len(ts_glyphs) == 1:
                        if "Common" in ts_glyphs[0]["name"]:
                            staff.time_beats = 4
                            staff.time_beat_type = 4
                else:
                    staff.time_beats = score_time_beats
                    staff.time_beat_type = score_time_beat_type

                # Notes and rests belonging to this staff vertically
                # Between staff.top_y - 30 and staff.bottom_y + 30
                staff_notes_raw = []

                for nh in noteheads:
                    if staff.top_y - 25 <= nh["y"] <= staff.bottom_y + 25 and staff.x_min - 5 <= nh["x"] <= staff.x_max + 5:
                        staff_notes_raw.append({
                            "is_rest": False,
                            "glyph_name": nh["name"],
                            "x": nh["x"],
                            "y": nh["y"],
                            "bbox": nh["bbox"],
                        })

                for r in rests:
                    if staff.top_y - 15 <= r["y"] <= staff.bottom_y + 15 and staff.x_min - 5 <= r["x"] <= staff.x_max + 5:
                        staff_notes_raw.append({
                            "is_rest": True,
                            "glyph_name": r["name"],
                            "x": r["x"],
                            "y": r["y"],
                            "bbox": r["bbox"],
                        })

                # Sort by x
                staff_notes_raw.sort(key=lambda n: n["x"])

                # Attach accidentals and compute pitch
                for n in staff_notes_raw:
                    if not n["is_rest"]:
                        # Look for explicit accidental to the left of the notehead
                        nearby_acc = [
                            a for a in accidentals
                            if 0 < n["x"] - a["x"] < 16.0 and abs(n["y"] - a["y"]) < 6.0
                        ]
                        exp_acc = nearby_acc[0]["name"] if nearby_acc else None
                        
                        pitch = calculate_pitch_from_y(
                            orig_y=n["y"],
                            staff_bottom_y=staff.bottom_y,
                            line_spacing=staff.line_spacing,
                            clef_sign=staff.clef_sign,
                            clef_line=staff.clef_line,
                            clef_octave=staff.clef_octave,
                            explicit_accidental=exp_acc,
                            key_fifths=staff.key_fifths,
                        )
                        n["step"] = pitch.step
                        n["octave"] = pitch.octave
                        n["alter"] = pitch.alter
                        n["accidental"] = pitch.accidental

                # Resolve durations
                resolve_note_durations(
                    notes=staff_notes_raw,
                    beams=beams,
                    flags=flags,
                    dots=dots,
                    time_beats=staff.time_beats or score_time_beats,
                    time_beat_type=staff.time_beat_type or score_time_beat_type,
                )

                # Next staff top y for lyrics band
                next_top = sys.staves[s_idx + 1].top_y if s_idx + 1 < len(sys.staves) else staff.bottom_y + 90.0
                
                # Extract lyrics
                lyric_lines = extract_lyrics_for_staff(
                    page_words=words,
                    staff_bottom_y=staff.bottom_y,
                    next_staff_top_y=next_top,
                    staff_x_min=staff.x_min,
                    staff_x_max=staff.x_max,
                )
                match_syllables_to_notes(staff_notes_raw, lyric_lines)

                # Extract barlines and create measures
                barlines = extract_barlines_for_staff(drawings, staff, repeat_dots=repeat_dots)
                measures = create_measures_for_staff(staff, barlines, staff_notes_raw)

                # Renumber measures consecutively
                for m in measures:
                    m.measure_number = part_measure_counters[p_idx_for_staff]
                    part_measure_counters[p_idx_for_staff] += 1

                    # Prepare dict for writer
                    m_dict = {
                        "number": m.measure_number,
                        "key_fifths": staff.key_fifths,
                        "time_beats": staff.time_beats or score_time_beats,
                        "time_beat_type": staff.time_beat_type or score_time_beat_type,
                        "clef_sign": staff.clef_sign,
                        "clef_line": staff.clef_line,
                        "clef_octave": staff.clef_octave,
                        "notes": m.notes,
                    }
                    if m.end_barline and m.end_barline.repeat_direction:
                        m_dict["end_barline"] = {"repeat_direction": m.end_barline.repeat_direction}

                    parts[p_idx_for_staff]["measures"].append(m_dict)

    # Detect title from top text on first page
    title = ""
    composer = ""
    lyricist = ""
    first_page_words = page_data[0]["words"] if page_data else []
    first_staff_top = all_systems[0].top_y if all_systems else 200.0

    header_words = [w for w in first_page_words if w[3] < first_staff_top - 15]
    # Cluster header words by y
    if header_words:
        h_lines = {}
        for w in header_words:
            y = round(w[1], -1)
            h_lines.setdefault(y, []).append(w)
        for y, wlist in sorted(h_lines.items()):
            wlist.sort(key=lambda w: w[0])
            txt = " ".join(w[4] for w in wlist)
            # Largest font / center text is usually title
            if not title and len(txt) > 2 and not txt.startswith("©"):
                title = txt

    return {
        "title": title,
        "composer": composer,
        "lyricist": lyricist,
        "key_fifths": score_key_fifths,
        "time_beats": score_time_beats,
        "time_beat_type": score_time_beat_type,
        "tempo": score_tempo,
        "parts": parts,
    }
