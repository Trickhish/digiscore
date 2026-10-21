"""
Full vector score parser.
Extracts staves, systems, clefs, notes, measures, and lyrics from vector PDF pages.
"""

from typing import Dict, List, Optional, Tuple
import pymupdf

from digiscore.vector.ties import find_arcs, mark_ties
from digiscore.smufl import detect_capella_aliases, detect_named_fonts, glyph_trace, detect_mscore_flavour, get_glyph_name, is_clef, is_notehead, is_rest
from digiscore.vector.duration import resolve_note_durations
from digiscore.vector.lyrics import extract_lyrics_for_staff, extract_text_tokens, match_syllables_to_notes, styled_tokens
from digiscore.vector.measure import create_measures_for_staff, extract_barlines_for_staff
from digiscore.vector.pitch import calculate_pitch_from_y
from digiscore.vector.staff import Staff, System, detect_staves_on_page, group_staves_into_systems
from digiscore.vector.stems import extract_stems_and_beams, find_stem_for_note
from digiscore.vector.voices import assign_voices


def _head_staff(head: dict, staves: List[Staff], heads: List[dict], stems) -> Optional[Staff]:
    """
    The staff of a notehead: the nearest, except a head between two staves that
    shares its stem with heads lying inside the other staff (a chord written
    across the gap belongs to the staff holding most of it).
    """
    near = nearest_staff(head, staves)
    if near is None or near.top_y <= head["y"] <= near.bottom_y:
        return near
    sp = near.line_spacing
    stem = find_stem_for_note(head, stems, sp)
    if stem is None:
        return near
    votes: Dict[int, int] = {}
    for h in heads:
        if h is head or abs(h["x"] - head["x"]) > 0.5 * sp:
            continue
        if not (stem.y0 - sp <= h["y"] <= stem.y1 + sp):
            continue
        st = nearest_staff(h, staves)
        if st is not None and st.top_y <= h["y"] <= st.bottom_y:
            votes[id(st)] = votes.get(id(st), 0) + 1
    if votes:
        best = max(votes, key=votes.get)
        for st in staves:
            if id(st) == best:
                return st
    return near


def nearest_staff(glyph: dict, staves: List[Staff]) -> Optional[Staff]:
    """The staff whose vertical span is closest to the glyph (each symbol belongs to one staff)."""
    best, best_d = None, None
    for s in staves:
        if not (s.x_min - 5 <= glyph["x"] <= s.x_max + 5):
            continue
        y = glyph["y"]
        d = 0.0 if s.top_y <= y <= s.bottom_y else min(abs(y - s.top_y), abs(y - s.bottom_y))
        if best is None or d < best_d:
            best, best_d = s, d
    return best


def _note_out(n: dict, voice: int) -> dict:
    """Copy of a note for the writer, without geometry objects."""
    out = {k: v for k, v in n.items() if k not in ("stem", "bbox")}
    out["voice"] = voice
    return out


_TIME_SYMBOLS = {"timeSigCommon": (4, 4, "common"), "timeSigCutCommon": (2, 2, "cut")}


def read_time_signatures(glyphs: List[dict], staff: Staff, x_lo: float, x_hi: float = float("inf")
                         ) -> List[Tuple[float, int, int, Optional[str]]]:
    """
    Time signatures on a staff between x_lo and x_hi: (x, beats, beat_type, symbol).
    A signature is a common/cut symbol, or a row of digits above the middle line
    over a row below it (each row may hold several digits, e.g. 12 over 8).
    """
    sp = staff.line_spacing
    mid = (staff.top_y + staff.bottom_y) / 2
    found = [
        g for g in glyphs
        if g["name"].startswith("timeSig") and x_lo - 1 <= g["x"] < x_hi
        and staff.top_y - sp <= g["y"] <= staff.bottom_y + sp
    ]
    out = []
    for g in found:
        if g["name"] in _TIME_SYMBOLS:
            out.append((g["x"], *_TIME_SYMBOLS[g["name"]]))
    digits = sorted((g for g in found if g["name"][7:].isdigit()), key=lambda g: g["x"])
    # Cluster digits that overlap horizontally into one signature
    clusters: List[List[dict]] = []
    for g in digits:
        if clusters and g["x"] - max(d["x"] for d in clusters[-1]) < 1.5 * sp:
            clusters[-1].append(g)
        else:
            clusters.append([g])
    for cl in clusters:
        top = sorted((g for g in cl if g["y"] <= mid + 0.5), key=lambda g: g["x"])
        bot = sorted((g for g in cl if g["y"] > mid + 0.5), key=lambda g: g["x"])
        if not top or not bot:
            continue
        beats = int("".join(g["name"][7:] for g in top))
        beat_type = int("".join(g["name"][7:] for g in bot))
        if beats > 0 and beat_type in (1, 2, 4, 8, 16, 32):
            out.append((min(g["x"] for g in cl), beats, beat_type, None))
    return sorted(out)


def find_key_changes(accidentals, staff: Staff, measures, heads, header_end: float) -> Dict[int, Tuple[int, float]]:
    """
    Key signatures at the start of a bar inside the staff: a run of accidentals
    right after the opening barline, none of which belongs to a notehead
    (naturals cancel the old key). Returns {id(measure): (fifths, end x)}.
    """
    sp = staff.line_spacing
    out = {}
    for m in measures:
        if m.x0 <= header_end + sp:
            continue
        first_head = min((h["x"] for h in heads if m.x0 < h["x"] < m.x1), default=m.x1)
        run = sorted(
            (a for a in accidentals
             if m.x0 - 0.5 * sp < a["x"] < first_head
             and staff.top_y - 2 * sp <= a["y"] <= staff.bottom_y + 2 * sp),
            key=lambda a: a["x"],
        )
        if not run:
            continue
        # A note's own accidental sits right against its notehead, at its height
        def _own(a):
            return any(abs(h["y"] - a["y"]) < 0.4 * sp and -0.2 * sp < h["bbox"][0] - a["bbox"][2] < 1.0 * sp
                       for h in heads if m.x0 < h["x"] < m.x1)
        if _own(run[-1]):
            run = run[:-1]
        signed = [a for a in run if "Sharp" in a["name"] or "Flat" in a["name"]]
        naturals = [a for a in run if "Natural" in a["name"]]
        if not signed and len(naturals) < 2:
            continue
        sharps = sum("Sharp" in a["name"] for a in signed)
        flats = sum("Flat" in a["name"] for a in signed)
        if sharps and flats:
            continue
        out[id(m)] = (sharps - flats, run[-1]["x"] + 0.5)
    return out


def find_time_signature_changes(glyphs: List[dict], staff: Staff, after_x: float):
    """Time signatures on the staff right of the staff header."""
    return read_time_signatures(glyphs, staff, after_x)


def _pair_ties(parts: List[dict]) -> None:
    """Keep a tie only when its start and stop are consecutive notes of one voice at the same pitch."""
    def key(n):
        return (n.get("step"), n.get("octave"), n.get("alter", 0))

    for p in parts:
        seqs: Dict[int, List[dict]] = {}
        for m in p["measures"]:
            for v_idx, v in enumerate(m.get("voices", [])):
                for ev in v:
                    if ev:
                        seqs.setdefault(v_idx, []).append(ev)
        for seq in seqs.values():
            ok = set()
            for i in range(len(seq) - 1):
                for a in seq[i]:
                    for b in seq[i + 1]:
                        # An accidental carries through a tie into the next bar
                        if (a.get("tie_start") and b.get("tie_stop") and not a.get("is_rest")
                                and (a.get("step"), a.get("octave")) == (b.get("step"), b.get("octave"))
                                and a.get("alter", 0) != b.get("alter", 0) and not b.get("accidental")):
                            b["alter"] = a.get("alter", 0)
                        if a.get("tie_start") and b.get("tie_stop") and not a.get("is_rest") and key(a) == key(b):
                            ok.update({(id(a), "tie_start"), (id(b), "tie_stop")})
            for ev in seq:
                for n in ev:
                    for flag in ("tie_start", "tie_stop"):
                        if n.get(flag) and (id(n), flag) not in ok:
                            n.pop(flag)


def _resolve_grace(notes: List[dict], sp: float) -> None:
    """
    Small heads are grace notes only when packed tightly in front of a
    full-size note (directly or through other grace notes); small notes spaced
    out on their own are cue notes, which take their time.
    """
    heads = sorted((n for n in notes if not n.get("is_rest")), key=lambda n: n["x"])
    for n in heads:
        n["small"] = bool(n.get("grace"))
    xs = sorted({round(n["x"], 1) for n in heads})
    by_x: Dict[float, List[dict]] = {}
    for n in heads:
        by_x.setdefault(round(n["x"], 1), []).append(n)
    grace_x: Dict[float, bool] = {}
    for i in range(len(xs) - 1, -1, -1):
        x = xs[i]
        group = by_x[x]
        if not all(n.get("grace") for n in group):
            grace_x[x] = False
            continue
        ok = False
        if i + 1 < len(xs):
            nx = xs[i + 1]
            d = nx - x
            nxt_small = all(n.get("small") for n in by_x[nx])
            # Led into a full-size note, directly or through a tight run of grace notes
            ok = (not nxt_small and d < 4.0 * sp) or (grace_x.get(nx, False) and d < 2.5 * sp)
        grace_x[x] = ok
        for n in group:
            n["grace"] = ok


def _settle_grace(notes: List[dict], bar_len: float, sp: float, mid: float) -> None:
    """
    Candidate grace notes become grace notes (no time) when the bar is too
    long with them counted; otherwise they are cue notes and keep their value.
    """
    cand = [n for n in notes if n.get("grace")]
    if not cand:
        return
    for n in cand:
        n["grace"] = False
    longest = max((sum(e[0].get("duration_quarters", 0.0) for e in v)
                   for v in assign_voices(notes, sp, mid)), default=0.0)
    if longest <= bar_len + 1e-6:
        return
    for n in cand:
        n["grace"] = True
        n["duration_quarters"] = 0.0
        n["dot_count"] = 0


def _apply_ottavas(staff: Staff, notes: List[dict], glyphs: List[dict], drawings: List[dict]) -> None:
    """
    8va / 8vb lines: the "8" glyph, then a dashed line to the end of the
    passage. Heads of this staff under (8va) or over (8vb) it move an octave.
    """
    sp = staff.line_spacing
    for g in glyphs:
        name = g["name"]
        if not name.startswith("ottava"):
            continue
        x, y = g["x"], g["y"]
        # A bare "8" says which by where it stands: over the staff 8va, under 8vb
        if staff.top_y - 8 * sp < y < staff.top_y:
            below = "Bassa" in name
        elif staff.bottom_y < y < staff.bottom_y + 8 * sp:
            below = True
        else:
            continue
        if not staff.x_min - 5 <= x <= staff.x_max + 5:
            continue
        x_end = x + 3 * sp
        for d in drawings:
            items = d.get("items", [])
            if len(items) != 1 or items[0][0] != "l":
                continue
            p1, p2 = items[0][1], items[0][2]
            if abs(p1.y - p2.y) > 0.5 or not d.get("dashes") or d.get("dashes") in ("[] 0",):
                continue
            lx0, lx1 = min(p1.x, p2.x), max(p1.x, p2.x)
            if 0 <= lx0 - x < 4 * sp and abs(p1.y - y) < 3 * sp:
                x_end = max(x_end, lx1)
        # A closing hook after the dashes may be a separate short vertical line
        for n in notes:
            if n.get("is_rest") or n.get("octave") is None:
                continue
            if x - 1 <= n["x"] <= x_end + 1:
                n["octave"] += -1 if below else 1


def _staff_signature(staff: Staff, glyphs: List[dict]) -> tuple:
    """(clef, staff size) of a staff, to recognise the same part in another system."""
    cl = [g for g in glyphs if is_clef(g["name"])
          and staff.x_min - 15 <= g["x"] <= staff.x_min + 60
          and staff.top_y - 15 <= g["y"] <= staff.bottom_y + 15]
    cl.sort(key=lambda g: g["x"])
    name = cl[0]["name"] if cl else ""
    sign = "F" if "fClef" in name else "C" if name.startswith("cClef") else "G"
    return sign, "8vb" in name, round(staff.line_spacing, 1)


def _align_staves(sig: List[tuple], full: List[tuple]) -> List[int]:
    """
    Order-preserving match of a system's staves to the fullest system's ones:
    the part index of each staff. Staff size weighs most, then the clef; among
    equal matches the most compact one (parts left out together) wins.
    """
    n, m = len(sig), len(full)

    def score(a, b):
        v = 0.0
        if a[0] == b[0]:
            v += 2
            v += 1 if a[1] == b[1] else 0.5
        # A part keeps its staff size; its clef may change between systems
        if abs(a[2] - b[2]) < 0.15 * max(a[2], b[2]):
            v += 3
        return v

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best(i, j, prev):
        # (score, -gaps, choice list) for staves i.. mapped to template j..
        if i == n:
            return (0.0, 0, ())
        if m - j < n - i:
            return (-1e9, 0, ())
        out = None
        for k in range(j, m - (n - i) + 1):
            sc, gp, ch = best(i + 1, k + 1, k)
            gap = 0 if prev < 0 or k == prev + 1 else 1
            cand = (sc + score(sig[i], full[k]), gp - gap, (k,) + ch)
            if out is None or cand[:2] > out[:2]:
                out = cand
        return out

    return list(best(0, 0, -1)[2])


def _merge_split_systems(page_data: List[dict]) -> None:
    """
    A system whose staves are not joined by any line (a first system with
    incipits, say) is read as single staves: a run of as many single staves as
    the full systems have, aligned and spaced like them, is one system.
    """
    all_sys = [sy for pd in page_data for sy in pd["systems"]]
    width = max(len(sy.staves) for sy in all_sys)
    if width < 2:
        return
    gaps = [b.top_y - a.top_y for sy in all_sys if len(sy.staves) == width
            for a, b in zip(sy.staves, sy.staves[1:])]
    if not gaps:
        return
    gap = sorted(gaps)[len(gaps) // 2]
    for pd in page_data:
        out, i, systems = [], 0, pd["systems"]
        while i < len(systems):
            run = systems[i:i + width]
            if (len(run) == width and all(len(sy.staves) == 1 for sy in run)
                    and all(abs(a.staves[0].x_min - b.staves[0].x_min) < 2
                            and abs(a.staves[0].x_max - b.staves[0].x_max) < 2
                            and abs((b.staves[0].top_y - a.staves[0].top_y) - gap) < 0.25 * gap
                            for a, b in zip(run, run[1:]))):
                head = run[0]
                head.staves = [sy.staves[0] for sy in run]
                for k, st in enumerate(head.staves):
                    st.part_idx = k
                    st.system_idx = head.system_idx
                head.bottom_y = head.staves[-1].bottom_y
                out.append(head)
                i += width
                continue
            out.append(systems[i])
            i += 1
        pd["systems"] = out


def _assign_parts(page_data: List[dict]) -> None:
    """Part index of each staff when systems leave some parts out."""
    _merge_split_systems(page_data)
    systems = [(sys, pd["glyphs"]) for pd in page_data for sys in pd["systems"]]
    width = max(len(sys.staves) for sys, _ in systems)
    if all(len(sys.staves) == width for sys, _ in systems):
        return
    full_sys, full_g = next((sy, g) for sy, g in systems if len(sy.staves) == width)
    full = [_staff_signature(st, full_g) for st in full_sys.staves]
    for sys, g in systems:
        if len(sys.staves) == width:
            continue
        idx = _align_staves([_staff_signature(st, g) for st in sys.staves], full)
        for st, k in zip(sys.staves, idx):
            st.part_idx = k


def _fill_absent_parts(parts, counters, next_sys, state) -> None:
    """
    Called before each system (and once at the end): the parts the previous
    system left out get bar rests for each of its bars, so all parts keep the
    same bar count.
    """
    prev = state.get("sys")
    if prev is not None:
        present = {st.part_idx for st in prev.staves}
        n_bars = max((counters[k] - state["start"][k] for k in present), default=0)
        ref = next((parts[k]["measures"][-n_bars:] for k in sorted(present)
                    if n_bars and len(parts[k]["measures"]) >= n_bars), [])
        for k, part in enumerate(parts):
            if k in present or not ref:
                continue
            last = part["measures"][-1] if part["measures"] else None
            for rm in ref:
                beats, bt = rm.get("time_beats") or 4, rm.get("time_beat_type") or 4
                dur = beats * 4.0 / bt
                rest = {"is_rest": True, "measure_rest": True, "x": 0.0, "y": 0.0,
                        "duration_quarters": dur, "type": "whole", "dot_count": 0, "lyrics": []}
                part["measures"].append({
                    "number": counters[k],
                    "key_fifths": rm.get("key_fifths"),
                    "time_beats": rm.get("time_beats"), "time_beat_type": rm.get("time_beat_type"),
                    "time_symbol": rm.get("time_symbol"),
                    "clef_sign": last.get("clef_sign") if last else None,
                    "clef_line": last.get("clef_line") if last else None,
                    "clef_octave": last.get("clef_octave") if last else None,
                    "filled": True,
                    "notes": [rest], "voices": [[[dict(rest, voice=1)]]],
                })
                counters[k] += 1
    state["sys"] = next_sys
    state["start"] = list(counters)


def parse_vector_score(doc: pymupdf.Document) -> dict:
    """
    Parses a vector music PDF and returns the full score dictionary.
    """
    all_systems: List[System] = []
    page_data = []
    detect_mscore_flavour(doc)
    detect_capella_aliases(doc)
    detect_named_fonts(doc)

    # 1. Extract staves and page elements per page
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        staves = detect_staves_on_page(page, page_idx)
        drawings = page.get_drawings()
        systems = group_staves_into_systems(staves, drawings)
        all_systems.extend(systems)

        raw = page.get_text("rawdict")
        words = extract_text_tokens(page)
        styled_words = set(styled_tokens(page))

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
                                    "size": span.get("size", 0.0),
                                    "x": ch["origin"][0],
                                    "y": ch["origin"][1],
                                    "bbox": ch["bbox"],
                                })

        glyphs.extend(glyph_trace(page))

        # Some programs (capella) draw shared symbols once per voice: the same
        # glyph at the same spot twice would double key signatures and clefs
        seen_g, unique = set(), []
        for g in glyphs:
            k = (g["name"], round(g["x"] * 2), round(g["y"] * 2))
            # Two voices on one unison share a head legitimately
            if k in seen_g and not g["name"].startswith("notehead"):
                continue
            seen_g.add(k)
            unique.append(g)
        glyphs = unique

        page_spacing = (
            sorted(s.line_spacing for s in staves)[len(staves) // 2] if staves else 5.0
        )
        stems, beams = extract_stems_and_beams(drawings, page_spacing)

        page_data.append({
            "page_idx": page_idx,
            "staves": staves,
            "systems": systems,
            "glyphs": glyphs,
            "beams": beams,
            "stems": stems,
            "words": words,
            "styled_words": styled_words,
            "drawings": drawings,
        })

    if not all_systems:
        return {}

    _assign_parts(page_data)
    all_systems = [sy for pd in page_data for sy in pd["systems"]]

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
    score_time_symbol = None
    score_time_seen = None
    part_time: Dict[int, tuple] = {}
    score_tempo = None

    # Track measure numbers across systems for each part
    part_measure_counters = [1] * part_count
    fill_state: dict = {}

    for p_info in page_data:
        p_idx = p_info["page_idx"]
        glyphs = p_info["glyphs"]
        beams = p_info["beams"]
        stems = p_info["stems"]
        words = p_info["words"]
        styled_words = p_info.get("styled_words", set())
        drawings = p_info["drawings"]
        tie_arcs = find_arcs(drawings)

        # Separate glyph types
        noteheads = [g for g in glyphs if is_notehead(g["name"])]
        # Grace notes: heads clearly smaller than the page's usual head size
        head_sizes = sorted(g.get("size", 0.0) for g in noteheads)
        usual_head = head_sizes[len(head_sizes) // 2] if head_sizes else 0.0
        rests = [g for g in glyphs if is_rest(g["name"])]
        clefs = [g for g in glyphs if is_clef(g["name"])]
        accidentals = [g for g in glyphs if "accidental" in g["name"].lower()]
        flags = [g for g in glyphs if "flag" in g["name"].lower()]
        dots = [g for g in glyphs if "dot" in g["name"].lower() and "repeat" not in g["name"].lower()]
        repeat_dots = [g for g in glyphs if "repeatdot" in g["name"].lower()]

        page_staves = p_info["staves"]

        for sys in p_info["systems"]:
            _fill_absent_parts(parts, part_measure_counters, sys, fill_state)
            upper_lyrics = []
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
                    elif c_name.startswith("cClef"):
                        # The clef's centre marks its line (alto: 3, tenor: 4)
                        bb = staff_clefs[0].get("bbox")
                        cy = (bb[1] + bb[3]) / 2 if bb and bb[3] - bb[1] < 6 * staff.line_spacing \
                            else staff_clefs[0]["y"]
                        line = int(round((staff.bottom_y - cy) / staff.line_spacing)) + 1
                        staff.clef_sign = "C"
                        staff.clef_line = min(5, max(1, line))
                        staff.clef_octave = 0
                    elif "gClef8vb" in c_name:
                        staff.clef_sign = "G"
                        staff.clef_line = 2
                        staff.clef_octave = -1
                    else:
                        staff.clef_sign = "G"
                        staff.clef_line = 2
                        staff.clef_octave = 0
                        # Separate small "8" drawn under the clef (Sibelius tenor clef)
                        if any(
                            g["name"] == "clef8"
                            and abs(g["x"] - clef_x) < 4 * staff.line_spacing
                            and staff.bottom_y < g["y"] < staff.bottom_y + 5 * staff.line_spacing
                            for g in glyphs
                        ):
                            staff.clef_octave = -1

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

                # Key signature sharps/flats (an accidental right before a notehead at the
                # same height is that note's accidental, not part of the key signature)
                sp_h = staff.line_spacing
                h_heads = [g for g in h_glyphs if "notehead" in g["name"]]

                def _attached_to_head(a):
                    return any(
                        0 < h["x"] - a["x"] < 2.5 * sp_h and abs(h["y"] - a["y"]) < 0.4 * sp_h
                        for h in h_heads
                    )

                key_accs = [
                    g for g in h_glyphs
                    if clef_x < g["x"] < first_header_note_x
                    and ("Sharp" in g["name"] or "Flat" in g["name"])
                    and not _attached_to_head(g)
                ]
                key_end_x = max((g["x"] for g in key_accs), default=clef_x)
                if key_accs:
                    sharps = sum(1 for g in key_accs if "Sharp" in g["name"])
                    flats = sum(1 for g in key_accs if "Flat" in g["name"])
                    staff.key_fifths = sharps - flats
                    if score_key_fifths == 0:
                        score_key_fifths = staff.key_fifths
                else:
                    staff.key_fifths = score_key_fifths

                # Time signature
                header_ts = read_time_signatures(h_glyphs, staff, clef_x + 1, first_header_note_x)
                if header_ts:
                    _, staff.time_beats, staff.time_beat_type, staff.time_symbol = header_ts[0]
                    if score_time_seen is None:
                        score_time_beats, score_time_beat_type = staff.time_beats, staff.time_beat_type
                        score_time_symbol = staff.time_symbol
                        score_time_seen = True
                else:
                    # The time in force on this part (a change earlier on another staff of the
                    # same system must not leak into this one)
                    staff.time_beats, staff.time_beat_type, staff.time_symbol = part_time.get(
                        p_idx_for_staff, (score_time_beats, score_time_beat_type, score_time_symbol))

                # Notes and rests belonging to this staff vertically
                # Between staff.top_y - 30 and staff.bottom_y + 30
                staff_notes_raw = []

                for nh in noteheads:
                    if _head_staff(nh, page_staves, noteheads, stems) is not staff:
                        continue
                    if staff.top_y - 25 <= nh["y"] <= staff.bottom_y + 25 and staff.x_min - 5 <= nh["x"] <= staff.x_max + 5:
                        staff_notes_raw.append({
                            "is_rest": False,
                            "glyph_name": nh["name"],
                            "x": nh["x"],
                            "y": nh["y"],
                            "bbox": nh["bbox"],
                            "grace": bool(usual_head and 0 < nh.get("size", usual_head) < 0.8 * usual_head),
                        })

                for r in rests:
                    if nearest_staff(r, page_staves) is not staff:
                        continue
                    # Rests of an upper or lower voice may sit well outside the staff
                    r_pad = max(15.0, 4.0 * staff.line_spacing)
                    if staff.top_y - r_pad <= r["y"] <= staff.bottom_y + r_pad and staff.x_min - 5 <= r["x"] <= staff.x_max + 5:
                        staff_notes_raw.append({
                            "is_rest": True,
                            "glyph_name": r["name"],
                            "x": r["x"],
                            "y": r["y"],
                            "bbox": r["bbox"],
                        })

                # Sort by x
                staff_notes_raw.sort(key=lambda n: n["x"])

                # Resolve durations
                resolve_note_durations(
                    notes=staff_notes_raw,
                    stems=stems,
                    beams=beams,
                    flags=flags,
                    dots=dots,
                    line_spacing=staff.line_spacing,
                )

                _resolve_grace(staff_notes_raw, staff.line_spacing)

                # Next staff top y for lyrics band
                below = [s.top_y for s in page_staves if s.top_y > staff.bottom_y + 1]
                next_top = min(below) if below else staff.bottom_y + 90.0
                
                # Extract lyrics
                lyric_lines = extract_lyrics_for_staff(
                    page_words=words,
                    staff_bottom_y=staff.bottom_y,
                    next_staff_top_y=next_top,
                    staff_x_min=staff.x_min,
                    staff_x_max=staff.x_max,
                    styled=styled_words,
                )
                # Two-staff choir layout (SA / TB): words printed between the
                # staves are sung by all voices when the lower staff has none
                if (not lyric_lines and s_idx == 1 and len(sys.staves) == 2
                        and upper_lyrics and staff.clef_sign == "F"):
                    lyric_lines = upper_lyrics
                if s_idx == 0:
                    upper_lyrics = lyric_lines
                match_syllables_to_notes(staff_notes_raw, lyric_lines)

                # Extract barlines and create measures
                barlines = extract_barlines_for_staff(
                    drawings, staff, repeat_dots=repeat_dots,
                    # Any head of the page: a stem may cross into this staff
                    # from a note of the staff above or below
                    noteheads=[n for n in staff_notes_raw if not n["is_rest"]] + [
                        {"x": h["x"], "y": h["y"], "bbox": h["bbox"]} for h in noteheads
                        if staff.top_y - 12 * staff.line_spacing <= h["y"] <= staff.bottom_y + 12 * staff.line_spacing],
                )
                measures = create_measures_for_staff(staff, barlines, staff_notes_raw)

                # Pitch: key signature, then explicit accidentals that persist to the barline
                header_end = key_end_x + 0.5
                sp = staff.line_spacing
                key_changes = find_key_changes(
                    accidentals, staff, measures, [n for n in staff_notes_raw if not n["is_rest"]], header_end)
                cur_key = staff.key_fifths
                for m in measures:
                    if id(m) in key_changes:
                        cur_key = key_changes[id(m)][0]
                    m.key_fifths = cur_key
                    key_zone_end = key_changes[id(m)][1] if id(m) in key_changes else header_end
                    acc_state: Dict = {}
                    for n in m.notes:
                        if n["is_rest"]:
                            continue
                        hx0 = n["bbox"][0] if n.get("bbox") else n["x"]
                        nearby_acc = [
                            a for a in accidentals
                            if a["x"] > key_zone_end
                            and 0 < hx0 - a["x"] < 3.2 * sp
                            and abs(n["y"] - a["y"]) < 0.4 * sp
                        ]
                        nearby_acc.sort(key=lambda a: hx0 - a["x"])
                        exp_acc = nearby_acc[0]["name"] if nearby_acc else None
                        pitch = calculate_pitch_from_y(
                            orig_y=n["y"],
                            staff_bottom_y=staff.bottom_y,
                            line_spacing=sp,
                            clef_sign=staff.clef_sign,
                            clef_line=staff.clef_line,
                            clef_octave=staff.clef_octave,
                            explicit_accidental=exp_acc,
                            key_fifths=cur_key,
                            measure_accidental_state=acc_state,
                        )
                        n["step"] = pitch.step
                        n["octave"] = pitch.octave
                        n["alter"] = pitch.alter
                        n["accidental"] = pitch.accidental

                # Ties (filled arcs joining two noteheads of the same pitch)
                _apply_ottavas(staff, staff_notes_raw, glyphs, drawings)

                mark_ties(staff, staff_notes_raw, tie_arcs)

                # Time signature changes inside the staff (stacked digit pairs after the header)
                ts_changes = find_time_signature_changes(glyphs, staff, first_header_note_x)

                # Renumber measures consecutively
                cur_beats = staff.time_beats or score_time_beats
                cur_beat_type = staff.time_beat_type or score_time_beat_type
                cur_symbol = staff.time_symbol or None
                for m in measures:
                    m.measure_number = part_measure_counters[p_idx_for_staff]
                    part_measure_counters[p_idx_for_staff] += 1
                    for tx, b, bt, sym in ts_changes:
                        # A time signature sits right after the barline that opens its bar
                        if m.x0 - 2 <= tx < m.x0 + 4 * staff.line_spacing:
                            cur_beats, cur_beat_type, cur_symbol = b, bt, sym
                    part_time[p_idx_for_staff] = (cur_beats, cur_beat_type, cur_symbol)

                    # Prepare dict for writer
                    m_dict = {
                        "number": m.measure_number,
                        "key_fifths": getattr(m, "key_fifths", staff.key_fifths),
                        "time_beats": cur_beats,
                        "time_beat_type": cur_beat_type,
                        "time_symbol": cur_symbol,
                        "clef_sign": staff.clef_sign,
                        "clef_line": staff.clef_line,
                        "clef_octave": staff.clef_octave,
                        "notes": m.notes,
                    }
                    _settle_grace(m.notes, cur_beats * 4.0 / cur_beat_type, staff.line_spacing,
                                  (staff.top_y + staff.bottom_y) / 2)
                    voices = assign_voices(
                        m.notes, staff.line_spacing, (staff.top_y + staff.bottom_y) / 2
                    )
                    m_dict["voices"] = [
                        [[_note_out(n, v_idx + 1) for n in ev] for ev in events]
                        for v_idx, events in enumerate(voices)
                    ]
                    if m.end_barline and m.end_barline.repeat_direction:
                        m_dict["end_barline"] = {"repeat_direction": m.end_barline.repeat_direction}

                    parts[p_idx_for_staff]["measures"].append(m_dict)

    _fill_absent_parts(parts, part_measure_counters, None, fill_state)
    # Rest bars before a part's first staff take the clef it starts with
    for part in parts:
        first = next((m for m in part["measures"] if not m.get("filled")), None)
        for m in part["measures"]:
            if m.get("clef_sign") is None:
                for k in ("clef_sign", "clef_line", "clef_octave"):
                    m[k] = first.get(k) if first else ("G", 2, 0)[("clef_sign", "clef_line", "clef_octave").index(k)]

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

    _pair_ties(parts)

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
