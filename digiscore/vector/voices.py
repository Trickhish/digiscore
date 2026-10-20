"""
Chord grouping and voice separation within a measure.

- Noteheads that share a stem form a chord (one rhythmic event).
- On a staff that carries two voices (e.g. S/A on one staff), stems up are voice 1
  and stems down are voice 2. A staff is treated as two-voiced only when both stem
  directions occur at the same horizontal position somewhere in the measure.
"""

from typing import Dict, List, Optional


def _group_events(notes: List[dict], tol: float) -> List[List[dict]]:
    """Group notes into events: same stem (if any), else same x for whole notes."""
    events: List[List[dict]] = []
    by_stem: Dict[int, List[dict]] = {}
    for n in sorted(notes, key=lambda n: n["x"]):
        stem = n.get("stem")
        if not n.get("is_rest") and stem is not None:
            key = id(stem)
            if key in by_stem:
                by_stem[key].append(n)
                continue
            by_stem[key] = [n]
            events.append(by_stem[key])
            continue
        if (
            not n.get("is_rest")
            and events
            and not events[-1][0].get("is_rest")
            and events[-1][0].get("stem") is None
            and abs(events[-1][0]["x"] - n["x"]) < tol
            and events[-1][0].get("duration_quarters") == n.get("duration_quarters")
        ):
            events[-1].append(n)
            continue
        events.append([n])
    events.sort(key=lambda e: min(n["x"] for n in e))
    return events


def _stem_dir(event: List[dict]) -> Optional[str]:
    return event[0].get("stem_dir")


def assign_voices(notes: List[dict], line_spacing: float, staff_mid_y: float) -> List[List[List[dict]]]:
    """
    Returns a list of voices, each a list of events (an event is a list of notes,
    more than one note meaning a chord).
    """
    tol = 0.6 * line_spacing
    events = _group_events(notes, tol)

    # Detect two-voice writing: an up-stem and a down-stem event at the same x
    ups = [e for e in events if _stem_dir(e) == "up"]
    downs = [e for e in events if _stem_dir(e) == "down"]
    two_voices = any(
        abs(min(n["x"] for n in u) - min(n["x"] for n in d)) < 1.5 * line_spacing
        for u in ups for d in downs
    )
    if not two_voices:
        return [events]

    v1: List[List[dict]] = []
    v2: List[List[dict]] = []
    for e in events:
        d = _stem_dir(e)
        if d in ("up", "down") and len(e) >= 2:
            # Both voices sharing one stem: top note(s) to voice 1, bottom to voice 2
            e_sorted = sorted(e, key=lambda n: n["y"])
            half = len(e_sorted) // 2
            v1.append(e_sorted[:half])
            v2.append(e_sorted[half:])
        elif d == "up":
            v1.append(e)
        elif d == "down":
            v2.append(e)
        else:
            # Stemless (whole notes, rests): split by height; a shared unison goes to both
            if e[0].get("is_rest"):
                (v1 if e[0]["y"] < staff_mid_y else v2).append(e)
                continue
            hi = [n for n in e if n["y"] <= staff_mid_y]
            lo = [n for n in e if n["y"] > staff_mid_y]
            if len(e) >= 2:
                e_sorted = sorted(e, key=lambda n: n["y"])
                v1.append(e_sorted[: len(e_sorted) // 2] or e_sorted[:1])
                v2.append(e_sorted[len(e_sorted) // 2:])
            elif hi:
                v1.append(hi)
            else:
                v2.append(lo)
    if not v2:
        return [v1]
    if not v1:
        return [v2]
    return [v1, _fill_gaps(v2, v1, line_spacing), ]


def _dur(event: List[dict]) -> float:
    return max(n.get("duration_quarters", 1.0) for n in event)


def _x(event: List[dict]) -> float:
    return min(n["x"] for n in event)


def _fill_gaps(voice: List[List[dict]], ref: List[List[dict]], line_spacing: float) -> List[List[dict]]:
    """
    Where `voice` has no notes but `ref` does, the passage is a unison written once
    (choral two-voice staves): copy the ref events. Anything else left becomes a
    hidden rest so both voices span the same time.
    """
    # Onsets of the reference voice, by x
    ref_on = []
    t = 0.0
    for e in ref:
        ref_on.append((_x(e), t, e))
        t += _dur(e)
    ref_total = t

    def onset_at(x: float) -> Optional[float]:
        for rx, rt, _ in ref_on:
            if abs(rx - x) < 1.5 * line_spacing:
                return rt
        return None

    # Place voice events in time: aligned with a ref onset when possible, else after the previous
    placed = []
    t = 0.0
    for e in voice:
        on = onset_at(_x(e))
        start = on if on is not None and on >= t - 1e-6 else t
        placed.append((start, e))
        t = start + _dur(e)

    out: List[List[dict]] = []
    t = 0.0
    for start, e in placed + [(ref_total, None)]:
        if start > t + 1e-6:
            gap_end = start
            for rx, rt, re_ in ref_on:
                if rt >= t - 1e-6 and rt + _dur(re_) <= gap_end + 1e-6:
                    if rt > t + 1e-6:
                        out.append([_hidden_rest(rt - t, rx)])
                    out.append([dict(n, unison_copy=True, lyrics=[]) for n in re_])
                    t = rt + _dur(re_)
            if gap_end > t + 1e-6:
                out.append([_hidden_rest(gap_end - t, start)])
                t = gap_end
        if e is not None:
            out.append(e)
            t = start + _dur(e)
    return out


_TYPES = [(4.0, "whole", 0), (3.0, "half", 1), (2.0, "half", 0), (1.5, "quarter", 1), (1.0, "quarter", 0),
          (0.75, "eighth", 1), (0.5, "eighth", 0), (0.375, "16th", 1), (0.25, "16th", 0), (0.125, "32nd", 0)]


def _hidden_rest(dur: float, x: float) -> dict:
    ntype, dots = "quarter", 0
    for d, t_, dt in _TYPES:
        if abs(d - dur) < 1e-6:
            ntype, dots = t_, dt
            break
    return {"is_rest": True, "hidden": True, "x": x, "y": 0.0,
            "duration_quarters": dur, "type": ntype, "dot_count": dots, "lyrics": []}
