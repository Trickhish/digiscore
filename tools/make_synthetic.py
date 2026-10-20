"""
Generate synthetic ground-truth scores: MusicXML from a seeded generator,
PDF rendered by MuseScore 4.

    python tools/make_synthetic.py [--mscore PATH] [--out examples/synthetic]

MuseScore is run under xvfb-run when no display is available.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

from music21 import (  # noqa: E402
    bar, clef, expressions, instrument, key, metadata, meter, note, pitch, repeat, stream, tie,
)

SYLLABLES = {
    "fr": ["le", "ciel", "a", "mour", "la", "nuit", "ver", "tu", "joie", "Dieu", "nous", "chan", "tons",
           "en", "fin", "pa", "ix", "ma", "ri", "e", "son", "coeur", "lu", "mi", "è", "re", "doux"],
    "la": ["glo", "ri", "a", "in", "ex", "cel", "sis", "De", "o", "pa", "cem", "do", "mi", "ne",
           "san", "ctus", "be", "ne", "di", "ctus", "a", "gnus", "al", "le", "lu", "ia"],
    "en": ["sing", "we", "now", "of", "Christ", "mas", "joy", "ful", "light", "ho", "ly", "night",
           "glad", "ti", "dings", "peace", "on", "earth", "an", "gels", "bright", "star", "shine"],
}

# Rhythm patterns per beat unit, in quarter lengths; each sums to one beat (or one bar for long notes)
BEAT_PATTERNS = {
    1.0: [[1.0], [1.0], [0.5, 0.5], [0.5, 0.5], [0.75, 0.25], [0.25, 0.25, 0.5], [0.25, 0.25, 0.25, 0.25]],
    1.5: [[1.5], [1.0, 0.5], [0.5, 0.5, 0.5], [0.5, 1.0], [0.75, 0.25, 0.5]],
    2.0: [[2.0], [1.0, 1.0], [1.5, 0.5], [1.0, 0.5, 0.5], [0.5, 0.5, 1.0]],
}

# (time signature, symbol, beat unit, beats per bar)
METERS = [
    ("2/2", "cut", 2.0, 2), ("4/4", "common", 1.0, 4), ("4/4", None, 1.0, 4), ("3/4", None, 1.0, 3),
    ("2/4", None, 1.0, 2), ("6/8", None, 1.5, 2), ("3/8", None, 1.5, 1), ("9/8", None, 1.5, 3),
    ("3/2", None, 2.0, 3), ("12/8", None, 1.5, 4),
]

VOICES = {
    "Soprano": ("G4", "C4", "G5", clef.TrebleClef),
    "Alto": ("D4", "G3", "D5", clef.TrebleClef),
    "Tenor": ("B3", "C3", "G4", clef.Treble8vbClef),
    "Bass": ("D3", "E2", "C4", clef.BassClef),
}


def scale_pitches(k: key.Key, lo: str, hi: str):
    lo_p, hi_p = pitch.Pitch(lo), pitch.Pitch(hi)
    return [p for p in k.getScale().getPitches(lo_p, hi_p)]


def bar_rhythm(rnd, beat, beats, long_ok=True):
    if long_ok and rnd.random() < 0.12:
        total = beat * beats
        if total in (2.0, 3.0, 4.0, 1.5, 6.0, 4.5):
            return [total]
    out = []
    for _ in range(beats):
        out += rnd.choice(BEAT_PATTERNS[beat])
    return out


def melody_line(rnd, scale, start_idx, n):
    idx = start_idx
    out = []
    for _ in range(n):
        step = rnd.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3])
        idx = max(0, min(len(scale) - 1, idx + step))
        out.append(idx)
    return out


def make_lyrics(rnd, lang, n):
    """n syllables with syllabic flags: words of 1-3 syllables."""
    out = []
    while len(out) < n:
        w = rnd.choice([1, 1, 2, 2, 3])
        w = min(w, n - len(out))
        sy = [rnd.choice(SYLLABLES[lang]) for _ in range(w)]
        for i, s in enumerate(sy):
            if w == 1:
                kind = "single"
            elif i == 0:
                kind = "begin"
            elif i == w - 1:
                kind = "end"
            else:
                kind = "middle"
            out.append((s, kind))
    return out


def make_score(seed: int):
    rnd = random.Random(seed)
    ts_str, symbol, beat, beats = METERS[seed % len(METERS)]
    tonic = rnd.choice(["C", "G", "D", "F", "B-", "E-", "A", "E", "a", "e", "d", "g"])
    k = key.Key(tonic)
    n_voices = rnd.choice([1, 1, 2, 4, 4])
    names = {1: ["Soprano"], 2: ["Soprano", "Alto"], 4: list(VOICES)}[n_voices]
    if n_voices == 1 and rnd.random() < 0.3:
        names = [rnd.choice(["Tenor", "Bass"])]
    n_bars = rnd.randint(10, 24)
    n_verses = rnd.choice([1, 1, 2, 3])
    lang = rnd.choice(list(SYLLABLES))
    homophonic = rnd.random() < 0.6
    use_repeat = rnd.random() < 0.3 and n_bars >= 8
    change_at = rnd.randint(n_bars // 2, n_bars - 2) if rnd.random() < 0.25 else None
    change_meter = METERS[(seed + 3) % len(METERS)] if change_at else None
    key_change_at = rnd.randint(n_bars // 2, n_bars - 2) if rnd.random() < 0.2 else None
    new_key = key.Key(rnd.choice(["C", "G", "F", "D"])) if key_change_at else None

    score = stream.Score()
    title = f"Synthetic {seed:03d} {ts_str}"
    score.metadata = metadata.Metadata(title=title, composer="digiScore generator")

    # Rhythm per bar: shared in homophonic mode
    def bar_meter(i):
        if change_at is not None and i >= change_at:
            return change_meter
        return (ts_str, symbol, beat, beats)

    shared = [bar_rhythm(rnd, bar_meter(i)[2], bar_meter(i)[3]) for i in range(n_bars)]
    # Last bar: one long note filling it
    lm = bar_meter(n_bars - 1)
    shared[-1] = [lm[2] * lm[3]]

    for vname in names:
        mid, lo, hi, clef_cls = VOICES[vname]
        part = stream.Part(id=vname)
        part.partName = vname
        part.insert(0, instrument.Vocalist())
        cur_key = k
        sc = scale_pitches(cur_key, lo, hi)
        idx = min(range(len(sc)), key=lambda i: abs(sc[i].midi - pitch.Pitch(mid).midi))
        rhythms = shared if homophonic else [
            bar_rhythm(rnd, bar_meter(i)[2], bar_meter(i)[3]) for i in range(n_bars - 1)
        ] + [shared[-1]]
        n_notes = sum(len(r) for r in rhythms)
        verses = [make_lyrics(rnd, lang, n_notes) for _ in range(n_verses)]
        syl_i = 0
        prev_note = None
        for i, rh in enumerate(rhythms):
            m = stream.Measure(number=i + 1)
            ms = bar_meter(i)
            if i == 0:
                m.append(clef_cls())
                m.append(key.KeySignature(cur_key.sharps))
                t = meter.TimeSignature(ms[0])
                if ms[1]:
                    t.symbol = ms[1]
                m.append(t)
            if change_at is not None and i == change_at:
                t = meter.TimeSignature(ms[0])
                if ms[1]:
                    t.symbol = ms[1]
                m.append(t)
            if key_change_at is not None and i == key_change_at:
                cur_key = new_key
                m.append(key.KeySignature(cur_key.sharps))
                sc = scale_pitches(cur_key, lo, hi)
                idx = min(idx, len(sc) - 1)
            steps = melody_line(rnd, sc, idx, len(rh))
            for j, ql in enumerate(rh):
                is_rest = rnd.random() < 0.07 and not (i == len(rhythms) - 1)
                if is_rest:
                    m.append(note.Rest(quarterLength=ql))
                    prev_note = None
                    syl_i += 1
                    continue
                idx = steps[j]
                p = pitch.Pitch(sc[idx].nameWithOctave)
                if rnd.random() < 0.05 and i != len(rhythms) - 1:
                    p = p.transpose(rnd.choice([1, -1]))
                    # Keep the spelling readable
                    p = p.simplifyEnharmonic(mostCommon=True)
                n = note.Note(p, quarterLength=ql)
                # Tie across the barline now and then
                tied_from_prev = (
                    prev_note is not None and j == 0 and i > 0 and rnd.random() < 0.08
                )
                if tied_from_prev:
                    n.pitch = pitch.Pitch(prev_note.pitch.nameWithOctave)
                    prev_note.tie = tie.Tie("start") if prev_note.tie is None else tie.Tie("continue")
                    n.tie = tie.Tie("stop")
                else:
                    for v, lyr in enumerate(verses, start=1):
                        if syl_i < len(lyr):
                            text, kind = lyr[syl_i]
                            n.addLyric(text, lyricNumber=v)
                            n.lyrics[-1].syllabic = kind
                    syl_i += 1
                m.append(n)
                prev_note = n
            if use_repeat and i == 0:
                m.leftBarline = bar.Repeat(direction="start")
            if use_repeat and i == min(7, n_bars - 2):
                m.rightBarline = bar.Repeat(direction="end")
            if i == len(rhythms) - 1:
                m.rightBarline = bar.Barline("final")
            part.append(m)
        score.append(part)
    return score


def find_mscore(explicit=None):
    for c in [explicit, os.environ.get("MSCORE"), "/tmp/opencode/tools/ms/squashfs-root/AppRun",
              shutil.which("mscore4portable"), shutil.which("mscore"), shutil.which("musescore")]:
        if c and os.path.exists(c):
            return c
    return None


def _mscore(mscore, src, dst):
    cmd = [mscore, "-o", dst, src]
    if not os.environ.get("DISPLAY"):
        cmd = ["xvfb-run", "-a"] + cmd
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    return os.path.isfile(dst)


def render(mscore, src_xml, mxl, pdf):
    """PDF and ground-truth MXL both written by MuseScore, so they match exactly."""
    return _mscore(mscore, src_xml, mxl) and _mscore(mscore, mxl, pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/synthetic")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--mscore")
    a = ap.parse_args()
    mscore = find_mscore(a.mscore)
    if not mscore:
        sys.exit("MuseScore 4 not found (set MSCORE or --mscore)")
    os.makedirs(a.out, exist_ok=True)
    for seed in range(a.start, a.start + a.count):
        s = make_score(seed)
        stem = f"synthetic-{seed:03d}-{METERS[seed % len(METERS)][0].replace('/', '-')}"
        src = os.path.abspath(os.path.join(a.out, f".{stem}.musicxml"))
        s.write("musicxml", fp=src)
        mxl = os.path.abspath(os.path.join(a.out, f"{stem}.mxl"))
        ok = render(mscore, src, mxl, os.path.abspath(os.path.join(a.out, f"{stem}.pdf")))
        os.remove(src)
        print(("[+] " if ok else "[-] ") + stem)


if __name__ == "__main__":
    main()
