"""Split-voice check: each part of a split SA/TB staff against the matching voice of the reference."""
import sys, glob, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'tools')
import pymupdf, music21
from bench_synthetic import ratio
from digiscore.vector.parser import parse_vector_score
from digiscore.vector.choir import apply_choir_mode
from digiscore.vector.split import split_voices
from digiscore.writer.musicxml import build_musicxml

def seq_pred(part, vid):
    return [('R' if n.isRest else sorted(p.midi for p in n.pitches)[-1 if vid == 0 else 0], float(n.quarterLength)) for n in part.flatten().notesAndRests
            if not (n.isRest and n.quarterLength == 0)]

def ref_voice(part, vid):
    out = []
    for m in part.getElementsByClass('Measure'):
        vs = list(m.voices)
        if not vs:
            src = list(m.notesAndRests)
            evs = []
            for n in src:
                if n.isChord:
                    ps = sorted(p.midi for p in n.pitches)
                    evs.append((ps[-1] if vid == 0 else ps[0], float(n.quarterLength)))
                else:
                    evs.append(('R' if n.isRest else n.pitch.midi, float(n.quarterLength)))
        else:
            v = vs[min(vid, len(vs) - 1)]
            evs = []
            for n in v.notesAndRests:
                if n.isRest and n.style.hideObjectOnPrint: continue
                if n.isChord:
                    ps = sorted(p.midi for p in n.pitches); evs.append((ps[-1] if vid == 0 else ps[0], float(n.quarterLength)))
                else: evs.append(('R' if n.isRest else n.pitch.midi, float(n.quarterLength)))
        out.extend(evs)
    return out

def merge_rests(s):
    out = []
    for p, q in s:
        if p == 'R' and out and out[-1][0] == 'R': out[-1] = ('R', out[-1][1] + q)
        else: out.append((p, q))
    return out

def main():
    tot = []
    for f in sorted(glob.glob('examples/cpdl/*.pdf')):
        if len(sys.argv) > 1 and sys.argv[1] not in f: continue
        try: base = apply_choir_mode(parse_vector_score(pymupdf.open(f)))
        except Exception: continue
        if 'parts' not in base: continue
        sd = split_voices(base)
        if len(sd['parts']) != 2 * len(base['parts']) or len(base['parts']) != 2: continue
        open('/tmp/opencode/sp.musicxml', 'w').write(build_musicxml(sd))
        S = music21.converter.parse('/tmp/opencode/sp.musicxml')
        G = music21.converter.parse(f[:-4] + '.mxl')
        rs = []
        for i in range(4):
            a = merge_rests(seq_pred(S.parts[i], i % 2)); b = merge_rests(ref_voice(G.parts[i // 2], i % 2))
            rs.append(round(max(ratio(a, b), ratio([(p if p == 'R' else p + 12, q) for p, q in a], b), ratio([(p if p == 'R' else p - 12, q) for p, q in a], b)), 3))
        tot.append(min(rs))
        print(f.split('/')[-1][:30], [p.partName for p in S.parts], rs)
    print('mean worst', round(sum(tot) / len(tot), 3))


if __name__ == "__main__":
    main()
