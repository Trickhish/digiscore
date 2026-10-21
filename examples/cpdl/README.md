# CPDL choral scores

Real choral scores from the Choral Public Domain Library (https://www.cpdl.org),
each with the PDF and the MusicXML file the editor published next to it.
They come from many notation programs (MuseScore 2/3, Sibelius, Finale,
Dorico, LilyPond, capella, printer drivers) and test the vector path on
fonts and layouts other than those in `examples/synthetic/`.

The reference MusicXML is the editor's own export, so it can differ from the
PDF in small ways (a tenor clef written as a plain treble clef, "-" entered as
a lyric); `tools/bench_synthetic.py` allows for these.

Each edition's page on CPDL gives its editor and licence (most are the CPDL
licence). Pages: search the title on https://www.cpdl.org.

```bash
python tools/bench_synthetic.py --dir examples/cpdl
```
