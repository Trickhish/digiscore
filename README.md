# digiScore

A modern, high-accuracy singing score digitizer that transforms PDF sheet music (both native vector PDFs and scanned/raster sheets) into standard MusicXML 4.0 (`.musicxml` or `.mxl`) files.

## Features

- **Dual-Engine Architecture**:
  - **Native Vector Pipeline**: Mathematical glyph & geometry extraction for vector PDFs (MuseScore, Sibelius, Finale, LilyPond, Dorico). Preserves 100% notehead, clef, pitch, duration, and multi-verse lyric precision.
  - **NoteLite / Audiveris OMR Batch Engine**: Integrated headless optical music recognition for scanned, historical, or raster scores with Tesseract OCR support.
- **Multi-Verse Choral Lyrics**: Precise horizontal alignment of syllables to noteheads with support for multi-line verses and divisi/SATB systems.
- **Key & Time Signatures**: Automatic detection of clefs (Treble, Bass, Tenor, clef-octave), accidentals, key alterations, and time signatures.
- **AI Enhancement**: Seamless metadata extraction (title, composer, lyricist, arranger, tempo, parts) and lyrics verification via `ai.dury.dev`.
- **Compliant MusicXML 4.0 & MXL**: Full compatibility with MuseScore, Dorico, Finale, and Sibelius.

## Installation

```bash
git clone <repo-url>
cd digiScore
pip install -e .
```

### NoteLite OMR Setup (for Scanned Scores)

digiScore automatically detects and uses NoteLite / Audiveris when available:
- Place NoteLite in `/tmp/opencode/tools/app-5.13.0/bin/NoteLite` or export `NOTELITE_BIN=/path/to/NoteLite`.
- Tesseract legacy OCR models can be configured via `TESSDATA_PREFIX=/opt/tessdata`.

## CLI Usage

### Convert a Vector Score
```bash
digiscore path/to/score.pdf -o output.musicxml
```

### Generate Compressed `.mxl` Format
```bash
digiscore path/to/score.pdf -o output.mxl
```

### Convert a Scanned Score (Using NoteLite OMR)
```bash
digiscore path/to/scanned_score.pdf -o output.musicxml --sheets 1
```

### Engine Selection & Options
- `--engine {auto,vector,notelite}`: Choose specific engine or let digiScore auto-detect (default: `auto`).
- `--no-ai`: Disable AI metadata extraction and lyrics verification.
- `--sheets SHEETS`: Specify sheet numbers/ranges for OMR (e.g. `1` or `1-3`).
- `-v, --verbose`: Display verbose progress output.

## Python API

```python
from digiscore import convert_score, is_vector_score

# Check if score is vector or raster
if is_vector_score("score.pdf"):
    print("Vector PDF detected")

# Convert score to MusicXML
output_path = convert_score(
    input_path="score.pdf",
    output_path="score.musicxml",
    engine="auto",
    use_ai=True
)
```

## Running Tests

```bash
pytest -v tests/
```

## Synthetic ground truth

`examples/synthetic/` holds generated scores (various meters including cut and
common time, 1-4 voices, multiple verses, ties, repeats, key and time changes)
with the MusicXML that MuseScore 4 rendered them from:

```bash
python tools/make_synthetic.py --count 20          # needs MuseScore 4 (MSCORE=path) and xvfb-run
python tools/bench_synthetic.py                    # vector path
python tools/bench_synthetic.py --scan 130 --engine notelite   # same scores as 130 DPI fake scans
python tools/bench_synthetic.py --dir examples/cpdl          # real choral scores from CPDL
python tools/bench_synthetic.py --dir examples/musetrainer   # piano scores
```
