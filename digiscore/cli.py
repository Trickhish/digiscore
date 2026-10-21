"""
CLI entrypoint for digiScore.
"""

import os
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import sys
from pathlib import Path

from digiscore.pipeline import convert_score, is_vector_score


def main():
    parser = argparse.ArgumentParser(
        prog="digiscore",
        description="Digitize singing score PDFs (vector and scanned) into compliant MusicXML 4.0 files.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the score PDF file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path for output file (.musicxml or .mxl). Defaults to <input_name>.musicxml",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["auto", "vector", "notelite"],
        default="auto",
        help="Processing engine to use (default: auto - detects vector vs scanned/raster).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI-based metadata extraction and lyric verification.",
    )
    parser.add_argument(
        "--sheets",
        type=str,
        default=None,
        help="Sheet range for scanned OMR processing (e.g. '1' or '1-3').",
    )
    parser.add_argument(
        "--choir",
        action="store_true",
        help="Choir mode: keep the sung parts (SATB, SA/TB...) and drop the accompaniment.",
    )
    parser.add_argument(
        "--split-voices",
        action="store_true",
        help="One part per voice: staves shared by two voices (SA, TB) become two parts.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        is_vec = is_vector_score(str(input_path))
        print(f"[*] Input: {input_path}")
        print(f"[*] Detected format: {'Vector PDF' if is_vec else 'Scanned/Raster PDF'}")
        print(f"[*] Engine: {args.engine} (effective: {'vector' if is_vec and args.engine == 'auto' else ('notelite' if args.engine == 'auto' else args.engine)})")
        print(f"[*] AI enrichment: {'Disabled' if args.no_ai else 'Enabled'}")

    try:
        out_file = convert_score(
            input_path=str(input_path),
            output_path=args.output,
            engine=args.engine,
            use_ai=not args.no_ai,
            sheets=args.sheets,
            choir=args.choir,
            split_voices=args.split_voices,
        )
        print(f"[+] Successfully generated: {out_file}")
    except Exception as e:
        print(f"[-] Conversion failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
