"""
digiScore - Digitize singing score PDFs into compliant MusicXML 4.0 files.
"""

import os
import sys
import warnings

# Suppress onnxruntime C-level thread affinity log noise during initialization
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    _stderr_fd = sys.stderr.fileno()
    with open(os.devnull, "w") as _null:
        _null_fd = _null.fileno()
        _saved_stderr = os.dup(_stderr_fd)
        os.dup2(_null_fd, _stderr_fd)
        try:
            import pymupdf
        finally:
            os.dup2(_saved_stderr, _stderr_fd)
            os.close(_saved_stderr)
except Exception:
    import pymupdf

from digiscore.pipeline import convert_score, is_vector_score

__version__ = "0.1.0"
__all__ = ["convert_score", "is_vector_score"]


