"""
OMR module for scanned and raster score recognition using NoteLite / Audiveris.
"""

from digiscore.omr.notelite import run_notelite_omr, is_notelite_available

__all__ = ["run_notelite_omr", "is_notelite_available"]
