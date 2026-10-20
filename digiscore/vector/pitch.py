"""
Pitch calculation from geometric coordinates and clef context.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DIATONIC = ["C", "D", "E", "F", "G", "A", "B"]

# Circle of fifths accidentals order:
# Sharps: F, C, G, D, A, E, B
SHARPS_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
# Flats: B, E, A, D, G, C, F
FLATS_ORDER = ["B", "E", "A", "D", "G", "C", "F"]


def get_key_signature_alterations(fifths: int) -> Dict[str, int]:
    """
    Returns map of step -> alter for the given key signature (fifths).
    fifths > 0: sharps (+1)
    fifths < 0: flats (-1)
    """
    alters = {}
    if fifths > 0:
        for s in SHARPS_ORDER[:fifths]:
            alters[s] = 1
    elif fifths < 0:
        for s in FLATS_ORDER[:-fifths]:
            alters[s] = -1
    return alters


@dataclass
class Pitch:
    step: str           # C, D, E, F, G, A, B
    octave: int         # e.g. 4
    alter: int = 0      # -2=double flat, -1=flat, 0=natural, 1=sharp, 2=double sharp
    accidental: Optional[str] = None  # "sharp", "flat", "natural", etc.


def calculate_pitch_from_y(
    orig_y: float,
    staff_bottom_y: float,
    line_spacing: float,
    clef_sign: str = "G",
    clef_line: int = 2,
    clef_octave: int = 0,
    explicit_accidental: Optional[str] = None,
    key_fifths: int = 0,
    measure_accidental_state: Optional[Dict[str, int]] = None,
) -> Pitch:
    """
    Calculates exact pitch from vertical coordinate relative to staff bottom line (Line 1).
    """
    step_size = line_spacing / 2.0
    # delta: 0 = Line 1, 1 = Space 1, 2 = Line 2, etc.
    delta = int(round((staff_bottom_y - orig_y) / step_size))

    # Base pitch of line 1 (index in DIATONIC, octave)
    if clef_sign == "G":
        # G on line 2 (step 2) -> line 1 (step 0) is E4
        base_step_idx = 2  # E
        base_octave = 4 + clef_octave
    elif clef_sign == "F":
        # F on line 4 (step 6) -> line 1 (step 0) is G2
        base_step_idx = 4  # G
        base_octave = 2 + clef_octave
    elif clef_sign == "C":
        # C on clef_line (e.g. line 3 = step 4) -> line 1 (step 0) is F3
        base_step_idx = 3  # F
        base_octave = 3 + clef_octave
    else:
        base_step_idx = 2
        base_octave = 4

    total_steps = base_step_idx + delta
    step_name = DIATONIC[total_steps % 7]
    octave = base_octave + (total_steps // 7)

    # Determine alteration
    alter = 0
    acc_type = None

    if explicit_accidental:
        if "DoubleSharp" in explicit_accidental:
            alter = 2
            acc_type = "double-sharp"
        elif "Sharp" in explicit_accidental:
            alter = 1
            acc_type = "sharp"
        elif "DoubleFlat" in explicit_accidental:
            alter = -2
            acc_type = "flat-flat"
        elif "Flat" in explicit_accidental:
            alter = -1
            acc_type = "flat"
        elif "Natural" in explicit_accidental:
            alter = 0
            acc_type = "natural"
            
        if measure_accidental_state is not None:
            measure_accidental_state[(step_name, octave)] = alter
    else:
        # Check measure accidentals state first, then key signature
        if measure_accidental_state is not None and (step_name, octave) in measure_accidental_state:
            alter = measure_accidental_state[(step_name, octave)]
        else:
            key_alters = get_key_signature_alterations(key_fifths)
            alter = key_alters.get(step_name, 0)

    return Pitch(
        step=step_name,
        octave=octave,
        alter=alter,
        accidental=acc_type,
    )
