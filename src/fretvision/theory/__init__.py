"""Music theory utilities, tuning definitions, chord naming, and key detection."""

from .key_detector import KeyCandidate, KeyEstimate, KeyEstimator
from .music import (
    NOTE_NAMES_SHARP,
    TUNING_PRESETS,
    ChordMatch,
    GuitarPosition,
    base_open_midis,
    effective_open_midis,
    freq_to_midi_and_cents,
    get_valid_positions,
    midi_to_freq,
    midi_to_name,
    midi_to_pc,
    pc_to_name,
    recognize_chord,
)

__all__ = [
    "ChordMatch",
    "GuitarPosition",
    "KeyCandidate",
    "KeyEstimate",
    "KeyEstimator",
    "NOTE_NAMES_SHARP",
    "TUNING_PRESETS",
    "base_open_midis",
    "effective_open_midis",
    "freq_to_midi_and_cents",
    "get_valid_positions",
    "midi_to_freq",
    "midi_to_name",
    "midi_to_pc",
    "pc_to_name",
    "recognize_chord",
]
