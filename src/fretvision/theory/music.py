"""Core music theory primitives: tunings, note conversions, fretboard mapping, and chord recognition.

String indexing convention used across the entire FretVision codebase:
    string_idx 0 = 6th string (thickest / Low E2 in standard tuning)
    string_idx 1 = 5th string (A2)
    string_idx 2 = 4th string (D3)
    string_idx 3 = 3rd string (G3)
    string_idx 4 = 2nd string (B3)
    string_idx 5 = 1st string (thinnest / High E4 in standard tuning)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

NOTE_NAMES_SHARP: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

# MIDI values ordered from string 0 (6th/lowest) to string 5 (1st/highest).
TUNING_PRESETS: dict[str, tuple[int, int, int, int, int, int]] = {
    "Standard (EADGBE)": (40, 45, 50, 55, 59, 64),
    "Drop D (DADGBE)": (38, 45, 50, 55, 59, 64),
    "Eb Standard (D#G#C#F#A#D#)": (39, 44, 49, 54, 58, 63),
    "D Standard (DGCFAD)": (38, 43, 48, 53, 57, 62),
    "Drop C (CGCFAD)": (36, 43, 48, 53, 57, 62),
    "DADGAD": (38, 45, 50, 55, 57, 62),
    "Open G (DGDGBD)": (38, 43, 50, 55, 59, 62),
    "Open D (DADF#AD)": (38, 45, 50, 54, 57, 62),
}


@dataclass(frozen=True)
class GuitarPosition:
    """Physical location of a pitch on the guitar neck."""

    string_idx: int  # 0 (6th/low) .. 5 (1st/high)
    fret: int        # 0 (open relative to capo) .. max_fret
    midi: int

    @property
    def string_number(self) -> int:
        """Human 1-based guitar string number (1 = High E, 6 = Low E)."""
        return 6 - self.string_idx

    @property
    def note_name(self) -> str:
        """Scientific pitch name (e.g., E2, G#3)."""
        return midi_to_name(self.midi)


@dataclass(frozen=True)
class ChordMatch:
    """Best recognized chord symbol for a set of active MIDI notes."""

    symbol: str
    root_pc: int
    quality: str
    bass_pc: int
    confidence: float


# (chord_suffix, required_intervals, optional_intervals, penalty_weight)
_CHORD_TEMPLATES: tuple[tuple[str, frozenset[int], frozenset[int], float], ...] = (
    ("", frozenset({0, 4, 7}), frozenset(), 0.0),
    ("m", frozenset({0, 3, 7}), frozenset(), 0.0),
    ("5", frozenset({0, 7}), frozenset(), 0.06),
    ("7", frozenset({0, 4, 10}), frozenset({7}), 0.03),
    ("maj7", frozenset({0, 4, 11}), frozenset({7}), 0.03),
    ("m7", frozenset({0, 3, 10}), frozenset({7}), 0.03),
    ("mMaj7", frozenset({0, 3, 11}), frozenset({7}), 0.08),
    ("sus2", frozenset({0, 2, 7}), frozenset(), 0.04),
    ("sus4", frozenset({0, 5, 7}), frozenset(), 0.04),
    ("dim", frozenset({0, 3, 6}), frozenset(), 0.05),
    ("dim7", frozenset({0, 3, 6, 9}), frozenset(), 0.07),
    ("m7b5", frozenset({0, 3, 6, 10}), frozenset(), 0.06),
    ("aug", frozenset({0, 4, 8}), frozenset(), 0.07),
    ("6", frozenset({0, 4, 9}), frozenset({7}), 0.06),
    ("m6", frozenset({0, 3, 9}), frozenset({7}), 0.06),
    ("add9", frozenset({0, 2, 4, 7}), frozenset(), 0.05),
    ("madd9", frozenset({0, 2, 3, 7}), frozenset(), 0.05),
    ("9", frozenset({0, 2, 4, 10}), frozenset({7}), 0.06),
    ("m9", frozenset({0, 2, 3, 10}), frozenset({7}), 0.06),
    ("7sus4", frozenset({0, 5, 7, 10}), frozenset(), 0.06),
)


def midi_to_freq(midi: float) -> float:
    """Convert MIDI note number (fractional allowed) to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def freq_to_midi_and_cents(freq_hz: float) -> tuple[int, float]:
    """Convert frequency in Hz to nearest integer MIDI note and signed cents offset."""
    if freq_hz <= 0.0:
        raise ValueError(f"Frequency must be positive, got {freq_hz}")
    midi_float = 69.0 + 12.0 * math.log2(freq_hz / 440.0)
    nearest_midi = int(round(midi_float))
    cents = (midi_float - nearest_midi) * 100.0
    return nearest_midi, cents


def midi_to_pc(midi: int) -> int:
    """Return pitch class 0..11 (0 = C)."""
    return int(midi) % 12


def pc_to_name(pc: int) -> str:
    """Return pitch class name (e.g., 0 -> 'C', 6 -> 'F#')."""
    return NOTE_NAMES_SHARP[int(pc) % 12]


def midi_to_name(midi: int, include_octave: bool = True) -> str:
    """Convert MIDI number to note name (e.g., 60 -> 'C4', 40 -> 'E2')."""
    pc = int(midi) % 12
    octave = (int(midi) // 12) - 1
    name = NOTE_NAMES_SHARP[pc]
    return f"{name}{octave}" if include_octave else name


def base_open_midis(
    tuning_name: str = "Standard (EADGBE)",
    custom_tuning: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Return open string MIDI notes (0=Low E .. 5=High E) with no capo applied."""
    return (
        custom_tuning
        if custom_tuning is not None
        else TUNING_PRESETS.get(tuning_name, TUNING_PRESETS["Standard (EADGBE)"])
    )


def effective_open_midis(
    tuning_name: str = "Standard (EADGBE)",
    capo: int = 0,
    custom_tuning: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Return the effective sounding MIDI note for open string 0..5 with capo applied."""
    base = base_open_midis(tuning_name, custom_tuning)
    capo_clamped = max(0, min(12, int(capo)))
    return tuple(m + capo_clamped for m in base)  # type: ignore[return-value]


def get_valid_positions(
    midi_note: int,
    base_midis: tuple[int, int, int, int, int, int],
    max_fret: int = 22,
    capo: int = 0,
) -> list[GuitarPosition]:
    """Enumerate all playable (string_idx, physical_fret) pairs producing midi_note.

    Args:
        midi_note: Target sounding MIDI pitch.
        base_midis: Un-capoed open string MIDI notes (string 0..5).
        max_fret: Highest fret on the guitar neck.
        capo: Capo fret (0 = no capo). Note that playing the open string above
            a capo at fret C is represented as physical fret C.
    """
    capo_clamped = max(0, min(12, int(capo)))
    positions: list[GuitarPosition] = []
    for string_idx, open_midi in enumerate(base_midis):
        capo_midi = int(open_midi) + capo_clamped
        delta = int(midi_note) - capo_midi
        if delta >= 0:
            physical_fret = capo_clamped + delta
            if physical_fret <= max_fret:
                positions.append(
                    GuitarPosition(
                        string_idx=string_idx,
                        fret=physical_fret,
                        midi=int(midi_note),
                    )
                )
    return positions


def recognize_chord(midis: list[int] | set[int] | tuple[int, ...]) -> ChordMatch | None:
    """Identify the most likely guitar chord symbol from active MIDI pitches."""
    unique_midis = sorted({int(m) for m in midis})
    if len(unique_midis) < 2:
        return None

    pcs = {m % 12 for m in unique_midis}
    if len(pcs) < 2:
        return None

    bass_pc = unique_midis[0] % 12
    best: ChordMatch | None = None
    best_score = -1e9

    for root in sorted(pcs):
        rel_pcs = frozenset((pc - root) % 12 for pc in pcs)
        for suffix, required, optional, penalty in _CHORD_TEMPLATES:
            if not required.issubset(rel_pcs):
                continue
            allowed = required | optional
            extra = rel_pcs - allowed
            # Penalize unexplained extra tones heavily, reward bass == root
            bass_bonus = 0.28 if root == bass_pc else 0.0
            # Slightly prefer 3+ note chords over power-chord subset if all tones fit
            richness = 0.08 * len(required)
            extra_penalty = 0.35 * len(extra)
            score = 1.0 + bass_bonus + richness - penalty - extra_penalty
            if len(extra) > 1:
                continue

            if score > best_score:
                best_score = score
                root_name = pc_to_name(root)
                symbol = f"{root_name}{suffix}"
                if root != bass_pc:
                    symbol = f"{symbol}/{pc_to_name(bass_pc)}"
                confidence = max(0.1, min(1.0, score / 1.4))
                best = ChordMatch(
                    symbol=symbol,
                    root_pc=root,
                    quality=suffix or "maj",
                    bass_pc=bass_pc,
                    confidence=confidence,
                )

    return best
