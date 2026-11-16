"""Tests for music theory primitives, chord naming, and real-time key estimation."""

from __future__ import annotations

from fretvision.theory.key_detector import KeyEstimator
from fretvision.theory.music import (
    TUNING_PRESETS,
    base_open_midis,
    effective_open_midis,
    freq_to_midi_and_cents,
    get_valid_positions,
    midi_to_freq,
    midi_to_name,
    recognize_chord,
)


def test_midi_frequency_round_trip() -> None:
    assert abs(midi_to_freq(69) - 440.0) < 1e-6
    assert abs(midi_to_freq(40) - 82.406889) < 1e-3

    midi, cents = freq_to_midi_and_cents(440.0)
    assert midi == 69
    assert abs(cents) < 1e-4

    # +50 cents sharp on A4 (~452.89 Hz) -> still near A4 or A#4 boundary
    f_sharp25 = 440.0 * (2.0 ** (0.25 / 12.0))
    m2, c2 = freq_to_midi_and_cents(f_sharp25)
    assert m2 == 69
    assert abs(c2 - 25.0) < 0.2
    assert midi_to_name(40) == "E2"
    assert midi_to_name(60) == "C4"


def test_valid_positions_standard_and_capo() -> None:
    std = base_open_midis("Standard (EADGBE)")
    positions = get_valid_positions(64, std, max_fret=22, capo=0)  # E4
    pairs = {(p.string_idx, p.fret) for p in positions}
    assert pairs == {(5, 0), (4, 5), (3, 9), (2, 14), (1, 19)}

    # With capo on fret 2, sounding F#4 (66) can be played "open" on string 5 at physical fret 2
    capo_positions = get_valid_positions(66, std, max_fret=22, capo=2)
    capo_pairs = {(p.string_idx, p.fret) for p in capo_positions}
    assert (5, 2) in capo_pairs
    assert effective_open_midis("Standard (EADGBE)", capo=2)[0] == 42

    drop_d = TUNING_PRESETS["Drop D (DADGBE)"]
    d_pos = get_valid_positions(38, drop_d, max_fret=22, capo=0)
    assert {(p.string_idx, p.fret) for p in d_pos} == {(0, 0)}


def test_chord_recognition() -> None:
    g_major = recognize_chord([43, 47, 50, 55, 59, 67])
    assert g_major is not None
    assert g_major.symbol == "G"

    e_minor = recognize_chord([40, 47, 52, 55, 59, 64])
    assert e_minor is not None
    assert e_minor.symbol == "Em"

    am7 = recognize_chord([45, 52, 55, 60, 64])
    assert am7 is not None
    assert am7.symbol == "Am7"

    dsus4 = recognize_chord([50, 57, 62, 67])
    assert dsus4 is not None
    assert dsus4.symbol == "Dsus4"

    c_over_e = recognize_chord([40, 48, 52, 55, 60])
    assert c_over_e is not None
    assert c_over_e.symbol == "C/E"


def test_key_estimator_major_progression_and_pentatonic() -> None:
    est = KeyEstimator(half_life_s=8.0)
    # G - Em - C - D progression
    progression = [
        [43, 47, 50, 55, 59, 67],
        [40, 47, 52, 55, 59, 64],
        [48, 52, 55, 60, 64],
        [50, 57, 62, 66],
        [43, 47, 50, 55, 59, 67],
    ]
    t = 0.0
    out = None
    for chord in progression:
        out = est.update(t, chord, is_onset=True)
        t += 0.6
    assert out is not None
    assert out.display_name == "G Major"
    assert out.confidence > 0.45

    # Reset and feed A Minor Pentatonic riff anchored on A2 bass
    est.reset()
    a_minor_notes = [45, 48, 50, 52, 55, 57, 45, 52, 55, 45]
    for idx, m in enumerate(a_minor_notes):
        out = est.update(idx * 0.25, [m], is_onset=True)
    assert out is not None
    assert out.display_name == "A Minor"
    assert "Pentatonic" in out.scale_label


def test_key_estimator_modulation_with_hysteresis() -> None:
    est = KeyEstimator(half_life_s=3.0, dwell_seconds=0.8)
    t = 0.0
    # Establish C Major
    for _ in range(4):
        for chord in ([48, 52, 55, 60], [41, 45, 48, 53], [43, 47, 50, 55], [48, 52, 55, 60]):
            est.update(t, chord, is_onset=True)
            t += 0.35
    assert est.current_estimate(t).display_name == "C Major"  # type: ignore[union-attr]

    # Single chromatic passing tone must NOT flip the key
    est.update(t, [54], is_onset=True)
    t += 0.15
    assert est.current_estimate(t).display_name == "C Major"  # type: ignore[union-attr]

    # Modulate cleanly to E Major (E, A, B7, E) over > 2 seconds
    for _ in range(6):
        for chord in ([40, 47, 52, 56, 59], [45, 52, 57, 61, 64], [47, 54, 57, 59, 63], [40, 47, 52, 56, 59]):
            est.update(t, chord, is_onset=True)
            t += 0.35
    assert est.current_estimate(t).display_name == "E Major"  # type: ignore[union-attr]
