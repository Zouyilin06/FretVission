"""Tests for hybrid YIN + Harmonic NNLS guitar pitch detection."""

from __future__ import annotations

import numpy as np

from fretvision.audio.pitch_detector import HybridHarmonicPitchDetector, yin_pitch
from fretvision.theory.music import midi_to_freq


def _synth_guitar_signal(
    midis: list[float],
    sample_rate: int = 44100,
    n_samples: int = 4096,
    harmonics: int = 5,
) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    sig = np.zeros(n_samples, dtype=np.float64)
    for m in midis:
        f0 = midi_to_freq(float(m))
        for h in range(1, harmonics + 1):
            sig += (1.0 / (h ** 0.72)) * np.sin(2.0 * np.pi * f0 * h * t)
    sig /= max(1.0, np.max(np.abs(sig)))
    return (0.4 * sig).astype(np.float32)


def test_yin_monophonic_and_bend_cents() -> None:
    sr = 44100
    # E2 (82.41 Hz) with harmonics
    e2 = _synth_guitar_signal([40.0], sample_rate=sr, n_samples=4096)
    f_e2, conf_e2 = yin_pitch(e2, sr)
    assert f_e2 is not None
    assert conf_e2 > 0.85
    assert abs(f_e2 - midi_to_freq(40.0)) < 0.6

    # G3 bent +40 cents (MIDI 55.4)
    bent_g3 = _synth_guitar_signal([55.4], sample_rate=sr, n_samples=4096)
    f_b, conf_b = yin_pitch(bent_g3, sr)
    assert f_b is not None
    assert conf_b > 0.85
    assert abs(f_b - midi_to_freq(55.4)) < 1.0


def test_hybrid_detector_single_notes_and_silence() -> None:
    detector = HybridHarmonicPitchDetector(noise_gate_db=-50.0)
    sr = 44100

    # Silence check
    quiet = np.random.default_rng(1).normal(0.0, 1e-4, 4096).astype(np.float32)
    res_q = detector.analyze(quiet, sr)
    assert res_q.is_silent
    assert res_q.pitches == ()

    # Low E2 (40), D3 (50), E4 (64)
    for target_midi in (40, 45, 50, 55, 59, 64, 72):
        sig = _synth_guitar_signal([float(target_midi)], sample_rate=sr)
        res = detector.analyze(sig, sr)
        assert not res.is_silent
        detected_midis = [p.midi for p in res.pitches]
        assert target_midi in detected_midis


def test_hybrid_detector_polyphonic_chords() -> None:
    detector = HybridHarmonicPitchDetector(noise_gate_db=-50.0)

    for sr in (44100, 48000):
        # E minor 6-string chord (E2, B2, E3, G3, B3, E4)
        em_notes = {40, 47, 52, 55, 59, 64}
        em_sig = _synth_guitar_signal([float(m) for m in sorted(em_notes)], sample_rate=sr)
        res_em = detector.analyze(em_sig, sr)
        assert {p.midi for p in res_em.pitches} == em_notes

        # C major chord (C3, E3, G3, C4, E4)
        c_notes = {48, 52, 55, 60, 64}
        c_sig = _synth_guitar_signal([float(m) for m in sorted(c_notes)], sample_rate=sr)
        res_c = detector.analyze(c_sig, sr)
        assert {p.midi for p in res_c.pitches} == c_notes

        # D major chord (D3, A3, D4, F#4)
        d_notes = {50, 57, 62, 66}
        d_sig = _synth_guitar_signal([float(m) for m in sorted(d_notes)], sample_rate=sr)
        res_d = detector.analyze(d_sig, sr)
        assert {p.midi for p in res_d.pitches} == d_notes

        # F barre chord (F2, C3, F3, A3, C4, F4)
        f_notes = {41, 48, 53, 57, 60, 65}
        f_sig = _synth_guitar_signal([float(m) for m in sorted(f_notes)], sample_rate=sr)
        res_f = detector.analyze(f_sig, sr, visual_pitch_priors={m: 0.9 for m in f_notes})
        assert {p.midi for p in res_f.pitches} == f_notes
