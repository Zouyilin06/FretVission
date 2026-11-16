"""Adversarial edge-case tests for Stage 2A core hardening fixes."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from fretvision.audio.audio_source import AudioRingBuffer
from fretvision.audio.pitch_detector import AudioAnalysisResult, DetectedPitch, HybridHarmonicPitchDetector
from fretvision.config import CalibrationProfile
from fretvision.fusion.av_fusion import AudioVisualFusion
from fretvision.theory.music import midi_to_freq
from fretvision.ui.camera_widget import CameraViewWidget
from fretvision.ui.workers import HandHistoryBuffer
from fretvision.vision.fretboard_model import FretboardModel
from fretvision.vision.hand_tracker import FingerObservation, HandTracker, HandTrackingResult


def _synth(midis: list[float], sr: int = 44100, n: int = 4096) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    sig = sum((1.0 / (h ** 0.72)) * np.sin(2.0 * np.pi * midi_to_freq(m) * h * t) for m in midis for h in range(1, 5))
    sig = sig / max(1.0, float(np.max(np.abs(sig))))
    return (0.4 * sig).astype(np.float32)


def _hand_with_fingers(specs: list[tuple[str, int, int]], landmarks: list[tuple[float, float]] | None = None) -> HandTrackingResult:
    mat = np.zeros((6, 25), dtype=np.float32)
    fingers: list[FingerObservation] = []
    for lbl, s_idx, f_idx in specs:
        mat[s_idx, f_idx] = 0.95
        fingers.append(
            FingerObservation(
                finger_label=lbl,
                pixel_xy=(120.0, 140.0),
                string_float=float(s_idx),
                fret_float=float(f_idx),
                string_idx=s_idx,
                fret_idx=f_idx,
                in_bounds=True,
                confidence=0.95,
            )
        )
    return HandTrackingResult(
        hand_present=True,
        landmarks_px=landmarks or [],
        hand_bboxes=[],
        fingers=fingers,
        press_prob_matrix=mat,
        model_ready=True,
    )


def test_capo_hover_pitch_accuracy() -> None:
    """Hovering at physical fret 5 with capo=2 on Low E produces A2 (45), not G2 (43)."""
    fusion = AudioVisualFusion(tuning_name="Standard (EADGBE)", capo=2)
    hand = _hand_with_fingers([("2", 0, 5)])
    silent = AudioAnalysisResult(pitches=(), rms_db=-70.0, is_onset=False, is_silent=True, dominant_cents=0.0)
    res = fusion.fuse(0.1, silent, hand, fretboard_calibrated=True)
    assert len(res.hover_notes) == 1
    assert res.hover_notes[0].midi == 45
    assert res.hover_notes[0].note_name == "A2"


def test_same_string_higher_fret_blocks_lower_finger() -> None:
    """When two fingers sit on the same string (f5 and f7), f7 wins and f5 is blocked."""
    fusion = AudioVisualFusion(tuning_name="Standard (EADGBE)", capo=0)
    # G string (string_idx=3): index at f5, ring at f7 -> D4 (62) sounds at f7
    hand = _hand_with_fingers([("1", 3, 5), ("3", 3, 7)])
    # Also ensure audio detects D4 (62) and test D4 resolves to (3, 7), not lower strings
    audio = AudioAnalysisResult(
        pitches=(DetectedPitch(62, 293.66, 0.0, 0.92, True),),
        rms_db=-20.0,
        is_onset=True,
        is_silent=False,
        dominant_cents=0.0,
    )
    res = fusion.fuse(0.1, audio, hand, fretboard_calibrated=True)
    assert len(res.sounding_notes) == 1
    assert (res.sounding_notes[0].string_idx, res.sounding_notes[0].fret) == (3, 7)

    # Now if ring lifts and only index presses f5 -> C4 (60) at (3, 5) without higher block
    hand_lift = _hand_with_fingers([("1", 3, 5)])
    audio_c4 = AudioAnalysisResult(
        pitches=(DetectedPitch(60, 261.63, 0.0, 0.92, True),),
        rms_db=-20.0,
        is_onset=False,
        is_silent=False,
        dominant_cents=0.0,
    )
    res_p = fusion.fuse(0.25, audio_c4, hand_lift, fretboard_calibrated=True)
    assert (res_p.sounding_notes[0].string_idx, res_p.sounding_notes[0].fret) == (3, 5)
    assert res_p.sounding_notes[0].technique == "p"


def test_power_chords_and_double_stops_preserved() -> None:
    """Two-note power chords and intervals must NOT be collapsed by YIN monophonic path."""
    det = HybridHarmonicPitchDetector(noise_gate_db=-50.0)
    for pair in ({40, 47}, {45, 52}, {55, 59}, {59, 60}):  # E5, A5, G+B 3rd, B+C minor 2nd
        sig = _synth([float(m) for m in sorted(pair)])
        res = det.analyze(sig, 44100)
        got = {p.midi for p in res.pitches}
        assert pair.issubset(got), f"Expected {pair}, got {got}"


def test_high_frets_detuned_and_muted_noise_rejection() -> None:
    det = HybridHarmonicPitchDetector(noise_gate_db=-50.0)
    # High fret 19 on High E = B5 (83) slightly detuned +18 cents
    sig_high = _synth([83.18], sr=48000)
    res_high = det.analyze(sig_high, 48000)
    assert [p.midi for p in res_high.pitches] == [83]
    assert abs(res_high.dominant_cents - 18.0) < 4.0

    # Broadband white-noise pick scrape should be rejected by flatness gate
    rng = np.random.default_rng(7)
    scrape = rng.normal(0.0, 0.12, 4096).astype(np.float32)
    res_scrape = det.analyze(scrape, 44100)
    assert res_scrape.is_silent
    assert res_scrape.pitches == ()


def test_temporal_glitch_debounce_and_legato_slide() -> None:
    fusion = AudioVisualFusion()
    # Single-frame weak non-onset audio_only blip gets suppressed until confirmed on frame 2
    weak_blip = AudioAnalysisResult(
        pitches=(DetectedPitch(64, 329.63, 0.0, 0.55, True),),
        rms_db=-26.0,
        is_onset=False,
        is_silent=False,
        dominant_cents=0.0,
    )
    r1 = fusion.fuse(0.10, weak_blip, None, fretboard_calibrated=False)
    assert r1.sounding_notes == ()
    r2 = fusion.fuse(0.13, weak_blip, None, fretboard_calibrated=False)
    assert len(r2.sounding_notes) == 1

    # Same-finger slide along D string from fret 5 (G3=55) to fret 7 (A3=57) without pick onset
    fusion.fuse(0.20, AudioAnalysisResult((DetectedPitch(55, 196.0, 0.0, 0.9, True),), -20.0, True, False, 0.0), _hand_with_fingers([("3", 2, 5)]), True)
    res_slide = fusion.fuse(0.32, AudioAnalysisResult((DetectedPitch(57, 220.0, 0.0, 0.9, True),), -21.0, False, False, 0.0), _hand_with_fingers([("3", 2, 7)]), True)
    assert res_slide.sounding_notes[0].technique == "/"


def test_degenerate_calibration_corners_rejected() -> None:
    # Collinear points
    collinear = np.array([[10, 10], [100, 10], [200, 10], [300, 10]], dtype=np.float32)
    with pytest.raises(ValueError):
        FretboardModel(collinear, 0, 12)

    # Bowtie / crossed quad
    bowtie = np.array([[50, 50], [300, 50], [50, 180], [300, 180]], dtype=np.float32)
    with pytest.raises(ValueError):
        FretboardModel(bowtie, 0, 12)


def test_project_to_fretboard_preserves_barre_chord() -> None:
    corners = np.array([[50, 100], [550, 100], [550, 220], [50, 220]], dtype=np.float32)
    fb = FretboardModel(corners, start_fret=0, end_fret=12)
    ht = HandTracker(max_fret=22)

    # Place 21 synthetic landmarks where index joints 6, 7, 8 span strings 0..5 at fret 1
    pts = [(20.0, 20.0)] * 21
    p_top = fb.fret_string_center_px(0, 1)
    p_mid = fb.fret_string_center_px(3, 1)
    p_bot = fb.fret_string_center_px(5, 1)
    pts[6] = p_top
    pts[7] = p_mid
    pts[8] = p_bot
    pts[4] = (p_top[0], p_top[1] - 25.0)
    pts[12] = fb.fret_string_center_px(3, 2)
    pts[16] = fb.fret_string_center_px(1, 3)
    pts[20] = fb.fret_string_center_px(2, 3)

    raw = HandTrackingResult(
        hand_present=True,
        landmarks_px=pts,
        hand_bboxes=[],
        fingers=[
            FingerObservation("1", pts[8], 0.0, 0.0, None, None, False, 0.9),
            FingerObservation("2", pts[12], 0.0, 0.0, None, None, False, 0.9),
            FingerObservation("3", pts[16], 0.0, 0.0, None, None, False, 0.9),
            FingerObservation("4", pts[20], 0.0, 0.0, None, None, False, 0.9),
        ],
        press_prob_matrix=np.zeros((6, 23), dtype=np.float32),
        model_ready=True,
    )
    projected = ht.project_to_fretboard(raw, fb)
    # Barre across all 6 strings at fret 1 must remain intact!
    assert np.all(projected.press_prob_matrix[:, 1] >= 0.70)


def test_audio_ring_delay_and_hand_history_sync() -> None:
    ring = AudioRingBuffer(capacity=1024)
    ring.push(np.arange(500, dtype=np.float32))
    latest = ring.read_latest(100, delay_samples=0)
    delayed = ring.read_latest(100, delay_samples=150)
    assert np.allclose(latest, np.arange(400, 500, dtype=np.float32))
    assert np.allclose(delayed, np.arange(250, 350, dtype=np.float32))

    buf = HandHistoryBuffer(max_age_s=0.5)
    h1 = _hand_with_fingers([("1", 4, 3)])
    h2 = _hand_with_fingers([("1", 4, 5)])
    buf.push(1.00, h1)
    buf.push(1.20, h2)
    assert buf.nearest(1.03) is h1
    assert buf.nearest(1.18) is h2


def test_profile_load_ignores_unknown_keys(tmp_path) -> None:
    p = tmp_path / "legacy.json"
    p.write_text('{"name": "legacy", "obsolete_field": 123, "guitar": {"capo": 3, "old_key": true}}', encoding="utf-8")
    loaded = CalibrationProfile.load("legacy", directory=tmp_path)
    assert loaded.name == "legacy"
    assert loaded.guitar.capo == 3


def test_camera_widget_corner_drag_emits_adjusted_quad() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    w = CameraViewWidget()
    w.resize(640, 360)
    corners = np.array([[80, 100], [520, 100], [520, 220], [80, 220]], dtype=np.float32)
    fb = FretboardModel(corners, 0, 12)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    w.update_scene(frame, fb, None, ())
    w.grab()  # populates _image_rect
    w.set_adjust_mode(True)

    emitted: list[np.ndarray] = []
    w.corners_adjusted.connect(lambda c: emitted.append(c))

    wp0 = w._img_to_widget(80.0, 100.0)
    wp_new = w._img_to_widget(95.0, 110.0)
    w.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress, wp0, wp0, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    w.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove, wp_new, wp_new, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    w.mouseReleaseEvent(QMouseEvent(QMouseEvent.Type.MouseButtonRelease, wp_new, wp_new, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))

    assert len(emitted) == 1
    assert abs(float(emitted[0][0, 0]) - 95.0) < 2.0
    assert abs(float(emitted[0][0, 1]) - 110.0) < 2.0
