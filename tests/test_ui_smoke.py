"""Headless PySide6 smoke test verifying MainWindow instantiation and scene updates."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from fretvision.audio.pitch_detector import AudioAnalysisResult, DetectedPitch
from fretvision.fusion.av_fusion import ActiveNote, FusedFrameResult
from fretvision.fusion.tab_recorder import TabColumn
from fretvision.theory.key_detector import KeyCandidate, KeyEstimate
from fretvision.theory.music import ChordMatch
from fretvision.ui.main_window import MainWindow
from fretvision.ui.workers import EngineSnapshot
from fretvision.vision.fretboard_model import FretboardModel
from fretvision.vision.neck_tracker import NeckTrackingStatus


def test_main_window_headless_snapshot_render() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    win = MainWindow(start_workers=False)

    corners = np.array([[60, 100], [580, 100], [580, 220], [60, 220]], dtype=np.float32)
    fb = FretboardModel(corners, start_fret=0, end_fret=12)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    ke = KeyEstimate(
        tonic_pc=7,
        mode="Major",
        display_name="G Major",
        relative_name="E Minor",
        confidence=0.88,
        scale_label="G Major",
        scale_pcs=frozenset({7, 9, 11, 0, 2, 4, 6}),
        top_candidates=(
            KeyCandidate(7, "Major", 0.85),
            KeyCandidate(4, "Minor", 0.72),
            KeyCandidate(0, "Major", 0.55),
        ),
        active_note_count=8,
    )
    notes = (
        ActiveNote(string_idx=0, fret=3, midi=43, note_name="G2", confidence=0.95, source="av_fused", finger_label="2"),
        ActiveNote(string_idx=4, fret=0, midi=59, note_name="B3", confidence=0.92, source="av_fused"),
        ActiveNote(string_idx=5, fret=3, midi=67, note_name="G4", confidence=0.94, source="av_fused", finger_label="3"),
    )
    fused = FusedFrameResult(
        timestamp_s=1.0,
        sounding_notes=notes,
        hover_notes=(),
        chord=ChordMatch(symbol="G", root_pc=7, quality="maj", bass_pc=7, confidence=0.92),
        key_estimate=ke,
        rms_db=-20.0,
        is_onset=True,
        dominant_cents=4.0,
    )
    snap = EngineSnapshot(
        frame_bgr=frame,
        fretboard=fb,
        hand_result=None,
        neck_status=NeckTrackingStatus.LOCKED,
        tracked_points=32,
        fused=fused,
        tab_columns=[
            TabColumn(timestamp_s=1.0, frets={0: 3, 4: 0, 5: 3}, chord_label="G", bar_idx=0, grid_slot=0),
            TabColumn(timestamp_s=2.0, frets={5: 5}, techniques={5: "h"}, bar_idx=1, grid_slot=0),
        ],
        video_fps=30.0,
        audio_latency_ms=2.1,
        tuning_name="Standard (EADGBE)",
        demo_mode=False,
        clip_active=True,
        clip_position_s=12.0,
        clip_duration_s=45.0,
        clip_playing=True,
    )

    win._on_snapshot(snap)
    assert win.key_hero_lbl.text() == "G Major"
    assert win.chord_badge_lbl.text() == "G"
    assert "G2" in win.note_hero_lbl.text()
    assert win.key_conf_bar.value() == 88
    assert not win.clip_bar.isHidden()
    assert win.clip_time_lbl.text() == "00:12 / 00:45"

    win.tab_widget.quantize_chk.setChecked(True)
    win.tab_widget.bpm_spin.setValue(120)
    assert win.worker.tab_recorder.mode == "quantized"
    assert win.worker.tab_recorder.bpm == 120

    # Render widgets offscreen to verify QPainter code paths don't raise
    win.resize(1280, 800)
    pixmap = win.grab()
    assert pixmap.width() > 0 and pixmap.height() > 0

    # Unused imports check sanity
    _ = AudioAnalysisResult(pitches=(DetectedPitch(67, 392.0, 0.0, 0.9),), rms_db=-20.0, is_onset=True, is_silent=False, dominant_cents=0.0)
    win.close()
