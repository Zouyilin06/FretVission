"""Unit and widget tests for Stage 3 FretVision UI components and layout."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fretvision.audio.pitch_detector import HybridHarmonicPitchDetector
from fretvision.fusion.av_fusion import ActiveNote
from fretvision.theory.key_detector import KeyCandidate, KeyEstimate
from fretvision.theory.music import midi_to_freq
from fretvision.ui import fretboard_widget
from fretvision.ui.fretboard_widget import VirtualFretboardWidget
from fretvision.ui.main_window import MainWindow
from fretvision.ui.telemetry_widgets import (
    ChordHistoryStrip,
    MiniSpectrumWidget,
    PrecisionTunerGauge,
)
from fretvision.ui.theme import PALETTE, get_stylesheet


def _app() -> QApplication:
    inst = QApplication.instance()
    return inst if isinstance(inst, QApplication) else QApplication([])


def test_chord_history_strip_progression_and_clear() -> None:
    _app()
    strip = ChordHistoryStrip(max_chords=8)

    # Non-onset noise does not prematurely commit if consecutive count < 2
    assert strip.ingest_chord("G", is_onset=False) is False
    assert strip.chords == []

    # Confirmed onset sequence avoids consecutive duplicate entries
    assert strip.ingest_chord("G", is_onset=True) is True
    assert strip.ingest_chord("G", is_onset=True) is False
    assert strip.ingest_chord("Em", is_onset=True) is True
    assert strip.ingest_chord("C", is_onset=True) is True
    assert strip.ingest_chord("D", is_onset=True) is True
    assert strip.chords == ["G", "Em", "C", "D"]

    # Bounded FIFO length
    for sym in ("Am", "F", "Dm", "G7", "Cmaj7"):
        strip.ingest_chord(sym, is_onset=True)
    assert len(strip.chords) == 8
    assert strip.chords[-1] == "Cmaj7"

    strip.clear()
    assert strip.chords == []


def test_spectrum_bands_and_precision_tuner_gauge() -> None:
    _app()
    detector = HybridHarmonicPitchDetector(noise_gate_db=-50.0)
    sr = 44100
    n_samples = 4096

    quiet = np.zeros(n_samples, dtype=np.float32)
    res_quiet = detector.analyze(quiet, sr)
    assert res_quiet.is_silent
    assert res_quiet.spectrum_bands == ()

    t = np.arange(n_samples, dtype=np.float64) / sr
    f0 = midi_to_freq(57.0)  # A3
    tone = (0.4 * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)
    res_tone = detector.analyze(tone, sr)
    assert len(res_tone.spectrum_bands) == 24
    assert all(0.0 <= b <= 1.0 for b in res_tone.spectrum_bands)
    assert max(res_tone.spectrum_bands) > 0.25

    spectrum = MiniSpectrumWidget(n_bands=24)
    spectrum.resize(260, 44)
    spectrum.set_bands(res_quiet.spectrum_bands)
    spectrum.repaint()
    spectrum.set_bands(res_tone.spectrum_bands)
    spectrum.repaint()

    gauge = PrecisionTunerGauge()
    gauge.resize(260, 42)
    gauge.set_state(-14.5, True, "A3")
    gauge.repaint()
    gauge.set_state(0.0, False)
    gauge.repaint()


def test_fretboard_degrees_palette_and_pinning() -> None:
    app = _app()
    assert PALETTE["audio_only"] == "#A855F7"
    assert PALETTE["accent_amber"] == "#F59E0B"
    assert PALETTE["audio_only"].lower() != PALETTE["accent_amber"].lower()

    fb = VirtualFretboardWidget()
    fb.set_guitar_setup("Standard (E A D G B E)", capo=0, handedness="right", max_fret=15)
    fb.resize(960, 220)
    fb.show()
    app.processEvents()

    g_major = KeyEstimate(
        tonic_pc=7,
        mode="Major",
        display_name="G Major",
        relative_name="E Minor",
        confidence=0.92,
        scale_label="G Major",
        scale_pcs=frozenset({7, 9, 11, 0, 2, 4, 6}),
        top_candidates=(KeyCandidate(7, "Major", 0.92),),
        active_note_count=12,
    )
    sounding = (
        ActiveNote(string_idx=0, fret=3, midi=43, note_name="G2", confidence=0.95, source="av_fused"),
        ActiveNote(string_idx=1, fret=5, midi=50, note_name="D3", confidence=0.88, source="audio_only"),
    )
    fb.update_state(sounding, (), g_major)
    app.processEvents()

    # Check interval vocabulary & toggle between Notes and Degrees
    assert fretboard_widget._INTERVAL_NAMES[0] == "R"
    assert fretboard_widget._INTERVAL_NAMES[7] == "5"
    assert fb.notes_btn.isChecked()

    img_notes = fb.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    modes_seen: list[str] = []
    fb.label_mode_changed.connect(modes_seen.append)

    fb.degrees_btn.click()
    app.processEvents()
    assert fb.degrees_btn.isChecked()
    assert not fb.notes_btn.isChecked()
    assert modes_seen == ["degrees"]
    img_degrees = fb.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    assert img_notes != img_degrees

    # Locate a clickable fretboard cell and verify click-to-pin / toggle / right-click clear
    hit_pos: QPoint | None = None
    hit_cell: tuple[int, int] | None = None
    for x in range(40, fb.canvas.width() - 20, 15):
        for y in range(24, fb.canvas.height() - 16, 12):
            cell = fb.canvas._cell_at_widget_pos(x, y)
            if cell is not None:
                hit_pos = QPoint(x, y)
                hit_cell = cell
                break
        if hit_pos is not None:
            break

    assert hit_pos is not None and hit_cell is not None
    QTest.mouseClick(fb.canvas, Qt.MouseButton.LeftButton, pos=hit_pos)
    assert hit_cell in fb.canvas._pinned

    QTest.mouseClick(fb.canvas, Qt.MouseButton.LeftButton, pos=hit_pos)
    assert hit_cell not in fb.canvas._pinned

    QTest.mouseClick(fb.canvas, Qt.MouseButton.LeftButton, pos=hit_pos)
    assert hit_cell in fb.canvas._pinned
    QTest.mouseClick(fb.canvas, Qt.MouseButton.RightButton, pos=hit_pos)
    assert fb.canvas._pinned == set()

    fb.close()


def test_main_window_drawer_stage_mode_theme_shortcuts_and_persistence(tmp_path: Path) -> None:
    app = _app()
    state_path = tmp_path / "ui_state.json"

    win = MainWindow(start_workers=False, ui_state_path=state_path)
    win.resize(1360, 880)
    win.show()
    app.processEvents()

    # Verify registered keyboard shortcuts
    expected_keys = {
        QKeySequence(k).toString()
        for k in ("Space", "F", "C", "R", "M", "D", "Escape")
    }
    registered_keys = {sc.key().toString() for sc in win._shortcuts}
    assert expected_keys.issubset(registered_keys)

    # Drawer starts closed and opens/closes cleanly
    assert win.setup_drawer.isHidden()
    win.setup_drawer_btn.setChecked(True)
    app.processEvents()
    assert win.setup_drawer.isVisible()

    # Stage Mode hides camera and drawer, enlarges hero labels
    win.stage_mode_btn.setChecked(True)
    app.processEvents()
    assert win.camera_card.isHidden()
    assert win.setup_drawer.isHidden()
    assert win.key_hero_lbl.objectName() == "StageHeroValue"
    assert win.note_hero_lbl.objectName() == "StageHeroValue"

    # Exiting Stage Mode restores camera card & HeroValue
    win.stage_mode_btn.setChecked(False)
    app.processEvents()
    assert win.camera_card.isVisible()
    assert win.key_hero_lbl.objectName() == "HeroValue"
    assert win.note_hero_lbl.objectName() == "HeroValue"

    # Theme switch applies High-Contrast stylesheet
    hc_idx = win.theme_combo.findData("high_contrast")
    assert hc_idx >= 0
    win.theme_combo.setCurrentIndex(hc_idx)
    app.processEvents()
    assert win.config.theme_name == "high_contrast"
    assert win.styleSheet() == get_stylesheet("high_contrast")

    # Persist custom splitter sizes, degrees mode, drawer open, and theme
    win.setup_drawer_btn.setChecked(True)
    win.fretboard_widget.degrees_btn.click()
    win.h_splitter.setSizes([640, 380])
    win.v_splitter.setSizes([380, 170, 180])
    app.processEvents()
    saved_h = win.h_splitter.sizes()
    saved_v = win.v_splitter.sizes()

    win.save_ui_state()
    assert state_path.exists()
    win.close()

    # Reload into a second MainWindow and confirm restoration
    win2 = MainWindow(start_workers=False, ui_state_path=state_path)
    win2.show()
    app.processEvents()
    assert win2.config.theme_name == "high_contrast"
    assert win2.config.fretboard_label_mode == "degrees"
    assert win2.fretboard_widget.degrees_btn.isChecked()
    assert win2.setup_drawer_btn.isChecked()
    assert win2.setup_drawer.isVisible()
    assert win2.h_splitter.sizes() == saved_h
    assert win2.v_splitter.sizes() == saved_v
    win2.close()
