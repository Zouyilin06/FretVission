"""Tests for audio-visual string/fret disambiguation, barre chords, bends, and tab export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fretvision.audio.pitch_detector import AudioAnalysisResult, DetectedPitch
from fretvision.fusion.av_fusion import AudioVisualFusion
from fretvision.fusion.tab_recorder import TabRecorder
from fretvision.vision.hand_tracker import FingerObservation, HandTrackingResult


def _make_hand_result(
    finger_specs: list[tuple[str, int, int]],
    barre_fret: int | None = None,
) -> HandTrackingResult:
    mat = np.zeros((6, 25), dtype=np.float32)
    fingers: list[FingerObservation] = []
    for lbl, s_idx, f_idx in finger_specs:
        mat[s_idx, f_idx] = 0.95
        fingers.append(
            FingerObservation(
                finger_label=lbl,
                pixel_xy=(100.0, 100.0),
                string_float=float(s_idx),
                fret_float=float(f_idx),
                string_idx=s_idx,
                fret_idx=f_idx,
                in_bounds=True,
                confidence=0.95,
            )
        )
    if barre_fret is not None:
        for s_idx in range(6):
            mat[s_idx, barre_fret] = max(float(mat[s_idx, barre_fret]), 0.80)
    return HandTrackingResult(
        hand_present=True,
        landmarks_px=[],
        hand_bboxes=[],
        fingers=fingers,
        press_prob_matrix=mat,
        model_ready=True,
    )


def _audio_for_midis(midis: list[int], is_onset: bool = True) -> AudioAnalysisResult:
    pitches = tuple(
        DetectedPitch(
            midi=m,
            freq_hz=440.0,
            cents_offset=0.0,
            confidence=0.95,
            is_monophonic=len(midis) == 1,
        )
        for m in midis
    )
    return AudioAnalysisResult(
        pitches=pitches,
        rms_db=-18.0,
        is_onset=is_onset,
        is_silent=False,
        dominant_cents=0.0,
    )


def test_disambiguates_same_pitch_across_strings() -> None:
    fusion = AudioVisualFusion()
    # E4 (MIDI 64) at B-string fret 5 (string_idx=4, fret=5)
    hand_b5 = _make_hand_result([("1", 4, 5)])
    res1 = fusion.fuse(0.1, _audio_for_midis([64]), hand_b5, fretboard_calibrated=True)
    assert len(res1.sounding_notes) == 1
    assert res1.sounding_notes[0].string_idx == 4
    assert res1.sounding_notes[0].fret == 5
    assert res1.sounding_notes[0].source == "av_fused"

    # Same E4 (MIDI 64) played at G-string fret 9 (string_idx=3, fret=9)
    fusion2 = AudioVisualFusion()
    hand_g9 = _make_hand_result([("3", 3, 9)])
    res2 = fusion2.fuse(0.2, _audio_for_midis([64]), hand_g9, fretboard_calibrated=True)
    assert len(res2.sounding_notes) == 1
    assert res2.sounding_notes[0].string_idx == 3
    assert res2.sounding_notes[0].fret == 9

    # Same E4 played open on high E string (string_idx=5, fret=0) with fingers away
    fusion3 = AudioVisualFusion()
    hand_open = _make_hand_result([("1", 1, 2)])  # pressing A-string fret 2, not strings 3/4/5
    res3 = fusion3.fuse(0.3, _audio_for_midis([64]), hand_open, fretboard_calibrated=True)
    assert len(res3.sounding_notes) == 1
    assert res3.sounding_notes[0].string_idx == 5
    assert res3.sounding_notes[0].fret == 0


def test_f_major_barre_chord_and_string_uniqueness() -> None:
    fusion = AudioVisualFusion()
    # F major barre at fret 1: F2(41), C3(48), F3(53), A3(57), C4(60), F4(65)
    hand = _make_hand_result([("3", 1, 3), ("4", 2, 3), ("2", 3, 2)], barre_fret=1)
    audio = _audio_for_midis([41, 48, 53, 57, 60, 65])
    res = fusion.fuse(0.5, audio, hand, fretboard_calibrated=True)

    assert len(res.sounding_notes) == 6
    frets_by_str = {n.string_idx: n.fret for n in res.sounding_notes}
    assert frets_by_str == {0: 1, 1: 3, 2: 3, 3: 2, 4: 1, 5: 1}
    assert res.chord is not None
    assert res.chord.symbol == "F"


def test_string_bend_keeps_anchor_fret() -> None:
    fusion = AudioVisualFusion()
    hand = _make_hand_result([("3", 3, 7)])  # D4 (62) on G string fret 7
    res0 = fusion.fuse(0.1, _audio_for_midis([62]), hand, fretboard_calibrated=True)
    assert res0.sounding_notes[0].fret == 7

    # Bend up whole step to E4 (64) while finger stays on G string fret 7
    res1 = fusion.fuse(0.25, _audio_for_midis([64], is_onset=False), hand, fretboard_calibrated=True)
    assert res1.sounding_notes[0].string_idx == 3
    assert res1.sounding_notes[0].fret == 7
    assert res1.sounding_notes[0].bend_semitones == 2
    assert res1.sounding_notes[0].note_name == "E4"


def test_tab_recorder_and_ascii_export(tmp_path: Path) -> None:
    fusion = AudioVisualFusion()
    rec = TabRecorder()

    # Strum Em chord slightly rolled across 40 ms -> should merge cleanly
    hand_em = _make_hand_result([("2", 1, 2), ("3", 2, 2)])
    r1 = fusion.fuse(0.10, _audio_for_midis([40, 47, 52], is_onset=True), hand_em, True)
    r2 = fusion.fuse(0.14, _audio_for_midis([55, 59, 64], is_onset=True), hand_em, True)
    rec.ingest(r1)
    rec.ingest(r2)
    assert len(rec.columns) == 1
    assert rec.columns[0].frets == {0: 0, 1: 2, 2: 2, 3: 0, 4: 0, 5: 0}

    # Play fret 5 on B string 0.4s later
    hand_b5 = _make_hand_result([("1", 4, 5)])
    r3 = fusion.fuse(0.55, _audio_for_midis([64], is_onset=True), hand_b5, True)
    rec.ingest(r3)
    assert len(rec.columns) == 2

    txt_path = rec.export_ascii(tmp_path / "tab.txt")
    txt = txt_path.read_text(encoding="utf-8")
    assert "e |" in txt
    assert "E |" in txt
    assert "5" in txt

    json_path = rec.export_json(tmp_path / "tab.json")
    assert '"columns"' in json_path.read_text(encoding="utf-8")
