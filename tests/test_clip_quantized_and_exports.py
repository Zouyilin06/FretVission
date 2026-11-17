"""Tests for Phase 2C: Metronome, quantized tab, technique notation, clip replay, MIDI & MusicXML."""

from __future__ import annotations

import struct
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from fretvision.audio.metronome import MetronomeEngine
from fretvision.audio.pitch_detector import HybridHarmonicPitchDetector
from fretvision.fusion.av_fusion import ActiveNote, FusedFrameResult
from fretvision.fusion.clip_replay import ClipReplaySource
from fretvision.fusion.tab_recorder import TabColumn, TabRecorder
from fretvision.theory.music import ChordMatch, midi_to_freq


def _fused(
    ts: float,
    notes: tuple[ActiveNote, ...],
    chord_sym: str | None = None,
    is_onset: bool = True,
) -> FusedFrameResult:
    chord = ChordMatch(chord_sym, 0, "maj", 0, 0.9) if chord_sym else None
    return FusedFrameResult(
        timestamp_s=ts,
        sounding_notes=notes,
        hover_notes=(),
        chord=chord,
        key_estimate=None,
        rms_db=-20.0,
        is_onset=is_onset,
        dominant_cents=0.0,
    )


def test_metronome_click_timing_synthesis() -> None:
    met = MetronomeEngine(bpm=120, time_signature="4/4", sample_rate=44100)
    assert met.beats_per_bar() == 4
    # At 120 BPM, 1 beat = 22050 samples. Buffer 0..512 triggers accented downbeat
    buf0 = np.zeros((512, 1), dtype=np.float32)
    met._callback(buf0, 512, None, None)
    assert float(np.max(np.abs(buf0))) > 0.15

    # Buffer after the 26 ms (~1147 sample) downbeat click ends should be silent
    buf_mid = np.zeros((20000, 1), dtype=np.float32)
    met._callback(buf_mid, 20000, None, None)
    assert float(np.max(np.abs(buf_mid[1000:]))) < 1e-4

    # Advance across beat 1 (sample 22050)
    buf_b1 = np.zeros((3000, 1), dtype=np.float32)
    met._callback(buf_b1, 3000, None, None)
    assert float(np.max(np.abs(buf_b1))) > 0.10


def test_quantized_tab_bars_and_techniques() -> None:
    # 120 BPM, 4/4, 8th-note grid => 0.25s per slot, 8 slots per bar (2.0s per bar)
    rec = TabRecorder(mode="quantized", bpm=120, time_signature="4/4", subdivision=8)

    n_f5 = ActiveNote(3, 5, 60, "C4", 0.95, "av_fused", "1")
    n_h7 = ActiveNote(3, 7, 62, "D4", 0.95, "av_fused", "3", technique="h")
    n_p5 = ActiveNote(3, 5, 60, "C4", 0.95, "av_fused", "1", technique="p")
    n_s9 = ActiveNote(3, 9, 64, "E4", 0.95, "av_fused", "3", technique="/")
    n_b9 = ActiveNote(3, 9, 66, "F#4", 0.95, "av_fused", "3", bend_semitones=2, technique="b")

    rec.ingest(_fused(0.01, (n_f5,), is_onset=True))   # bar 0, slot 0
    rec.ingest(_fused(0.51, (n_h7,), is_onset=False))  # bar 0, slot 2
    rec.ingest(_fused(1.01, (n_p5,), is_onset=False))  # bar 0, slot 4
    rec.ingest(_fused(1.52, (n_s9,), is_onset=False))  # bar 0, slot 6
    rec.ingest(_fused(2.02, (n_b9,), is_onset=True))   # bar 1, slot 0

    assert len(rec.columns) == 5
    assert rec.columns[0].bar_idx == 0 and rec.columns[0].grid_slot == 0
    assert rec.columns[1].bar_idx == 0 and rec.columns[1].grid_slot == 2
    assert rec.columns[4].bar_idx == 1 and rec.columns[4].grid_slot == 0

    ascii_tab = rec.format_ascii()
    assert "h7" in ascii_tab
    assert "p5" in ascii_tab
    assert "/9" in ascii_tab
    assert "9b" in ascii_tab
    # Solid measure barline inserted between bar 0 and bar 1
    assert "-|-" in ascii_tab or "|9b" in ascii_tab


def test_midi_and_musicxml_exports(tmp_path: Path) -> None:
    rec = TabRecorder(mode="quantized", bpm=110, time_signature="4/4", subdivision=8)
    g_chord = (
        ActiveNote(0, 3, 43, "G2", 0.95, "av_fused"),
        ActiveNote(4, 0, 59, "B3", 0.95, "av_fused"),
        ActiveNote(5, 3, 67, "G4", 0.95, "av_fused"),
    )
    rec.ingest(_fused(0.0, g_chord, chord_sym="G", is_onset=True))
    rec.ingest(_fused(0.6, (ActiveNote(3, 7, 62, "D4", 0.95, "av_fused", technique="h"),), is_onset=False))
    rec.ingest(_fused(1.2, (ActiveNote(3, 7, 64, "E4", 0.95, "av_fused", bend_semitones=2, technique="b"),), is_onset=False))

    mid_path = rec.export_midi(tmp_path / "song.mid")
    raw = mid_path.read_bytes()
    assert raw[:4] == b"MThd"
    hdr_len, fmt, ntracks, ppq = struct.unpack(">IHHH", raw[4:14])
    assert (hdr_len, fmt, ntracks, ppq) == (6, 0, 1, 480)
    assert b"MTrk" in raw
    assert b"\xFF\x51\x03" in raw  # Set Tempo
    assert b"\x90" in raw          # Note On
    assert b"\x80" in raw          # Note Off
    assert raw.endswith(b"\xFF\x2F\x00")

    xml_path = rec.export_musicxml(tmp_path / "song.musicxml")
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    assert root.tag == "score-partwise"
    assert root.attrib.get("version") == "4.0"
    assert len(root.findall(".//staff-tuning")) == 6
    frets = [el.text for el in root.findall(".//technical/fret")]
    assert "3" in frets and "7" in frets
    assert root.find(".//harmony") is not None
    assert root.find(".//technical/hammer-on") is not None
    assert root.find(".//technical/bend") is not None


def test_clip_replay_source_video_audio_sync(tmp_path: Path) -> None:
    vid_path = tmp_path / "clip.avi"
    wav_path = tmp_path / "clip.wav"

    fps = 20.0
    sr = 44100
    duration_s = 1.2
    n_frames = int(fps * duration_s)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (160, 120))
    assert writer.isOpened()
    for i in range(n_frames):
        # First half dark blue, second half brighter green so frame transition is unambiguous
        col = (180, 40, 20) if i < n_frames // 2 else (40, 200, 40)
        frame = np.full((120, 160, 3), col, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Synthesize paired WAV: 0.0..0.6s = E2 (40), 0.6..1.2s = A2 (45)
    n_samp = int(sr * duration_s)
    t = np.arange(n_samp, dtype=np.float64) / sr
    half = n_samp // 2
    sig = np.zeros(n_samp, dtype=np.float64)
    for h in range(1, 5):
        sig[:half] += (1.0 / (h ** 0.72)) * np.sin(2.0 * np.pi * midi_to_freq(40.0) * h * t[:half])
        sig[half:] += (1.0 / (h ** 0.72)) * np.sin(2.0 * np.pi * midi_to_freq(45.0) * h * t[half:])
    sig_i16 = (np.clip(sig / np.max(np.abs(sig)) * 0.6, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sig_i16.tobytes())

    clip = ClipReplaySource(vid_path)
    assert clip.open()
    assert clip.duration_s >= 1.15

    det = HybridHarmonicPitchDetector()
    clip.seek(0.30)
    f0, a0, sr0, _ = clip.read_at_current_position(4096)
    assert f0 is not None
    assert float(np.mean(f0[:, :, 0])) > float(np.mean(f0[:, :, 1]))  # Blue channel dominant
    res0 = det.analyze(a0, sr0)
    assert [p.midi for p in res0.pitches] == [40]

    clip.seek(0.95)
    f1, a1, sr1, _ = clip.read_at_current_position(4096)
    assert f1 is not None
    assert float(np.mean(f1[:, :, 1])) > float(np.mean(f1[:, :, 0]))  # Green channel dominant
    res1 = det.analyze(a1, sr1)
    assert [p.midi for p in res1.pitches] == [45]

    clip.advance(0.5, loop=True)
    assert 0.0 <= clip.position_s < clip.duration_s
    clip.close()
