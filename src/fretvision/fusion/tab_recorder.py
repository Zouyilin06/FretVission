"""Live guitar tablature timeline recorder (free-time & quantized) and ASCII/JSON/MIDI/MusicXML export."""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from fretvision.audio.metronome import TIME_SIGNATURES
from fretvision.theory.music import base_open_midis, pc_to_name

from .av_fusion import FusedFrameResult

TabTimingMode = Literal["free_time", "quantized"]


@dataclass
class TabColumn:
    """Single vertical column on a 6-string guitar tablature staff."""

    timestamp_s: float
    frets: dict[int, int]  # string_idx (0=Low E .. 5=High E) -> fret number
    chord_label: str | None = None
    techniques: dict[int, str] = field(default_factory=dict)  # string_idx -> "h"|"p"|"/"|"b"
    bar_idx: int | None = None
    grid_slot: int | None = None

    def format_token(self, string_idx: int) -> str:
        fret = self.frets[string_idx]
        tech = self.techniques.get(string_idx)
        if tech in ("h", "p", "/"):
            return f"{tech}{fret}"
        if tech == "b":
            return f"{fret}b"
        return str(fret)


def _encode_vlq(value: int) -> bytes:
    """Encode a non-negative integer as a MIDI Variable-Length Quantity."""
    v = max(0, int(value))
    out = [v & 0x7F]
    v >>= 7
    while v > 0:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.reverse()
    return bytes(out)


def _midi_to_xml_pitch(midi: int) -> tuple[str, int, int]:
    """Return (step, alter, octave) for MusicXML pitch representation."""
    names = (
        ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
        ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
    )
    step, alter = names[int(midi) % 12]
    octave = (int(midi) // 12) - 1
    return step, alter, octave


class TabRecorder:
    """Converts fused live note/chord frames into free-time or beat-quantized tablature."""

    def __init__(
        self,
        max_columns: int = 240,
        chord_merge_window_s: float = 0.075,
        repeat_debounce_s: float = 0.12,
        mode: TabTimingMode = "free_time",
        bpm: int = 100,
        time_signature: str = "4/4",
        subdivision: int = 8,
    ) -> None:
        self.max_columns = int(max_columns)
        self.chord_merge_window_s = float(chord_merge_window_s)
        self.repeat_debounce_s = float(repeat_debounce_s)
        self.mode: TabTimingMode = mode
        self.bpm = max(40, min(240, int(bpm)))
        self.time_signature = time_signature if time_signature in TIME_SIGNATURES else "4/4"
        self.subdivision = 16 if int(subdivision) == 16 else 8

        self.columns: list[TabColumn] = []
        self._paused: bool = False
        self._last_active_set: dict[int, int] = {}
        self._last_tech_set: dict[int, str] = {}
        self._last_emit_time: float = -1e9
        self._quantize_origin_s: float | None = None

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def set_timing(
        self,
        mode: TabTimingMode,
        bpm: int = 100,
        time_signature: str = "4/4",
        subdivision: int = 8,
    ) -> None:
        self.mode = mode
        self.bpm = max(40, min(240, int(bpm)))
        if time_signature in TIME_SIGNATURES:
            self.time_signature = time_signature
        self.subdivision = 16 if int(subdivision) == 16 else 8

    def reset_quantize_clock(self) -> None:
        self._quantize_origin_s = None

    def clear(self) -> None:
        self.columns.clear()
        self._last_active_set.clear()
        self._last_tech_set.clear()
        self._last_emit_time = -1e9
        self._quantize_origin_s = None

    def _quantize_position(self, timestamp_s: float) -> tuple[int, int, float]:
        """Snap timestamp_s onto (bar_idx, grid_slot_in_bar, snapped_time_s)."""
        if self._quantize_origin_s is None:
            self._quantize_origin_s = timestamp_s
        beats_per_bar, beat_unit = TIME_SIGNATURES.get(self.time_signature, (4, 4))
        slots_per_quarter = self.subdivision / 4.0
        quarter_dur_s = 60.0 / float(self.bpm)
        slot_dur_s = quarter_dur_s / slots_per_quarter
        slots_per_bar = max(1, int(round(beats_per_bar * (4.0 / beat_unit) * slots_per_quarter)))

        rel_s = max(0.0, timestamp_s - self._quantize_origin_s)
        total_slot = int(round(rel_s / slot_dur_s))
        bar_idx = total_slot // slots_per_bar
        grid_slot = total_slot % slots_per_bar
        snapped_s = self._quantize_origin_s + total_slot * slot_dur_s
        return bar_idx, grid_slot, snapped_s

    def ingest(self, result: FusedFrameResult) -> bool:
        """Record a new tab column if a note onset or distinct fretting change occurred."""
        if self._paused:
            return False

        if not result.sounding_notes:
            self._last_active_set.clear()
            self._last_tech_set.clear()
            return False

        current_map = {n.string_idx: n.fret for n in result.sounding_notes}
        tech_map = {n.string_idx: n.technique for n in result.sounding_notes if n.technique}
        changed = (current_map != self._last_active_set) or (
            any(t == "b" for t in tech_map.values()) and tech_map != self._last_tech_set
        )
        dt_since_last = result.timestamp_s - self._last_emit_time
        if dt_since_last < 0.0:
            # Clock rewound (e.g. clip seek/loop)
            self._last_emit_time = -1e9
            dt_since_last = 1e9

        if not result.is_onset and not changed:
            return False
        if not changed and dt_since_last < self.repeat_debounce_s:
            return False
        if changed and not result.is_onset and dt_since_last < 0.05:
            return False

        chord_label = result.chord.symbol if result.chord is not None else None

        if self.mode == "quantized":
            bar_idx, grid_slot, snapped_s = self._quantize_position(result.timestamp_s)
            if (
                self.columns
                and self.columns[-1].bar_idx == bar_idx
                and self.columns[-1].grid_slot == grid_slot
            ):
                self.columns[-1].frets.update(current_map)
                self.columns[-1].techniques.update(tech_map)
                if chord_label:
                    self.columns[-1].chord_label = chord_label
            else:
                self.columns.append(
                    TabColumn(
                        timestamp_s=snapped_s,
                        frets=dict(current_map),
                        chord_label=chord_label,
                        techniques=dict(tech_map),
                        bar_idx=bar_idx,
                        grid_slot=grid_slot,
                    )
                )
        else:
            # Free-time strum merging
            if (
                self.columns
                and (result.timestamp_s - self.columns[-1].timestamp_s) <= self.chord_merge_window_s
                and not any(
                    s in self.columns[-1].frets and self.columns[-1].frets[s] != f
                    for s, f in current_map.items()
                )
            ):
                self.columns[-1].frets.update(current_map)
                self.columns[-1].techniques.update(tech_map)
                if chord_label:
                    self.columns[-1].chord_label = chord_label
            else:
                self.columns.append(
                    TabColumn(
                        timestamp_s=result.timestamp_s,
                        frets=dict(current_map),
                        chord_label=chord_label,
                        techniques=dict(tech_map),
                    )
                )

        if len(self.columns) > self.max_columns:
            self.columns = self.columns[-self.max_columns :]

        self._last_active_set = current_map
        self._last_tech_set = tech_map
        self._last_emit_time = result.timestamp_s
        return True

    def format_ascii(
        self,
        tuning_name: str = "Standard (EADGBE)",
        cols_per_line: int = 18,
    ) -> str:
        """Render recorded columns as standard 6-line ASCII guitar tablature."""
        if not self.columns:
            return "(No tablature notes recorded yet)"

        base_midis = base_open_midis(tuning_name)
        string_labels = [
            f"{pc_to_name(base_midis[s] % 12):<2}" if s != 5 else "e "
            for s in reversed(range(6))
        ]
        if base_midis[5] == 64:
            string_labels[0] = "e "

        blocks: list[str] = []
        for start in range(0, len(self.columns), cols_per_line):
            chunk = self.columns[start : start + cols_per_line]
            chord_row = "   "
            staff_rows = [f"{lbl}|-" for lbl in string_labels]
            prev_bar: int | None = chunk[0].bar_idx

            for col in chunk:
                if col.bar_idx is not None and prev_bar is not None and col.bar_idx != prev_bar:
                    chord_row += " "
                    staff_rows = [r + "|" for r in staff_rows]
                prev_bar = col.bar_idx

                tokens = {s: col.format_token(s) for s in col.frets}
                max_len = max((len(t) for t in tokens.values()), default=1)
                cell_w = max_len + 1
                lbl = (col.chord_label or "")[: cell_w + 1]
                chord_row += f"{lbl:<{cell_w + 1}}"

                for row_idx, s_idx in enumerate(reversed(range(6))):
                    if s_idx in tokens:
                        staff_rows[row_idx] += tokens[s_idx].ljust(cell_w, "-") + "-"
                    else:
                        staff_rows[row_idx] += "-" * (cell_w + 1)

            staff_rows = [r + "|" for r in staff_rows]
            if any(c.chord_label for c in chunk):
                blocks.append(chord_row.rstrip() + "\n" + "\n".join(staff_rows))
            else:
                blocks.append("\n".join(staff_rows))

        return "\n\n".join(blocks)

    def export_ascii(self, path: str | Path, tuning_name: str = "Standard (EADGBE)") -> Path:
        """Write ASCII tablature to a .txt file."""
        target = Path(path)
        target.write_text(self.format_ascii(tuning_name=tuning_name), encoding="utf-8")
        return target

    def export_json(self, path: str | Path, tuning_name: str = "Standard (EADGBE)") -> Path:
        """Write structured tablature timeline to a .json file."""
        target = Path(path)
        payload = {
            "tuning": tuning_name,
            "mode": self.mode,
            "bpm": self.bpm,
            "time_signature": self.time_signature,
            "columns": [asdict(c) for c in self.columns],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def export_midi(
        self,
        path: str | Path,
        tuning_name: str = "Standard (EADGBE)",
        bpm: int | None = None,
    ) -> Path:
        """Export recorded tablature as a Standard MIDI File (.mid, Format 0)."""
        target = Path(path)
        use_bpm = max(40, min(240, int(bpm or self.bpm)))
        ppq = 480
        base_midis = base_open_midis(tuning_name)

        events: list[tuple[int, bytes]] = []
        mpqn = int(round(60_000_000 / use_bpm))
        events.append((0, b"\xFF\x51\x03" + mpqn.to_bytes(3, "big")))
        events.append((0, bytes([0xC0, 25])))  # Acoustic Guitar (steel)

        if self.columns:
            t0 = self.columns[0].timestamp_s
            ticks_per_sec = (use_bpm / 60.0) * ppq
            for idx, col in enumerate(self.columns):
                start_tick = max(0, int(round((col.timestamp_s - t0) * ticks_per_sec)))
                if idx + 1 < len(self.columns):
                    next_tick = max(
                        start_tick + 60,
                        int(round((self.columns[idx + 1].timestamp_s - t0) * ticks_per_sec)),
                    )
                    dur_ticks = min(ppq * 2, next_tick - start_tick)
                else:
                    dur_ticks = ppq

                for s_idx, fret in sorted(col.frets.items()):
                    midi_note = max(0, min(127, int(base_midis[s_idx] + fret)))
                    events.append((start_tick, bytes([0x90, midi_note, 96])))
                    events.append((start_tick + dur_ticks, bytes([0x80, midi_note, 0])))

        # Sort by tick (Note-Off before Note-On at identical ticks)
        events.sort(key=lambda item: (item[0], 0 if item[1][0] == 0x80 else 1))

        track_bytes = bytearray()
        prev_tick = 0
        for tick, msg in events:
            delta = max(0, tick - prev_tick)
            track_bytes.extend(_encode_vlq(delta))
            track_bytes.extend(msg)
            prev_tick = tick
        track_bytes.extend(_encode_vlq(0) + b"\xFF\x2F\x00")

        header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq)
        track = b"MTrk" + struct.pack(">I", len(track_bytes)) + bytes(track_bytes)
        target.write_bytes(header + track)
        return target

    def export_musicxml(
        self,
        path: str | Path,
        tuning_name: str = "Standard (EADGBE)",
        bpm: int | None = None,
        time_signature: str | None = None,
    ) -> Path:
        """Export recorded tablature as MusicXML 4.0 (.musicxml) with string/fret notation."""
        target = Path(path)
        use_bpm = max(40, min(240, int(bpm or self.bpm)))
        use_ts = time_signature if time_signature in TIME_SIGNATURES else self.time_signature
        beats, beat_type = TIME_SIGNATURES[use_ts]
        base_midis = base_open_midis(tuning_name)
        divisions = 4  # 16th-note resolution per quarter note

        score = ET.Element("score-partwise", version="4.0")
        part_list = ET.SubElement(score, "part-list")
        sp = ET.SubElement(part_list, "score-part", id="P1")
        ET.SubElement(sp, "part-name").text = "Guitar"
        part = ET.SubElement(score, "part", id="P1")

        # Group columns by bar_idx (or 4 columns per measure in free_time)
        measures: list[list[TabColumn]] = []
        if self.columns and all(c.bar_idx is not None for c in self.columns):
            cur_bar: int | None = None
            for col in self.columns:
                if col.bar_idx != cur_bar:
                    measures.append([])
                    cur_bar = col.bar_idx
                measures[-1].append(col)
        else:
            for i in range(0, max(1, len(self.columns)), 4):
                measures.append(self.columns[i : i + 4])

        for m_num, cols in enumerate(measures, start=1):
            measure = ET.SubElement(part, "measure", number=str(m_num))
            if m_num == 1:
                attrs = ET.SubElement(measure, "attributes")
                ET.SubElement(attrs, "divisions").text = str(divisions)
                time_el = ET.SubElement(attrs, "time")
                ET.SubElement(time_el, "beats").text = str(beats)
                ET.SubElement(time_el, "beat-type").text = str(beat_type)
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = "TAB"
                ET.SubElement(clef, "line").text = "5"
                sd_el = ET.SubElement(attrs, "staff-details")
                ET.SubElement(sd_el, "staff-lines").text = "6"
                for line_idx, open_midi in enumerate(base_midis, start=1):
                    step, alter, octave = _midi_to_xml_pitch(open_midi)
                    st = ET.SubElement(sd_el, "staff-tuning", line=str(line_idx))
                    ET.SubElement(st, "tuning-step").text = step
                    if alter:
                        ET.SubElement(st, "tuning-alter").text = str(alter)
                    ET.SubElement(st, "tuning-octave").text = str(octave)
                direction = ET.SubElement(measure, "direction", placement="above")
                ET.SubElement(direction, "sound", tempo=str(use_bpm))

            if not cols:
                note_el = ET.SubElement(measure, "note")
                ET.SubElement(note_el, "rest")
                ET.SubElement(note_el, "duration").text = str(divisions * beats)
                ET.SubElement(note_el, "type").text = "whole"
                continue

            for col in cols:
                if col.chord_label:
                    harm = ET.SubElement(measure, "harmony")
                    root_el = ET.SubElement(harm, "root")
                    ET.SubElement(root_el, "root-step").text = col.chord_label[0]
                    ET.SubElement(harm, "kind", text=col.chord_label).text = "other"

                for n_idx, (s_idx, fret) in enumerate(sorted(col.frets.items())):
                    note_el = ET.SubElement(measure, "note")
                    if n_idx > 0:
                        ET.SubElement(note_el, "chord")
                    midi_val = int(base_midis[s_idx] + fret)
                    step, alter, octave = _midi_to_xml_pitch(midi_val)
                    pitch_el = ET.SubElement(note_el, "pitch")
                    ET.SubElement(pitch_el, "step").text = step
                    if alter:
                        ET.SubElement(pitch_el, "alter").text = str(alter)
                    ET.SubElement(pitch_el, "octave").text = str(octave)
                    ET.SubElement(note_el, "duration").text = "4"
                    ET.SubElement(note_el, "type").text = "quarter"

                    notations = ET.SubElement(note_el, "notations")
                    tech = ET.SubElement(notations, "technical")
                    ET.SubElement(tech, "string").text = str(6 - s_idx)
                    ET.SubElement(tech, "fret").text = str(fret)
                    t_tag = col.techniques.get(s_idx)
                    if t_tag == "h":
                        ET.SubElement(tech, "hammer-on", type="start").text = "H"
                    elif t_tag == "p":
                        ET.SubElement(tech, "pull-off", type="start").text = "P"
                    elif t_tag == "b":
                        bend = ET.SubElement(tech, "bend")
                        ET.SubElement(bend, "bend-alter").text = "1"
                    elif t_tag == "/":
                        ET.SubElement(notations, "slide", type="start")

        tree = ET.ElementTree(score)
        ET.indent(tree, space="  ")
        tree.write(target, encoding="utf-8", xml_declaration=True)
        return target
