"""Live guitar tablature timeline recorder and ASCII / JSON exporter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fretvision.theory.music import base_open_midis, pc_to_name

from .av_fusion import FusedFrameResult


@dataclass
class TabColumn:
    """Single vertical column on a 6-string guitar tablature staff."""

    timestamp_s: float
    frets: dict[int, int]  # string_idx (0=Low E .. 5=High E) -> fret number
    chord_label: str | None = None


class TabRecorder:
    """Converts fused live note/chord frames into clean rolling guitar tablature."""

    def __init__(
        self,
        max_columns: int = 240,
        chord_merge_window_s: float = 0.075,
        repeat_debounce_s: float = 0.12,
    ) -> None:
        self.max_columns = int(max_columns)
        self.chord_merge_window_s = float(chord_merge_window_s)
        self.repeat_debounce_s = float(repeat_debounce_s)

        self.columns: list[TabColumn] = []
        self._paused: bool = False
        self._last_active_set: dict[int, int] = {}
        self._last_emit_time: float = -1e9

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def clear(self) -> None:
        self.columns.clear()
        self._last_active_set.clear()
        self._last_emit_time = -1e9

    def ingest(self, result: FusedFrameResult) -> bool:
        """Record a new tab column if a note onset or distinct fretting change occurred.

        Returns True when a column was added or updated.
        """
        if self._paused:
            return False

        if not result.sounding_notes:
            self._last_active_set.clear()
            return False

        current_map = {n.string_idx: n.fret for n in result.sounding_notes}
        changed = current_map != self._last_active_set
        dt_since_last = result.timestamp_s - self._last_emit_time

        if not result.is_onset and not changed:
            return False
        if not changed and dt_since_last < self.repeat_debounce_s:
            return False
        if changed and not result.is_onset and dt_since_last < 0.05:
            return False

        chord_label = result.chord.symbol if result.chord is not None else None

        # Merge rolled/strummed notes within chord_merge_window_s into current column
        if (
            self.columns
            and (result.timestamp_s - self.columns[-1].timestamp_s) <= self.chord_merge_window_s
            and not any(s in self.columns[-1].frets and self.columns[-1].frets[s] != f for s, f in current_map.items())
        ):
            self.columns[-1].frets.update(current_map)
            if chord_label:
                self.columns[-1].chord_label = chord_label
        else:
            self.columns.append(
                TabColumn(
                    timestamp_s=result.timestamp_s,
                    frets=dict(current_map),
                    chord_label=chord_label,
                )
            )
            if len(self.columns) > self.max_columns:
                self.columns = self.columns[-self.max_columns :]

        self._last_active_set = current_map
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
        # Display order: high E (string_idx=5) at top -> low E (string_idx=0) at bottom
        string_labels = [
            f"{pc_to_name(base_midis[s] % 12):<2}" if s != 5 else "e "
            for s in reversed(range(6))
        ]
        # Correct standard lowercase 'e' only for standard high E4
        if base_midis[5] == 64:
            string_labels[0] = "e "

        blocks: list[str] = []
        for start in range(0, len(self.columns), cols_per_line):
            chunk = self.columns[start : start + cols_per_line]
            chord_row = "   "
            staff_rows = [f"{lbl}|-" for lbl in string_labels]

            for col in chunk:
                max_digits = max((len(str(f)) for f in col.frets.values()), default=1)
                cell_w = max_digits + 1
                lbl = (col.chord_label or "")[: cell_w + 1]
                chord_row += f"{lbl:<{cell_w + 1}}"

                for row_idx, s_idx in enumerate(reversed(range(6))):
                    if s_idx in col.frets:
                        val = str(col.frets[s_idx])
                        staff_rows[row_idx] += val.ljust(cell_w, "-") + "-"
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
            "columns": [asdict(c) for c in self.columns],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target
