"""Live scrolling 6-string guitar tablature staff widget with metronome/quantize/export controls."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fretvision.audio.metronome import TIME_SIGNATURES
from fretvision.fusion.tab_recorder import TabColumn
from fretvision.theory.music import base_open_midis, pc_to_name


class _TabCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._columns: list[TabColumn] = []
        self._tuning_name: str = "Standard (EADGBE)"

    def update_tab(self, columns: list[TabColumn], tuning_name: str) -> None:
        self._columns = columns
        self._tuning_name = tuning_name
        self.update()

    def paintEvent(self, event: object) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#11161D"))

        w = float(self.width())
        h = float(self.height())
        left_pad = 42.0
        right_pad = 16.0
        top_y = 30.0
        bot_y = h - 16.0
        staff_h = max(72.0, bot_y - top_y)
        row_step = staff_h / 5.0

        base_midis = base_open_midis(self._tuning_name)
        string_ys: dict[int, float] = {}
        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))

        for row_idx, s_idx in enumerate(reversed(range(6))):
            sy = top_y + row_idx * row_step
            string_ys[s_idx] = sy
            painter.setPen(QPen(QColor("#2A3241"), 1.2))
            painter.drawLine(QPointF(left_pad, sy), QPointF(w - right_pad, sy))

            lbl = "e" if (s_idx == 5 and base_midis[5] == 64) else pc_to_name(base_midis[s_idx] % 12)
            painter.setPen(QColor("#8B949E"))
            painter.drawText(QRectF(6, sy - 9, 28, 18), Qt.AlignmentFlag.AlignCenter, lbl)

        # Nut bar
        painter.setPen(QPen(QColor("#4B5563"), 2.0))
        painter.drawLine(QPointF(left_pad, top_y), QPointF(left_pad, top_y + staff_h))

        if not self._columns:
            painter.setPen(QColor("#6E7681"))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(
                QRectF(left_pad, top_y, w - left_pad - right_pad, staff_h),
                Qt.AlignmentFlag.AlignCenter,
                "Play notes or chords to populate live tablature...",
            )
            return

        col_spacing = 38.0
        max_visible = max(1, int((w - left_pad - right_pad - 20.0) / col_spacing))
        visible = self._columns[-max_visible:]

        # Alternating subtle measure shading for quantized bars
        if any(c.bar_idx is not None for c in visible):
            for idx, col in enumerate(visible):
                if col.bar_idx is not None and (col.bar_idx % 2 == 1):
                    cx = left_pad + 24.0 + idx * col_spacing
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 255, 255, 8))
                    painter.drawRect(
                        QRectF(cx - col_spacing * 0.5, top_y, col_spacing, staff_h)
                    )

        prev_bar = visible[0].bar_idx
        for idx, col in enumerate(visible):
            cx = left_pad + 24.0 + idx * col_spacing

            # Draw measure barlines on bar transitions in quantized mode, else every 4 events
            draw_bar = False
            solid_bar = False
            if col.bar_idx is not None and prev_bar is not None and col.bar_idx != prev_bar:
                draw_bar = True
                solid_bar = True
            elif col.bar_idx is None and idx > 0 and idx % 4 == 0:
                draw_bar = True
            prev_bar = col.bar_idx

            if draw_bar:
                pen_style = Qt.PenStyle.SolidLine if solid_bar else Qt.PenStyle.DashLine
                pen_col = QColor("#4B5563") if solid_bar else QColor("#262D3A")
                painter.setPen(QPen(pen_col, 1.4 if solid_bar else 1.0, pen_style))
                painter.drawLine(
                    QPointF(cx - col_spacing * 0.5, top_y),
                    QPointF(cx - col_spacing * 0.5, top_y + staff_h),
                )

            if col.chord_label:
                painter.setPen(QColor("#F59E0B"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(QRectF(cx - 24, 4, 48, 18), Qt.AlignmentFlag.AlignCenter, col.chord_label)

            is_latest = idx == len(visible) - 1
            if is_latest:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(59, 130, 246, 22))
                painter.drawRoundedRect(
                    QRectF(cx - col_spacing * 0.42, top_y - 4, col_spacing * 0.84, staff_h + 8),
                    6.0,
                    6.0,
                )

            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            for s_idx in col.frets:
                token = col.format_token(s_idx)
                sy = string_ys[s_idx]
                pill_w = max(22.0, 9.0 * len(token) + 8.0)
                pill = QRectF(cx - pill_w * 0.5, sy - 9, pill_w, 18)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#2563EB") if is_latest else QColor("#11161D"))
                painter.drawRoundedRect(pill, 4.0, 4.0)
                painter.setPen(QColor("#FFFFFF") if is_latest else QColor("#E2E8F0"))
                painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, token)


class LiveTabWidget(QWidget):
    """Rolling tablature display with Metronome, Quantize, Pause/Clear, and Export controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("LIVE GUITAR TABLATURE")
        title.setObjectName("CardTitle")
        header.addWidget(title)

        header.addSpacing(8)
        header.addWidget(QLabel("BPM:"))
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(40, 220)
        self.bpm_spin.setValue(100)
        header.addWidget(self.bpm_spin)

        self.timesig_combo = QComboBox()
        self.timesig_combo.addItems(list(TIME_SIGNATURES.keys()))
        header.addWidget(self.timesig_combo)

        self.quantize_chk = QCheckBox("Quantize")
        header.addWidget(self.quantize_chk)

        self.metronome_btn = QPushButton("Metronome")
        self.metronome_btn.setCheckable(True)
        header.addWidget(self.metronome_btn)

        header.addStretch(1)

        self.pause_btn = QPushButton("Pause Tab")
        self.pause_btn.setCheckable(True)
        self.clear_btn = QPushButton("Clear")
        self.export_txt_btn = QPushButton("ASCII (.txt)")
        self.export_midi_btn = QPushButton("MIDI (.mid)")
        self.export_xml_btn = QPushButton("MusicXML")
        self.export_json_btn = QPushButton("JSON")

        header.addWidget(self.pause_btn)
        header.addWidget(self.clear_btn)
        header.addWidget(self.export_txt_btn)
        header.addWidget(self.export_midi_btn)
        header.addWidget(self.export_xml_btn)
        header.addWidget(self.export_json_btn)
        root.addLayout(header)

        self.canvas = _TabCanvas()
        root.addWidget(self.canvas)

    def update_tab(self, columns: list[TabColumn], tuning_name: str) -> None:
        self.canvas.update_tab(columns, tuning_name)
