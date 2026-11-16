"""Live scrolling 6-string guitar tablature staff widget."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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

        col_spacing = 36.0
        max_visible = max(1, int((w - left_pad - right_pad - 20.0) / col_spacing))
        visible = self._columns[-max_visible:]

        for idx, col in enumerate(visible):
            cx = left_pad + 22.0 + idx * col_spacing

            # Bar separator every 4 events for visual readability
            if idx > 0 and idx % 4 == 0:
                painter.setPen(QPen(QColor("#262D3A"), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(cx - col_spacing * 0.5, top_y), QPointF(cx - col_spacing * 0.5, top_y + staff_h))

            if col.chord_label:
                painter.setPen(QColor("#F59E0B"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(QRectF(cx - 22, 4, 44, 18), Qt.AlignmentFlag.AlignCenter, col.chord_label)

            is_latest = idx == len(visible) - 1
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            for s_idx, fret in col.frets.items():
                sy = string_ys[s_idx]
                pill = QRectF(cx - 11, sy - 9, 22, 18)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#1D4ED8") if is_latest else QColor("#11161D"))
                painter.drawRoundedRect(pill, 4.0, 4.0)
                painter.setPen(QColor("#FFFFFF") if is_latest else QColor("#E2E8F0"))
                painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, str(fret))


class LiveTabWidget(QWidget):
    """Rolling tablature display with Pause, Clear, and Export actions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("LIVE GUITAR TABLATURE")
        title.setObjectName("CardTitle")
        header.addWidget(title)
        header.addStretch(1)

        self.pause_btn = QPushButton("Pause Tab")
        self.pause_btn.setCheckable(True)
        self.clear_btn = QPushButton("Clear")
        self.export_txt_btn = QPushButton("Export ASCII (.txt)")
        self.export_json_btn = QPushButton("Export JSON")

        header.addWidget(self.pause_btn)
        header.addWidget(self.clear_btn)
        header.addWidget(self.export_txt_btn)
        header.addWidget(self.export_json_btn)
        root.addLayout(header)

        self.canvas = _TabCanvas()
        root.addWidget(self.canvas)

    def update_tab(self, columns: list[TabColumn], tuning_name: str) -> None:
        self.canvas.update_tab(columns, tuning_name)
