"""Interactive 2D virtual guitar fretboard widget showing scale, hover, and sounding notes."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from fretvision.config import Handedness
from fretvision.fusion.av_fusion import ActiveNote
from fretvision.theory.key_detector import KeyEstimate
from fretvision.theory.music import base_open_midis, midi_to_name, pc_to_name


class VirtualFretboardWidget(QWidget):
    """Draws a 6-string x 22-fret guitar neck with live notes and detected key scale overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(195)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._tuning_name: str = "Standard (EADGBE)"
        self._capo: int = 0
        self._max_fret: int = 22
        self._handedness: Handedness = "right"
        self._show_scale_overlay: bool = True

        self._sounding: tuple[ActiveNote, ...] = ()
        self._hover: tuple[ActiveNote, ...] = ()
        self._key_estimate: KeyEstimate | None = None

    def set_guitar_setup(
        self,
        tuning_name: str,
        capo: int,
        handedness: Handedness = "right",
        max_fret: int = 22,
    ) -> None:
        self._tuning_name = tuning_name
        self._capo = max(0, min(12, int(capo)))
        self._handedness = handedness
        self._max_fret = int(max_fret)
        self.update()

    def set_show_scale_overlay(self, enabled: bool) -> None:
        self._show_scale_overlay = bool(enabled)
        self.update()

    def update_state(
        self,
        sounding_notes: tuple[ActiveNote, ...],
        hover_notes: tuple[ActiveNote, ...],
        key_estimate: KeyEstimate | None,
    ) -> None:
        self._sounding = sounding_notes
        self._hover = hover_notes
        self._key_estimate = key_estimate
        self.update()

    def _fret_x(self, fret: int, left_x: float, neck_w: float) -> float:
        """Compute proportional fret wire x coordinate using gentle compressed 12-TET spacing."""
        if fret <= 0:
            norm = 0.0
        else:
            # Blend pure 12-TET and uniform spacing so upper frets (15-22) remain readable
            u_tet = (1.0 - 2.0 ** (-fret / 12.0)) / (1.0 - 2.0 ** (-self._max_fret / 12.0))
            u_lin = fret / float(self._max_fret)
            norm = 0.65 * u_tet + 0.35 * u_lin
        if self._handedness == "right":
            return left_x + norm * neck_w
        return left_x + (1.0 - norm) * neck_w

    def _fret_center_x(self, fret: int, left_x: float, neck_w: float, nut_x: float) -> float:
        if fret == 0:
            return nut_x + (-18.0 if self._handedness == "right" else 18.0)
        x0 = self._fret_x(fret - 1, left_x, neck_w)
        x1 = self._fret_x(fret, left_x, neck_w)
        return 0.5 * (x0 + x1)

    def paintEvent(self, event: object) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#161B22"))

        w = float(self.width())
        h = float(self.height())
        pad_x = 64.0
        top_y = 32.0
        bot_y = h - 26.0
        neck_h = max(90.0, bot_y - top_y)
        left_x = pad_x + (10.0 if self._handedness == "right" else 0.0)
        neck_w = max(200.0, w - 2.0 * pad_x - 10.0)

        # Header & Legend
        painter.setPen(QColor("#8B949E"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(QPointF(14.0, 20.0), "VIRTUAL FRETBOARD")
        self._draw_legend(painter, w - 14.0, 20.0)

        # Rosewood fretboard body
        painter.setPen(QPen(QColor("#30394A"), 1.2))
        painter.setBrush(QColor("#1A1614"))
        painter.drawRoundedRect(QRectF(left_x, top_y, neck_w, neck_h), 4.0, 4.0)

        # Fretboard dot inlays (3, 5, 7, 9, 12, 15, 17, 19, 21)
        nut_x = self._fret_x(0, left_x, neck_w)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2E3440"))
        for f in (3, 5, 7, 9, 12, 15, 17, 19, 21):
            if f > self._max_fret:
                continue
            cx = self._fret_center_x(f, left_x, neck_w, nut_x)
            if f == 12:
                painter.drawEllipse(QPointF(cx, top_y + neck_h * 0.30), 4.5, 4.5)
                painter.drawEllipse(QPointF(cx, top_y + neck_h * 0.70), 4.5, 4.5)
            else:
                painter.drawEllipse(QPointF(cx, top_y + neck_h * 0.50), 4.5, 4.5)

        # Fret wires & numbers
        painter.setFont(QFont("Segoe UI", 8))
        for f in range(0, self._max_fret + 1):
            fx = self._fret_x(f, left_x, neck_w)
            if f == 0:
                painter.setPen(QPen(QColor("#E2E8F0"), 4.0))
            elif f == self._capo and self._capo > 0:
                painter.setPen(QPen(QColor("#F59E0B"), 4.5))
            else:
                painter.setPen(QPen(QColor("#64748B"), 1.4))
            painter.drawLine(QPointF(fx, top_y), QPointF(fx, top_y + neck_h))

            if f in (0, 3, 5, 7, 9, 12, 15, 17, 19, 21):
                lx = fx if f == 0 else self._fret_center_x(f, left_x, neck_w, nut_x)
                painter.setPen(QColor("#8B949E"))
                painter.drawText(
                    QRectF(lx - 12, top_y + neck_h + 4, 24, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    str(f),
                )

        # String lines (display High E string_idx=5 at top -> Low E string_idx=0 at bottom)
        base_midis = base_open_midis(self._tuning_name)
        string_ys: dict[int, float] = {}
        for row_idx, s_idx in enumerate(reversed(range(6))):
            sy = top_y + (row_idx + 0.5) * (neck_h / 6.0)
            string_ys[s_idx] = sy
            thickness = 1.0 + (5 - s_idx) * 0.28
            painter.setPen(QPen(QColor("#94A3B8"), thickness))
            painter.drawLine(QPointF(left_x, sy), QPointF(left_x + neck_w, sy))

            # Open string tuning label cleanly separated from fret-0 scale/note badges
            open_pc_name = "e" if (s_idx == 5 and base_midis[5] == 64 and self._capo == 0) else pc_to_name(
                (base_midis[s_idx] + self._capo) % 12
            )
            lbl_x = nut_x - 56.0 if self._handedness == "right" else nut_x + 32.0
            painter.setPen(QColor("#8B949E"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(
                QRectF(lbl_x, sy - 9, 22, 18),
                Qt.AlignmentFlag.AlignCenter,
                open_pc_name,
            )

        # Layer 1: Scale notes for detected key
        if self._show_scale_overlay and self._key_estimate is not None and self._key_estimate.confidence >= 0.25:
            scale_pcs = self._key_estimate.scale_pcs
            tonic_pc = self._key_estimate.tonic_pc
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            for s_idx in range(6):
                sy = string_ys[s_idx]
                for f in range(self._capo, self._max_fret + 1):
                    midi = base_midis[s_idx] + f
                    pc = midi % 12
                    if pc not in scale_pcs:
                        continue
                    cx = self._fret_center_x(f, left_x, neck_w, nut_x)
                    is_root = pc == tonic_pc
                    if is_root:
                        painter.setPen(QPen(QColor("#F59E0B"), 1.6))
                        painter.setBrush(QColor(245, 158, 11, 55))
                    else:
                        painter.setPen(QPen(QColor(16, 185, 129, 130), 1.0))
                        painter.setBrush(QColor(16, 185, 129, 26))
                    painter.drawEllipse(QPointF(cx, sy), 8.0, 8.0)
                    painter.setPen(QColor("#FDE68A") if is_root else QColor("#A7F3D0"))
                    painter.drawText(
                        QRectF(cx - 10, sy - 7, 20, 14),
                        Qt.AlignmentFlag.AlignCenter,
                        pc_to_name(pc),
                    )

        # Layer 2: Visual finger hovers (no audio yet)
        for note in self._hover:
            if note.string_idx not in string_ys:
                continue
            cx = self._fret_center_x(note.fret, left_x, neck_w, nut_x)
            sy = string_ys[note.string_idx]
            painter.setPen(QPen(QColor("#06B6D4"), 1.8, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(6, 182, 212, 45))
            painter.drawEllipse(QPointF(cx, sy), 11.0, 11.0)

        # Layer 3: Sounding / Fused active notes
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for note in self._sounding:
            if note.string_idx not in string_ys:
                continue
            cx = self._fret_center_x(note.fret, left_x, neck_w, nut_x)
            sy = string_ys[note.string_idx]
            fill = QColor("#3B82F6") if note.source == "av_fused" else QColor("#F59E0B")
            painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
            painter.setBrush(fill)
            painter.drawEllipse(QPointF(cx, sy), 12.0, 12.0)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(
                QRectF(cx - 14, sy - 8, 28, 16),
                Qt.AlignmentFlag.AlignCenter,
                midi_to_name(note.midi, include_octave=False),
            )

    @staticmethod
    def _draw_legend(painter: QPainter, right_x: float, y: float) -> None:
        items = (
            ("Sounding (AV)", QColor("#3B82F6")),
            ("Audio Only", QColor("#F59E0B")),
            ("Finger Hover", QColor("#06B6D4")),
            ("Key Root", QColor("#F59E0B")),
            ("Scale", QColor("#10B981")),
        )
        painter.setFont(QFont("Segoe UI", 8))
        x = right_x
        for label, col in reversed(items):
            tw = painter.fontMetrics().horizontalAdvance(label)
            x -= tw + 16
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(col)
            painter.drawEllipse(QPointF(x, y - 4), 4.0, 4.0)
            painter.setPen(QColor("#8B949E"))
            painter.drawText(QPointF(x + 7, y), label)
            x -= 12
