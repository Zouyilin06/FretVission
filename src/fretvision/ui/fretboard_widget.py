"""Interactive 2D virtual guitar fretboard widget showing scale degrees, hover, pins, and notes."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from fretvision.config import FretboardLabelMode, Handedness
from fretvision.fusion.av_fusion import ActiveNote
from fretvision.theory.key_detector import KeyEstimate
from fretvision.theory.music import base_open_midis, midi_to_name, pc_to_name
from fretvision.ui.theme import PALETTE

_INTERVAL_NAMES: tuple[str, ...] = (
    "R", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"
)


class _FretboardCanvas(QWidget):
    """High-DPI QPainter fretboard supporting hover tooltips, target pins, and degree mode."""

    pin_toggled = Signal(int, int)  # string_idx, fret

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(175)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._tuning_name: str = "Standard (EADGBE)"
        self._capo: int = 0
        self._max_fret: int = 22
        self._handedness: Handedness = "right"
        self._show_scale_overlay: bool = True
        self._label_mode: FretboardLabelMode = "notes"

        self._sounding: tuple[ActiveNote, ...] = ()
        self._hover: tuple[ActiveNote, ...] = ()
        self._key_estimate: KeyEstimate | None = None
        self._pinned: set[tuple[int, int]] = set()
        self._hover_cell: tuple[int, int] | None = None
        self._stage_mode: bool = False

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

    def set_label_mode(self, mode: FretboardLabelMode) -> None:
        self._label_mode = mode
        self.update()

    def set_stage_mode(self, enabled: bool) -> None:
        self._stage_mode = bool(enabled)
        self.update()

    def clear_pins(self) -> None:
        self._pinned.clear()
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

    def _neck_geometry(self) -> tuple[float, float, float, float, float, float]:
        w = float(self.width())
        h = float(self.height())
        pad_x = 64.0
        top_y = 14.0
        bot_y = h - 22.0
        neck_h = max(90.0, bot_y - top_y)
        left_x = pad_x + (10.0 if self._handedness == "right" else 0.0)
        neck_w = max(200.0, w - 2.0 * pad_x - 10.0)
        return left_x, top_y, neck_w, neck_h, w, h

    def _fret_x(self, fret: int, left_x: float, neck_w: float) -> float:
        if fret <= 0:
            norm = 0.0
        else:
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

    def _cell_at_widget_pos(self, px: float, py: float) -> tuple[int, int] | None:
        left_x, top_y, neck_w, neck_h, _, _ = self._neck_geometry()
        if not (top_y <= py <= top_y + neck_h):
            return None
        row_idx = int((py - top_y) / (neck_h / 6.0))
        row_idx = max(0, min(5, row_idx))
        s_idx = 5 - row_idx

        nut_x = self._fret_x(0, left_x, neck_w)
        if abs(px - (nut_x + (-18.0 if self._handedness == "right" else 18.0))) <= 16.0:
            return s_idx, 0

        for f in range(1, self._max_fret + 1):
            x0 = self._fret_x(f - 1, left_x, neck_w)
            x1 = self._fret_x(f, left_x, neck_w)
            lo, hi = min(x0, x1), max(x0, x1)
            if lo <= px <= hi:
                return s_idx, f
        return None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        cell = self._cell_at_widget_pos(pos.x(), pos.y())
        if cell != self._hover_cell:
            self._hover_cell = cell
            self.update()
        if cell is not None:
            s_idx, fret = cell
            base_midis = base_open_midis(self._tuning_name)
            midi = base_midis[s_idx] + max(self._capo, fret)
            deg_str = ""
            if self._key_estimate is not None:
                rel = (midi % 12 - self._key_estimate.tonic_pc) % 12
                deg_str = f" · {_INTERVAL_NAMES[rel]} in {self._key_estimate.display_name}"
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"String {6 - s_idx} · Fret {fret} · {midi_to_name(midi)}{deg_str}",
                self,
            )
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: object) -> None:
        _ = event
        if self._hover_cell is not None:
            self._hover_cell = None
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.clear_pins()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            cell = self._cell_at_widget_pos(event.position().x(), event.position().y())
            if cell is not None:
                if cell in self._pinned:
                    self._pinned.remove(cell)
                else:
                    self._pinned.add(cell)
                self.pin_toggled.emit(cell[0], cell[1])
                self.update()
                return
        super().mousePressEvent(event)

    def _scale_badge_text(self, pc: int, tonic_pc: int) -> str:
        if self._label_mode == "degrees":
            return _INTERVAL_NAMES[(pc - tonic_pc) % 12]
        return pc_to_name(pc)

    def paintEvent(self, event: object) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#161B22"))

        left_x, top_y, neck_w, neck_h, _, _ = self._neck_geometry()

        # Subtle rosewood gradient fretboard body
        wood_grad = QLinearGradient(left_x, top_y, left_x, top_y + neck_h)
        wood_grad.setColorAt(0.0, QColor(PALETTE["fretboard_wood_top"]))
        wood_grad.setColorAt(0.5, QColor("#241C18"))
        wood_grad.setColorAt(1.0, QColor(PALETTE["fretboard_wood_bot"]))
        painter.setPen(QPen(QColor("#30394A"), 1.2))
        painter.setBrush(wood_grad)
        painter.drawRoundedRect(QRectF(left_x, top_y, neck_w, neck_h), 5.0, 5.0)

        nut_x = self._fret_x(0, left_x, neck_w)
        # Pearloid dot inlays
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#3B4252"))
        for f in (3, 5, 7, 9, 12, 15, 17, 19, 21):
            if f > self._max_fret:
                continue
            cx = self._fret_center_x(f, left_x, neck_w, nut_x)
            if f == 12:
                painter.drawEllipse(QPointF(cx, top_y + neck_h * 0.30), 5.0, 5.0)
                painter.drawEllipse(QPointF(cx, top_y + neck_h * 0.70), 5.0, 5.0)
            else:
                painter.drawEllipse(QPointF(cx, top_y + neck_h * 0.50), 5.0, 5.0)

        # Metallic fret wires & numbers
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if self._stage_mode else QFont.Weight.Normal))
        for f in range(0, self._max_fret + 1):
            fx = self._fret_x(f, left_x, neck_w)
            if f == 0:
                painter.setPen(QPen(QColor("#E2E8F0"), 4.5))
                painter.drawLine(QPointF(fx, top_y), QPointF(fx, top_y + neck_h))
            elif f == self._capo and self._capo > 0:
                painter.setPen(QPen(QColor("#F59E0B"), 5.0))
                painter.drawLine(QPointF(fx, top_y - 2), QPointF(fx, top_y + neck_h + 2))
            else:
                painter.setPen(QPen(QColor("#334155"), 2.2))
                painter.drawLine(QPointF(fx + 1.0, top_y), QPointF(fx + 1.0, top_y + neck_h))
                painter.setPen(QPen(QColor("#94A3B8"), 1.3))
                painter.drawLine(QPointF(fx, top_y), QPointF(fx, top_y + neck_h))

            if f in (0, 3, 5, 7, 9, 12, 15, 17, 19, 21):
                lx = fx if f == 0 else self._fret_center_x(f, left_x, neck_w, nut_x)
                painter.setPen(QColor("#94A3B8"))
                painter.drawText(
                    QRectF(lx - 12, top_y + neck_h + 3, 24, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    str(f),
                )

        # Strings (High E top -> Low E bottom)
        base_midis = base_open_midis(self._tuning_name)
        string_ys: dict[int, float] = {}
        for row_idx, s_idx in enumerate(reversed(range(6))):
            sy = top_y + (row_idx + 0.5) * (neck_h / 6.0)
            string_ys[s_idx] = sy
            thickness = 1.0 + (5 - s_idx) * 0.28
            painter.setPen(QPen(QColor("#A3B1C2"), thickness))
            painter.drawLine(QPointF(left_x, sy), QPointF(left_x + neck_w, sy))

            open_pc_name = (
                "e"
                if (s_idx == 5 and base_midis[5] == 64 and self._capo == 0)
                else pc_to_name((base_midis[s_idx] + self._capo) % 12)
            )
            lbl_x = nut_x - 56.0 if self._handedness == "right" else nut_x + 32.0
            painter.setPen(QColor("#94A3B8"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(
                QRectF(lbl_x, sy - 9, 22, 18),
                Qt.AlignmentFlag.AlignCenter,
                open_pc_name,
            )

        # Hover cell subtle highlight
        if self._hover_cell is not None and self._hover_cell[0] in string_ys:
            hs, hf = self._hover_cell
            hx = self._fret_center_x(hf, left_x, neck_w, nut_x)
            hy = string_ys[hs]
            painter.setPen(QPen(QColor(255, 255, 255, 110), 1.2))
            painter.setBrush(QColor(255, 255, 255, 25))
            painter.drawEllipse(QPointF(hx, hy), 11.5, 11.5)

        # Layer 1: Scale notes / degrees for detected key
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
                        painter.setPen(QPen(QColor(PALETTE["accent_amber"]), 1.7))
                        painter.setBrush(QColor(245, 158, 11, 58))
                    else:
                        painter.setPen(QPen(QColor(16, 185, 129, 135), 1.0))
                        painter.setBrush(QColor(16, 185, 129, 28))
                    painter.drawEllipse(QPointF(cx, sy), 8.5, 8.5)
                    painter.setPen(QColor("#FDE68A") if is_root else QColor("#A7F3D0"))
                    painter.drawText(
                        QRectF(cx - 11, sy - 7, 22, 14),
                        Qt.AlignmentFlag.AlignCenter,
                        self._scale_badge_text(pc, tonic_pc),
                    )

        # Pinned target markers
        for ps, pf in self._pinned:
            if ps in string_ys and self._capo <= pf <= self._max_fret:
                px = self._fret_center_x(pf, left_x, neck_w, nut_x)
                py = string_ys[ps]
                painter.setPen(QPen(QColor("#F472B6"), 2.0))
                painter.setBrush(QColor(244, 114, 182, 42))
                painter.drawEllipse(QPointF(px, py), 11.5, 11.5)

        # Layer 2: Visual finger hovers
        for note in self._hover:
            if note.string_idx not in string_ys:
                continue
            cx = self._fret_center_x(note.fret, left_x, neck_w, nut_x)
            sy = string_ys[note.string_idx]
            painter.setPen(QPen(QColor(PALETTE["accent_cyan"]), 1.8, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(6, 182, 212, 45))
            painter.drawEllipse(QPointF(cx, sy), 11.5, 11.5)

        # Layer 3: Sounding active notes (distinct AV blue vs Audio-Only violet)
        radius = 13.5 if self._stage_mode else 12.0
        painter.setFont(QFont("Segoe UI", 9 if self._stage_mode else 8, QFont.Weight.Bold))
        for note in self._sounding:
            if note.string_idx not in string_ys:
                continue
            cx = self._fret_center_x(note.fret, left_x, neck_w, nut_x)
            sy = string_ys[note.string_idx]
            fill = QColor(PALETTE["accent_blue"]) if note.source == "av_fused" else QColor(PALETTE["audio_only"])
            painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
            painter.setBrush(fill)
            painter.drawEllipse(QPointF(cx, sy), radius, radius)
            painter.setPen(QColor("#FFFFFF"))
            if self._label_mode == "degrees" and self._key_estimate is not None:
                badge = _INTERVAL_NAMES[(note.midi % 12 - self._key_estimate.tonic_pc) % 12]
            else:
                badge = midi_to_name(note.midi, include_octave=False)
            painter.drawText(
                QRectF(cx - 15, sy - 8, 30, 16),
                Qt.AlignmentFlag.AlignCenter,
                badge,
            )


class VirtualFretboardWidget(QWidget):
    """Studio Virtual Fretboard container with header legend, Notes/Degrees toggle, and canvas."""

    label_mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(205)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 4)
        root.setSpacing(2)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("VIRTUAL FRETBOARD")
        title.setObjectName("CardTitle")
        header.addWidget(title)

        self.notes_btn = QPushButton("Notes")
        self.notes_btn.setObjectName("CompactPill")
        self.notes_btn.setCheckable(True)
        self.notes_btn.setChecked(True)
        self.degrees_btn = QPushButton("Degrees")
        self.degrees_btn.setObjectName("CompactPill")
        self.degrees_btn.setCheckable(True)
        header.addWidget(self.notes_btn)
        header.addWidget(self.degrees_btn)

        self.clear_pins_btn = QPushButton("Clear Pins")
        self.clear_pins_btn.setObjectName("CompactPill")
        self.clear_pins_btn.setVisible(False)
        header.addWidget(self.clear_pins_btn)

        header.addStretch(1)

        legend_items = (
            ("Sounding (AV)", PALETTE["accent_blue"]),
            ("Audio Only", PALETTE["audio_only"]),
            ("Hover", PALETTE["accent_cyan"]),
            ("Key Root", PALETTE["accent_amber"]),
            ("Scale", PALETTE["accent_emerald"]),
        )
        for lbl_txt, hex_col in legend_items:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {hex_col}; font-size: 11px; background: transparent;")
            lbl = QLabel(lbl_txt)
            lbl.setObjectName("SubtleLabel")
            header.addWidget(dot)
            header.addWidget(lbl)
        root.addLayout(header)

        self.canvas = _FretboardCanvas()
        root.addWidget(self.canvas, 1)

        self.notes_btn.clicked.connect(lambda: self.set_label_mode("notes"))
        self.degrees_btn.clicked.connect(lambda: self.set_label_mode("degrees"))
        self.clear_pins_btn.clicked.connect(self._on_clear_pins)
        self.canvas.pin_toggled.connect(self._on_pin_changed)

    def _on_pin_changed(self, _s: int, _f: int) -> None:
        self.clear_pins_btn.setVisible(bool(self.canvas._pinned))

    def _on_clear_pins(self) -> None:
        self.canvas.clear_pins()
        self.clear_pins_btn.setVisible(False)

    def set_guitar_setup(
        self,
        tuning_name: str,
        capo: int,
        handedness: Handedness = "right",
        max_fret: int = 22,
    ) -> None:
        self.canvas.set_guitar_setup(tuning_name, capo, handedness, max_fret)

    def set_show_scale_overlay(self, enabled: bool) -> None:
        self.canvas.set_show_scale_overlay(enabled)

    def set_label_mode(self, mode: FretboardLabelMode) -> None:
        norm_mode: FretboardLabelMode = "degrees" if mode == "degrees" else "notes"
        self.notes_btn.setChecked(norm_mode == "notes")
        self.degrees_btn.setChecked(norm_mode == "degrees")
        self.canvas.set_label_mode(norm_mode)
        self.label_mode_changed.emit(norm_mode)

    def set_stage_mode(self, enabled: bool) -> None:
        self.canvas.set_stage_mode(enabled)

    def update_state(
        self,
        sounding_notes: tuple[ActiveNote, ...],
        hover_notes: tuple[ActiveNote, ...],
        key_estimate: KeyEstimate | None,
    ) -> None:
        self.canvas.update_state(sounding_notes, hover_notes, key_estimate)
