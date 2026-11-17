"""Musician telemetry widgets: Chord Progression Strip, Precision Tuner, and Mini Spectrum."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget


class ChordHistoryStrip(QWidget):
    """Displays a rolling progression of recent confirmed chords (e.g. G -> Em -> C -> D)."""

    def __init__(self, max_chords: int = 8, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.max_chords = int(max_chords)
        self._chords: list[str] = []
        self._pending_chord: str | None = None
        self._pending_count: int = 0
        self.setMinimumHeight(34)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(6)

        self._pills_layout = QHBoxLayout()
        self._pills_layout.setContentsMargins(0, 0, 0, 0)
        self._pills_layout.setSpacing(5)
        root.addLayout(self._pills_layout, 1)

        self._empty_lbl = QLabel("Chord history appears as you strum…")
        self._empty_lbl.setObjectName("SubtleLabel")
        self._pills_layout.addWidget(self._empty_lbl)
        self._pills_layout.addStretch(1)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("CompactPill")
        self.clear_btn.clicked.connect(self.clear)
        root.addWidget(self.clear_btn)

    @property
    def chords(self) -> list[str]:
        return list(self._chords)

    def clear(self) -> None:
        self._chords.clear()
        self._pending_chord = None
        self._pending_count = 0
        self._rebuild_pills()

    def ingest_chord(self, symbol: str | None, is_onset: bool = False) -> bool:
        """Record a chord into the progression strip after brief stability confirmation."""
        if not symbol:
            self._pending_chord = None
            self._pending_count = 0
            return False

        if self._chords and self._chords[-1] == symbol:
            self._pending_chord = None
            self._pending_count = 0
            return False

        if is_onset:
            self._chords.append(symbol)
            self._pending_chord = None
            self._pending_count = 0
        else:
            if self._pending_chord == symbol:
                self._pending_count += 1
            else:
                self._pending_chord = symbol
                self._pending_count = 1
            if self._pending_count < 2:
                return False
            self._chords.append(symbol)
            self._pending_chord = None
            self._pending_count = 0

        if len(self._chords) > self.max_chords:
            self._chords = self._chords[-self.max_chords :]
        self._rebuild_pills()
        return True

    def _rebuild_pills(self) -> None:
        while self._pills_layout.count():
            item = self._pills_layout.takeAt(0)
            if item.widget() is not None and item.widget() is not self._empty_lbl:
                item.widget().deleteLater()

        if not self._chords:
            self._empty_lbl.setVisible(True)
            self._pills_layout.addWidget(self._empty_lbl)
            self._pills_layout.addStretch(1)
            return

        self._empty_lbl.setVisible(False)
        for idx, sym in enumerate(self._chords):
            if idx > 0:
                arrow = QLabel("→")
                arrow.setObjectName("SubtleLabel")
                self._pills_layout.addWidget(arrow)
            pill = QLabel(sym)
            is_last = idx == len(self._chords) - 1
            if is_last:
                pill.setStyleSheet(
                    "background-color: rgba(245, 158, 11, 0.20); color: #FCD34D;"
                    "border: 1px solid rgba(245, 158, 11, 0.65); border-radius: 7px;"
                    "padding: 2px 9px; font-weight: 700; font-size: 12px;"
                )
            else:
                pill.setStyleSheet(
                    "background-color: #1F2631; color: #CBD5E1;"
                    "border: 1px solid #30394A; border-radius: 7px;"
                    "padding: 2px 8px; font-weight: 600; font-size: 12px;"
                )
            self._pills_layout.addWidget(pill)
        self._pills_layout.addStretch(1)


class PrecisionTunerGauge(QWidget):
    """High-clarity -50c .. +50c chromatic pitch & bend tuner rail."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        self._cents: float = 0.0
        self._active: bool = False
        self._note_label: str = ""

    def set_state(self, cents: float, active: bool, note_label: str = "") -> None:
        self._cents = max(-50.0, min(50.0, float(cents)))
        self._active = active
        self._note_label = note_label
        self.update()

    def paintEvent(self, event: object) -> None:
        _ = event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        rail_x = 24.0
        rail_w = max(80.0, w - 48.0)
        mid = rail_x + rail_w * 0.5
        rail_y = 8.0
        rail_h = 16.0

        # Track background
        p.setPen(QPen(QColor("#2A3241"), 1.0))
        p.setBrush(QColor("#0B0F15"))
        p.drawRoundedRect(QRectF(rail_x, rail_y, rail_w, rail_h), 6.0, 6.0)

        # In-tune deadband (+-6 cents)
        dead_w = (6.0 / 50.0) * (rail_w * 0.5) * 2.0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(16, 185, 129, 65))
        p.drawRect(QRectF(mid - dead_w * 0.5, rail_y + 1, dead_w, rail_h - 2))

        # Scale ticks (-50, -25, 0, +25, +50)
        p.setFont(QFont("Segoe UI", 7))
        for tick_c in (-50, -25, 0, 25, 50):
            tx = mid + (tick_c / 50.0) * (rail_w * 0.5 - 6.0)
            p.setPen(QPen(QColor("#64748B" if tick_c == 0 else "#334155"), 1.4 if tick_c == 0 else 1.0))
            p.drawLine(QPointF(tx, rail_y + 2), QPointF(tx, rail_y + rail_h - 2))
            p.setPen(QColor("#8B949E"))
            lbl = "0" if tick_c == 0 else f"{tick_c:+d}"
            p.drawText(QRectF(tx - 14, rail_y + rail_h + 1, 28, 11), Qt.AlignmentFlag.AlignCenter, lbl)

        # Flat / Sharp direction glyphs
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        flat_col = QColor("#F59E0B") if (self._active and self._cents < -6.0) else QColor("#475569")
        sharp_col = QColor("#F59E0B") if (self._active and self._cents > 6.0) else QColor("#475569")
        p.setPen(flat_col)
        p.drawText(QRectF(2, rail_y - 1, 20, rail_h + 2), Qt.AlignmentFlag.AlignCenter, "♭")
        p.setPen(sharp_col)
        p.drawText(QRectF(w - 22, rail_y - 1, 20, rail_h + 2), Qt.AlignmentFlag.AlignCenter, "♯")

        if self._active:
            nx = mid + (self._cents / 50.0) * (rail_w * 0.5 - 6.0)
            in_tune = abs(self._cents) <= 6.0
            col = QColor("#10B981") if in_tune else (QColor("#F59E0B") if abs(self._cents) <= 22.0 else QColor("#EF4444"))
            p.setPen(QPen(QColor("#FFFFFF"), 1.5))
            p.setBrush(col)
            p.drawEllipse(QPointF(nx, rail_y + rail_h * 0.5), 6.5, 6.5)


class MiniSpectrumWidget(QWidget):
    """24-band real-time audio spectrum visualizer with peak-hold indicators."""

    def __init__(self, n_bands: int = 24, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.n_bands = int(n_bands)
        self._levels: list[float] = [0.0] * self.n_bands
        self._peaks: list[float] = [0.0] * self.n_bands
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_bands(self, bands: tuple[float, ...] | list[float]) -> None:
        if not bands:
            self._levels = [v * 0.72 for v in self._levels]
            self._peaks = [max(0.0, p - 0.04) for p in self._peaks]
        else:
            vals = list(bands[: self.n_bands])
            if len(vals) < self.n_bands:
                vals.extend([0.0] * (self.n_bands - len(vals)))
            for i in range(self.n_bands):
                lvl = max(0.0, min(1.0, float(vals[i])))
                self._levels[i] = 0.65 * lvl + 0.35 * self._levels[i]
                self._peaks[i] = max(self._levels[i], self._peaks[i] - 0.03)
        self.update()

    def paintEvent(self, event: object) -> None:
        _ = event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())

        p.setPen(QPen(QColor("#252D3B"), 1.0))
        p.setBrush(QColor("#0B0F15"))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 6.0, 6.0)

        pad_x = 6.0
        pad_y = 5.0
        inner_w = max(20.0, w - 2.0 * pad_x)
        inner_h = max(10.0, h - 2.0 * pad_y)
        slot_w = inner_w / self.n_bands
        bar_w = max(2.0, slot_w * 0.68)

        grad = QLinearGradient(0.0, pad_y + inner_h, 0.0, pad_y)
        grad.setColorAt(0.0, QColor("#10B981"))
        grad.setColorAt(0.65, QColor("#3B82F6"))
        grad.setColorAt(1.0, QColor("#F59E0B"))

        for i in range(self.n_bands):
            bx = pad_x + i * slot_w + (slot_w - bar_w) * 0.5
            lvl = self._levels[i]
            bh = max(2.0, lvl * inner_h)
            by = pad_y + inner_h - bh
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(grad if lvl > 0.04 else QColor("#1E293B"))
            p.drawRoundedRect(QRectF(bx, by, bar_w, bh), 1.5, 1.5)

            pk = self._peaks[i]
            if pk > 0.08:
                py = pad_y + inner_h - pk * inner_h
                p.setPen(QPen(QColor("#93C5FD"), 1.2))
                p.drawLine(QPointF(bx, py), QPointF(bx + bar_w, py))
