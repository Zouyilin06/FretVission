"""Live camera feed widget with AR fretboard grid, hand skeleton, and 4-corner calibration."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from fretvision.fusion.av_fusion import ActiveNote
from fretvision.vision.fretboard_model import FretboardModel
from fretvision.vision.hand_tracker import HAND_CONNECTIONS, HandTrackingResult

_CALIBRATION_PROMPTS: tuple[str, ...] = (
    "Click 1/4: Start Fret — Low-E (Top) Edge",
    "Click 2/4: End Fret — Low-E (Top) Edge",
    "Click 3/4: End Fret — High-E (Bottom) Edge",
    "Click 4/4: Start Fret — High-E (Bottom) Edge",
)


class CameraViewWidget(QWidget):
    """Displays live BGR frames with AR guitar overlays and interactive calibration."""

    calibration_completed = Signal(object)  # np.ndarray shape (4, 2)
    calibration_cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(460, 290)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._qimage: QImage | None = None
        self._frame_size: tuple[int, int] = (0, 0)  # (w, h)
        self._image_rect: QRectF = QRectF()

        self._fretboard: FretboardModel | None = None
        self._hand_result: HandTrackingResult | None = None
        self._sounding_notes: tuple[ActiveNote, ...] = ()
        self._status_banner: str = "Connect a camera or enable Demo Mode to begin"

        self._calibrating: bool = False
        self._calib_points: list[tuple[float, float]] = []
        self._calib_start_fret: int = 0
        self._calib_end_fret: int = 12
        self._hover_img_pt: tuple[float, float] | None = None

    def set_status_banner(self, text: str) -> None:
        self._status_banner = text
        self.update()

    def start_calibration(self, start_fret: int, end_fret: int) -> None:
        """Enter 4-corner interactive calibration mode."""
        self._calibrating = True
        self._calib_points.clear()
        self._calib_start_fret = int(start_fret)
        self._calib_end_fret = int(end_fret)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def cancel_calibration(self) -> None:
        if self._calibrating:
            self._calibrating = False
            self._calib_points.clear()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.calibration_cancelled.emit()
            self.update()

    def update_scene(
        self,
        frame_bgr: np.ndarray | None,
        fretboard: FretboardModel | None,
        hand_result: HandTrackingResult | None,
        sounding_notes: tuple[ActiveNote, ...],
    ) -> None:
        """Update displayed video frame and AR annotations."""
        if frame_bgr is not None and frame_bgr.size > 0:
            h, w = frame_bgr.shape[:2]
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
            self._qimage = qimg
            self._frame_size = (w, h)
        self._fretboard = fretboard
        self._hand_result = hand_result
        self._sounding_notes = sounding_notes
        self.update()

    def _compute_image_rect(self) -> QRectF:
        fw, fh = self._frame_size
        if fw <= 0 or fh <= 0:
            return QRectF(self.rect())
        ww, wh = float(self.width()), float(self.height())
        scale = min(ww / fw, wh / fh)
        dw, dh = fw * scale, fh * scale
        ox = (ww - dw) * 0.5
        oy = (wh - dh) * 0.5
        return QRectF(ox, oy, dw, dh)

    def _img_to_widget(self, x: float, y: float) -> QPointF:
        fw, fh = self._frame_size
        if fw <= 0 or fh <= 0 or self._image_rect.width() <= 0:
            return QPointF(x, y)
        wx = self._image_rect.x() + (x / fw) * self._image_rect.width()
        wy = self._image_rect.y() + (y / fh) * self._image_rect.height()
        return QPointF(wx, wy)

    def _widget_to_img(self, wx: float, wy: float) -> tuple[float, float] | None:
        fw, fh = self._frame_size
        r = self._image_rect
        if fw <= 0 or fh <= 0 or r.width() <= 0 or r.height() <= 0:
            return None
        if not r.contains(wx, wy):
            return None
        ix = ((wx - r.x()) / r.width()) * fw
        iy = ((wy - r.y()) / r.height()) * fh
        return float(ix), float(iy)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._calibrating:
            pos = event.position()
            self._hover_img_pt = self._widget_to_img(pos.x(), pos.y())
            self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._calibrating:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.cancel_calibration()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            img_pt = self._widget_to_img(pos.x(), pos.y())
            if img_pt is not None:
                self._calib_points.append(img_pt)
                if len(self._calib_points) == 4:
                    corners = np.array(self._calib_points, dtype=np.float32)
                    self._calibrating = False
                    self._calib_points.clear()
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    self.calibration_completed.emit(corners)
                self.update()

    def paintEvent(self, event: object) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090C10"))

        if self._qimage is None:
            painter.setPen(QColor("#8B949E"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._status_banner)
            return

        self._image_rect = self._compute_image_rect()
        painter.drawImage(self._image_rect, self._qimage)

        if self._fretboard is not None:
            self._draw_fretboard_ar(painter, self._fretboard)

        if self._hand_result is not None and self._hand_result.hand_present:
            self._draw_hand_ar(painter, self._hand_result)

        if self._sounding_notes and self._fretboard is not None:
            self._draw_sounding_badges(painter, self._fretboard)

        if self._calibrating:
            self._draw_calibration_overlay(painter)

    def _draw_fretboard_ar(self, painter: QPainter, fb: FretboardModel) -> None:
        # Neck polygon outline
        poly_pts = fb.get_visible_neck_polygon(0, min(18, fb.max_fret))
        qpoly = QPolygonF([self._img_to_widget(float(x), float(y)) for x, y in poly_pts])
        painter.setPen(QPen(QColor(59, 130, 246, 145), 1.6))
        painter.setBrush(QColor(15, 23, 42, 35))
        painter.drawPolygon(qpoly)

        # String lines (0 = Low E thickest -> 5 = High E thinnest)
        for s in range(6):
            p0, p1 = fb.string_segment(s, 0, min(18, fb.max_fret))
            width = 2.0 - s * 0.22
            painter.setPen(QPen(QColor(226, 232, 240, 145), max(0.9, width)))
            painter.drawLine(self._img_to_widget(*p0), self._img_to_widget(*p1))

        # Fret wires
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for f in range(0, min(19, fb.max_fret + 1)):
            p0, p1 = fb.fret_wire_segment(f)
            w0 = self._img_to_widget(*p0)
            w1 = self._img_to_widget(*p1)
            if f == 0:
                painter.setPen(QPen(QColor(245, 158, 11, 210), 2.8))
            else:
                painter.setPen(QPen(QColor(148, 163, 184, 155), 1.2))
            painter.drawLine(w0, w1)
            if f in (3, 5, 7, 9, 12, 15, 17):
                painter.setPen(QColor(248, 250, 252, 185))
                painter.drawText(QPointF(w1.x() - 6, w1.y() + 14), str(f))

    def _draw_hand_ar(self, painter: QPainter, hand: HandTrackingResult) -> None:
        pts = hand.landmarks_px
        if len(pts) < 21:
            return
        painter.setPen(QPen(QColor(6, 182, 212, 150), 1.5))
        for i, j in HAND_CONNECTIONS:
            painter.drawLine(self._img_to_widget(*pts[i]), self._img_to_widget(*pts[j]))

        for f in hand.fingers:
            wp = self._img_to_widget(*f.pixel_xy)
            color = QColor("#10B981") if f.in_bounds else QColor("#94A3B8")
            painter.setPen(QPen(QColor("#0D1117"), 1.5))
            painter.setBrush(color)
            painter.drawEllipse(wp, 5.5, 5.5)

    def _draw_sounding_badges(self, painter: QPainter, fb: FretboardModel) -> None:
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        for note in self._sounding_notes:
            px, py = fb.fret_string_center_px(note.string_idx, note.fret)
            wp = self._img_to_widget(px, py)
            col = QColor("#3B82F6") if note.source == "av_fused" else QColor("#F59E0B")
            painter.setPen(QPen(QColor("#FFFFFF"), 1.6))
            painter.setBrush(col)
            painter.drawEllipse(wp, 10.5, 10.5)
            painter.drawText(
                QRectF(wp.x() - 16, wp.y() - 10, 32, 20),
                Qt.AlignmentFlag.AlignCenter,
                note.note_name[:-1],
            )

    def _draw_calibration_overlay(self, painter: QPainter) -> None:
        step = min(len(self._calib_points), 3)
        prompt = _CALIBRATION_PROMPTS[step]
        sub = f"Range: Fret {self._calib_start_fret} -> {self._calib_end_fret}  (Right-click to cancel)"

        banner = QRectF(12, 12, min(self.width() - 24, 460), 48)
        painter.setPen(QPen(QColor("#3B82F6"), 1.5))
        painter.setBrush(QColor(13, 17, 23, 220))
        painter.drawRoundedRect(banner, 8, 8)
        painter.setPen(QColor("#F0F6FC"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(banner.x() + 10, banner.y() + 5, banner.width() - 20, 20), prompt)
        painter.setPen(QColor("#93C5FD"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(QRectF(banner.x() + 10, banner.y() + 25, banner.width() - 20, 18), sub)

        # Draw placed corners and connecting edges
        w_pts = [self._img_to_widget(x, y) for x, y in self._calib_points]
        if len(w_pts) >= 2:
            painter.setPen(QPen(QColor("#F59E0B"), 2.0))
            for idx in range(len(w_pts) - 1):
                painter.drawLine(w_pts[idx], w_pts[idx + 1])
        if self._hover_img_pt is not None and w_pts:
            painter.setPen(QPen(QColor(245, 158, 11, 140), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(w_pts[-1], self._img_to_widget(*self._hover_img_pt))

        for idx, wp in enumerate(w_pts):
            painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
            painter.setBrush(QColor("#F59E0B"))
            painter.drawEllipse(wp, 6.0, 6.0)
            painter.drawText(QPointF(wp.x() + 8, wp.y() - 6), str(idx + 1))
