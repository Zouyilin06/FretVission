"""Main desktop window for FretVision."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fretvision.audio.audio_source import enumerate_audio_devices
from fretvision.config import AppConfig, CalibrationProfile
from fretvision.theory.music import TUNING_PRESETS, pc_to_name
from fretvision.vision.model_assets import default_model_path
from fretvision.vision.video_source import enumerate_cameras

from .camera_widget import CameraViewWidget
from .fretboard_widget import VirtualFretboardWidget
from .tab_widget import LiveTabWidget
from .theme import DARK_STYLESHEET
from .workers import EngineSnapshot, ModelDownloadWorker, ProcessingWorker


class _TunerGauge(QWidget):
    """Compact -50c .. +50c chromatic pitch / bend needle gauge."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(26)
        self._cents: float = 0.0
        self._active: bool = False

    def set_cents(self, cents: float, active: bool) -> None:
        self._cents = max(-50.0, min(50.0, float(cents)))
        self._active = active
        self.update()

    def paintEvent(self, event: object) -> None:
        _ = event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        mid = w * 0.5

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#0D1117"))
        p.drawRoundedRect(0, 4, int(w), int(h - 8), 5, 5)

        # Center target zone (-5c .. +5c)
        p.setBrush(QColor(16, 185, 129, 70))
        p.drawRect(int(mid - 10), 4, 20, int(h - 8))
        p.setPen(QPen(QColor("#4B5563"), 1.2))
        p.drawLine(int(mid), 2, int(mid), int(h - 2))

        if self._active:
            nx = mid + (self._cents / 50.0) * (mid - 10.0)
            in_tune = abs(self._cents) <= 8.0
            col = QColor("#10B981") if in_tune else QColor("#F59E0B")
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawEllipse(int(nx - 6), int(h * 0.5 - 6), 12, 12)


class MainWindow(QMainWindow):
    """Primary PySide6 desktop application window."""

    def __init__(
        self,
        config: AppConfig | None = None,
        start_workers: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("FretVision — Live Guitar Key, Note & Tab Analyzer")
        self.resize(1340, 880)
        self.setStyleSheet(DARK_STYLESHEET)

        self.config = config or AppConfig(profile=CalibrationProfile.load("default"))
        self.worker = ProcessingWorker(self.config)
        self._download_worker: ModelDownloadWorker | None = None
        self._latest_snapshot: EngineSnapshot | None = None

        self._build_ui()
        self._connect_signals()
        self._populate_devices()

        if start_workers:
            self.worker.start()
            QTimer.singleShot(500, self._check_hand_model)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        # --- Top Device & Guitar Setup Bar ---
        top_card = QFrame()
        top_card.setObjectName("Card")
        top_layout = QVBoxLayout(top_card)
        top_layout.setContentsMargins(10, 8, 10, 8)
        top_layout.setSpacing(6)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(145)
        row1.addWidget(self.camera_combo)

        self.phone_url_btn = QPushButton("Phone URL…")
        row1.addWidget(self.phone_url_btn)

        self.mirror_chk = QCheckBox("Mirror")
        self.mirror_chk.setChecked(self.config.camera.mirror)
        row1.addWidget(self.mirror_chk)

        row1.addWidget(QLabel("Rotate:"))
        self.rotate_combo = QComboBox()
        self.rotate_combo.addItems(["0°", "90°", "180°", "270°"])
        row1.addWidget(self.rotate_combo)

        row1.addWidget(QLabel("Audio:"))
        self.audio_combo = QComboBox()
        self.audio_combo.setMinimumWidth(210)
        row1.addWidget(self.audio_combo, 1)

        self.refresh_dev_btn = QPushButton("Refresh")
        row1.addWidget(self.refresh_dev_btn)

        row1.addWidget(QLabel("Tuning:"))
        self.tuning_combo = QComboBox()
        self.tuning_combo.addItems(list(TUNING_PRESETS.keys()))
        idx_t = self.tuning_combo.findText(self.config.profile.guitar.tuning_name)
        if idx_t >= 0:
            self.tuning_combo.setCurrentIndex(idx_t)
        row1.addWidget(self.tuning_combo)

        row1.addWidget(QLabel("Capo:"))
        self.capo_spin = QSpinBox()
        self.capo_spin.setRange(0, 12)
        self.capo_spin.setValue(self.config.profile.guitar.capo)
        row1.addWidget(self.capo_spin)

        self.handed_combo = QComboBox()
        self.handed_combo.addItems(["Right-Handed", "Left-Handed"])
        if self.config.profile.guitar.handedness == "left":
            self.handed_combo.setCurrentIndex(1)
        row1.addWidget(self.handed_combo)
        top_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("Calibration Frets:"))
        self.start_fret_spin = QSpinBox()
        self.start_fret_spin.setRange(0, 15)
        self.start_fret_spin.setValue(self.config.profile.start_fret)
        row2.addWidget(self.start_fret_spin)

        row2.addWidget(QLabel("to"))
        self.end_fret_spin = QSpinBox()
        self.end_fret_spin.setRange(3, 24)
        self.end_fret_spin.setValue(self.config.profile.end_fret)
        row2.addWidget(self.end_fret_spin)

        self.calibrate_btn = QPushButton("Calibrate Neck (4-Click)")
        self.calibrate_btn.setObjectName("PrimaryBtn")
        row2.addWidget(self.calibrate_btn)

        self.adjust_corners_btn = QPushButton("Adjust Corners")
        self.adjust_corners_btn.setCheckable(True)
        row2.addWidget(self.adjust_corners_btn)

        self.auto_guess_btn = QPushButton("Auto-Guess Neck")
        row2.addWidget(self.auto_guess_btn)

        self.lock_flow_chk = QCheckBox("Optical Flow Lock")
        self.lock_flow_chk.setChecked(self.config.profile.tracking_mode == "optical_flow")
        row2.addWidget(self.lock_flow_chk)

        self.snap_anchor_btn = QPushButton("Re-Snap Neck")
        row2.addWidget(self.snap_anchor_btn)

        row2.addWidget(QLabel("AV Sync:"))
        self.av_sync_spin = QSpinBox()
        self.av_sync_spin.setRange(-300, 300)
        self.av_sync_spin.setSingleStep(15)
        self.av_sync_spin.setSuffix(" ms")
        self.av_sync_spin.setValue(int(self.config.av_sync_offset_ms))
        self.av_sync_spin.setToolTip("Positive if phone/camera video lags behind microphone audio")
        row2.addWidget(self.av_sync_spin)

        self.reset_key_btn = QPushButton("Reset Key")
        row2.addWidget(self.reset_key_btn)

        self.scale_overlay_chk = QCheckBox("Key Scale Overlay")
        self.scale_overlay_chk.setChecked(True)
        row2.addWidget(self.scale_overlay_chk)

        row2.addStretch(1)
        self.demo_btn = QPushButton("Demo Mode")
        self.demo_btn.setCheckable(True)
        row2.addWidget(self.demo_btn)
        top_layout.addLayout(row2)

        root.addWidget(top_card)

        # --- Middle Split: AR Camera (Left) + Telemetry Cards (Right) ---
        mid_row = QHBoxLayout()
        mid_row.setSpacing(10)

        cam_card = QFrame()
        cam_card.setObjectName("Card")
        cam_lay = QVBoxLayout(cam_card)
        cam_lay.setContentsMargins(8, 8, 8, 8)
        self.camera_widget = CameraViewWidget()
        cam_lay.addWidget(self.camera_widget)
        mid_row.addWidget(cam_card, 6)

        # Right telemetry sidebar
        side_col = QVBoxLayout()
        side_col.setSpacing(10)

        # 1. Key & Scale Card
        key_card = QFrame()
        key_card.setObjectName("Card")
        kl = QVBoxLayout(key_card)
        kl.setContentsMargins(14, 12, 14, 12)
        kl.setSpacing(5)
        k_title = QLabel("DETECTED MUSICAL KEY")
        k_title.setObjectName("CardTitle")
        kl.addWidget(k_title)

        self.key_hero_lbl = QLabel("Listening…")
        self.key_hero_lbl.setObjectName("HeroValue")
        kl.addWidget(self.key_hero_lbl)

        self.key_sub_lbl = QLabel("Play a few notes or a chord progression")
        self.key_sub_lbl.setObjectName("SubtleLabel")
        kl.addWidget(self.key_sub_lbl)

        self.key_conf_bar = QProgressBar()
        self.key_conf_bar.setRange(0, 100)
        self.key_conf_bar.setValue(0)
        self.key_conf_bar.setTextVisible(False)
        kl.addWidget(self.key_conf_bar)

        self.key_candidates_lbl = QLabel("Top candidates: —")
        self.key_candidates_lbl.setObjectName("SubtleLabel")
        kl.addWidget(self.key_candidates_lbl)

        self.scale_notes_lbl = QLabel("Notes: —")
        self.scale_notes_lbl.setStyleSheet("color: #10B981; font-weight: 600; background: transparent;")
        kl.addWidget(self.scale_notes_lbl)
        side_col.addWidget(key_card)

        # 2. Pressed Note / Chord & Tuner Card
        note_card = QFrame()
        note_card.setObjectName("Card")
        nl = QVBoxLayout(note_card)
        nl.setContentsMargins(14, 12, 14, 12)
        nl.setSpacing(6)
        n_title = QLabel("PRESSED NOTE & CHORD")
        n_title.setObjectName("CardTitle")
        nl.addWidget(n_title)

        chord_note_row = QHBoxLayout()
        self.note_hero_lbl = QLabel("—")
        self.note_hero_lbl.setObjectName("HeroValue")
        chord_note_row.addWidget(self.note_hero_lbl, 1)

        self.chord_badge_lbl = QLabel("")
        self.chord_badge_lbl.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.chord_badge_lbl.setStyleSheet("color: #F59E0B; background: transparent;")
        chord_note_row.addWidget(self.chord_badge_lbl)
        nl.addLayout(chord_note_row)

        self.fret_detail_lbl = QLabel("String / Fret: —")
        self.fret_detail_lbl.setWordWrap(True)
        self.fret_detail_lbl.setStyleSheet("color: #CBD5E1; background: transparent;")
        nl.addWidget(self.fret_detail_lbl)

        tuner_row = QHBoxLayout()
        tuner_lbl = QLabel("Pitch / Bend:")
        tuner_lbl.setObjectName("SubtleLabel")
        self.cents_lbl = QLabel("0¢")
        self.cents_lbl.setObjectName("SubtleLabel")
        tuner_row.addWidget(tuner_lbl)
        tuner_row.addStretch(1)
        tuner_row.addWidget(self.cents_lbl)
        nl.addLayout(tuner_row)

        self.tuner_gauge = _TunerGauge()
        nl.addWidget(self.tuner_gauge)
        side_col.addWidget(note_card)

        # 3. Audio Level, Gate & Tracking Diagnostics Card
        diag_card = QFrame()
        diag_card.setObjectName("Card")
        dl = QVBoxLayout(diag_card)
        dl.setContentsMargins(14, 10, 14, 10)
        dl.setSpacing(6)
        d_title = QLabel("SIGNAL & TRACKING STATUS")
        d_title.setObjectName("CardTitle")
        dl.addWidget(d_title)

        rms_row = QHBoxLayout()
        rms_row.addWidget(QLabel("Input Level:"))
        self.rms_bar = QProgressBar()
        self.rms_bar.setRange(0, 100)
        self.rms_bar.setValue(0)
        self.rms_bar.setTextVisible(False)
        rms_row.addWidget(self.rms_bar, 1)
        dl.addLayout(rms_row)

        gate_row = QHBoxLayout()
        gate_row.addWidget(QLabel("Noise Gate:"))
        self.gate_slider = QSlider(Qt.Orientation.Horizontal)
        self.gate_slider.setRange(-72, -20)
        self.gate_slider.setValue(int(self.config.audio.noise_gate_db))
        self.gate_val_lbl = QLabel(f"{int(self.config.audio.noise_gate_db)} dB")
        gate_row.addWidget(self.gate_slider, 1)
        gate_row.addWidget(self.gate_val_lbl)
        dl.addLayout(gate_row)

        self.diag_lbl = QLabel("Neck: Uncalibrated  |  Hand Model: Checking…")
        self.diag_lbl.setObjectName("SubtleLabel")
        self.diag_lbl.setWordWrap(True)
        dl.addWidget(self.diag_lbl)

        self.download_model_btn = QPushButton("Download Hand Tracking Model (~8 MB)")
        self.download_model_btn.setVisible(False)
        dl.addWidget(self.download_model_btn)

        side_col.addWidget(diag_card)
        side_col.addStretch(1)
        mid_row.addLayout(side_col, 4)

        root.addLayout(mid_row, 5)

        # --- Bottom Virtual Fretboard & Live Tablature ---
        fb_card = QFrame()
        fb_card.setObjectName("Card")
        fb_lay = QVBoxLayout(fb_card)
        fb_lay.setContentsMargins(6, 4, 6, 4)
        self.fretboard_widget = VirtualFretboardWidget()
        self.fretboard_widget.set_guitar_setup(
            self.config.profile.guitar.tuning_name,
            self.config.profile.guitar.capo,
            self.config.profile.guitar.handedness,
            self.config.profile.guitar.max_fret,
        )
        fb_lay.addWidget(self.fretboard_widget)
        root.addWidget(fb_card, 2)

        tab_card = QFrame()
        tab_card.setObjectName("Card")
        tab_lay = QVBoxLayout(tab_card)
        tab_lay.setContentsMargins(2, 2, 2, 2)
        self.tab_widget = LiveTabWidget()
        tab_lay.addWidget(self.tab_widget)
        root.addWidget(tab_card, 2)

        self.statusBar().showMessage("Ready")

    def _connect_signals(self) -> None:
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.status_message.connect(self.statusBar().showMessage)

        self.camera_combo.activated.connect(self._on_camera_selected)
        self.phone_url_btn.clicked.connect(self._on_phone_url_clicked)
        self.mirror_chk.toggled.connect(self._on_orientation_changed)
        self.rotate_combo.currentIndexChanged.connect(self._on_orientation_changed)
        self.audio_combo.activated.connect(self._on_audio_selected)
        self.refresh_dev_btn.clicked.connect(self._populate_devices)

        self.tuning_combo.currentTextChanged.connect(self._on_guitar_changed)
        self.capo_spin.valueChanged.connect(self._on_guitar_changed)
        self.handed_combo.currentIndexChanged.connect(self._on_guitar_changed)

        self.calibrate_btn.clicked.connect(self._on_start_calibration)
        self.adjust_corners_btn.toggled.connect(self.camera_widget.set_adjust_mode)
        self.camera_widget.calibration_completed.connect(self._on_calibration_completed)
        self.camera_widget.corners_adjusted.connect(self._on_calibration_completed)
        self.auto_guess_btn.clicked.connect(self._on_auto_guess)
        self.lock_flow_chk.toggled.connect(
            lambda checked: self.worker.set_tracking_mode("optical_flow" if checked else "static")
        )
        self.snap_anchor_btn.clicked.connect(self.worker.reset_neck_to_anchor)
        self.av_sync_spin.valueChanged.connect(lambda val: self.worker.set_av_sync_offset(float(val)))
        self.reset_key_btn.clicked.connect(self.worker.reset_key)
        self.scale_overlay_chk.toggled.connect(self.fretboard_widget.set_show_scale_overlay)
        self.demo_btn.toggled.connect(self.worker.set_demo_mode)

        self.gate_slider.valueChanged.connect(self._on_gate_changed)
        self.download_model_btn.clicked.connect(self._start_model_download)

        self.tab_widget.pause_btn.toggled.connect(self.worker.set_tab_paused)
        self.tab_widget.clear_btn.clicked.connect(self.worker.clear_tab)
        self.tab_widget.export_txt_btn.clicked.connect(self._export_ascii_tab)
        self.tab_widget.export_json_btn.clicked.connect(self._export_json_tab)

    def _populate_devices(self) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        cams = enumerate_cameras(max_probes=4)
        for cam in cams:
            self.camera_combo.addItem(cam.name, cam.index)
        if not cams:
            self.camera_combo.addItem("Default Camera (0)", 0)
        if isinstance(self.config.camera.source, str):
            self.camera_combo.addItem(f"Stream: {self.config.camera.source}", self.config.camera.source)
            self.camera_combo.setCurrentIndex(self.camera_combo.count() - 1)
        self.camera_combo.blockSignals(False)

        self.audio_combo.blockSignals(True)
        self.audio_combo.clear()
        self.audio_combo.addItem("System Default Input", None)
        for dev in enumerate_audio_devices():
            self.audio_combo.addItem(dev.label, dev.index)
        self.audio_combo.blockSignals(False)

    def _check_hand_model(self) -> None:
        if not default_model_path().exists():
            self.download_model_btn.setVisible(True)
            self.statusBar().showMessage(
                "MediaPipe hand_landmarker.task not found — click Download Hand Tracking Model."
            )

    def _start_model_download(self) -> None:
        if self._download_worker is not None and self._download_worker.isRunning():
            return
        self.download_model_btn.setEnabled(False)
        self._download_worker = ModelDownloadWorker()
        self._download_worker.status_message.connect(self.statusBar().showMessage)
        self._download_worker.completed.connect(self._on_model_downloaded)
        self._download_worker.start()

    def _on_model_downloaded(self, ok: bool, detail: str) -> None:
        self.download_model_btn.setEnabled(True)
        if ok:
            self.download_model_btn.setVisible(False)
            self.worker.reload_hand_model()
            self.statusBar().showMessage(f"Hand tracking model ready: {detail}")
        else:
            QMessageBox.warning(self, "Download Failed", f"Could not download model:\n{detail}")

    def _on_camera_selected(self, index: int) -> None:
        src = self.camera_combo.itemData(index)
        if src is not None:
            self.demo_btn.setChecked(False)
            self.worker.set_video_source(src)

    def _on_phone_url_clicked(self) -> None:
        url, ok = QInputDialog.getText(
            self,
            "Connect Phone Camera Stream",
            "Enter MJPEG/HTTP or RTSP URL (e.g. http://192.168.1.50:8080/video):",
            text="http://192.168.1.50:8080/video",
        )
        if ok and url.strip():
            clean = url.strip()
            self.demo_btn.setChecked(False)
            self.camera_combo.addItem(f"Stream: {clean}", clean)
            self.camera_combo.setCurrentIndex(self.camera_combo.count() - 1)
            self.worker.set_video_source(clean)

    def _on_orientation_changed(self) -> None:
        deg = int(self.rotate_combo.currentText().replace("°", ""))
        self.worker.update_video_orientation(self.mirror_chk.isChecked(), deg)

    def _on_audio_selected(self, index: int) -> None:
        dev_idx = self.audio_combo.itemData(index)
        self.worker.set_audio_device(dev_idx)

    def _on_guitar_changed(self) -> None:
        tuning = self.tuning_combo.currentText()
        capo = self.capo_spin.value()
        handed = "left" if self.handed_combo.currentIndex() == 1 else "right"
        self.worker.set_guitar_setup(tuning, capo, handed)
        self.fretboard_widget.set_guitar_setup(
            tuning, capo, handed, self.config.profile.guitar.max_fret
        )

    def _on_start_calibration(self) -> None:
        s_fret = self.start_fret_spin.value()
        e_fret = self.end_fret_spin.value()
        if e_fret <= s_fret:
            QMessageBox.information(self, "Invalid Fret Range", "End fret must be greater than start fret.")
            return
        self.adjust_corners_btn.setChecked(False)
        self.camera_widget.start_calibration(s_fret, e_fret)
        self.statusBar().showMessage("Click 4 fretboard corners on the camera view (Right-click cancels).")

    def _on_calibration_completed(self, corners: object) -> None:
        self.worker.set_calibration(
            corners,  # type: ignore[arg-type]
            self.start_fret_spin.value(),
            self.end_fret_spin.value(),
        )

    def _on_auto_guess(self) -> None:
        s_fret = self.start_fret_spin.value()
        e_fret = self.end_fret_spin.value()
        if e_fret <= s_fret:
            return
        ok = self.worker.auto_guess_calibration(s_fret, e_fret)
        if not ok:
            self.statusBar().showMessage(
                "Auto-Guess couldn't isolate parallel neck edges cleanly — use 4-Click Calibrate."
            )

    def _on_gate_changed(self, value: int) -> None:
        self.gate_val_lbl.setText(f"{value} dB")
        self.worker.set_noise_gate(float(value))

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 6000)

    def _on_snapshot(self, snap: EngineSnapshot) -> None:
        self._latest_snapshot = snap
        fused = snap.fused

        self.camera_widget.update_scene(
            snap.frame_bgr,
            snap.fretboard,
            snap.hand_result,
            fused.sounding_notes,
        )
        self.fretboard_widget.update_state(
            fused.sounding_notes,
            fused.hover_notes,
            fused.key_estimate,
        )
        self.tab_widget.update_tab(snap.tab_columns, snap.tuning_name)

        # Update Key Card
        if fused.key_estimate is not None:
            ke = fused.key_estimate
            self.key_hero_lbl.setText(ke.display_name)
            self.key_sub_lbl.setText(f"Scale: {ke.scale_label}  •  Relative: {ke.relative_name}")
            self.key_conf_bar.setValue(int(ke.confidence * 100))
            cands = "  |  ".join(f"{c.name}" for c in ke.top_candidates)
            self.key_candidates_lbl.setText(f"Top: {cands}")
            ordered_pcs = sorted(
                ke.scale_pcs,
                key=lambda pc: (pc - ke.tonic_pc) % 12,
            )
            self.scale_notes_lbl.setText("Notes: " + "  ".join(pc_to_name(pc) for pc in ordered_pcs))
        else:
            self.key_hero_lbl.setText("Listening…")
            self.key_sub_lbl.setText("Play a few notes or a chord progression")
            self.key_conf_bar.setValue(0)
            self.key_candidates_lbl.setText("Top candidates: —")
            self.scale_notes_lbl.setText("Notes: —")

        # Update Note / Chord / Tuner Card
        if fused.sounding_notes:
            names = " ".join(n.note_name for n in fused.sounding_notes)
            self.note_hero_lbl.setText(names if len(fused.sounding_notes) <= 3 else f"{len(fused.sounding_notes)} Notes")
            self.chord_badge_lbl.setText(fused.chord.symbol if fused.chord else "")
            details = [
                f"Str {n.string_number} · Fret {n.fret} ({n.note_name})"
                + (f" +{n.bend_semitones}st bend" if n.bend_semitones else "")
                for n in fused.sounding_notes
            ]
            self.fret_detail_lbl.setText("  |  ".join(details))
            cents = fused.dominant_cents
            self.cents_lbl.setText(f"{cents:+.0f}¢")
            self.tuner_gauge.set_cents(cents, True)
        elif fused.hover_notes:
            self.note_hero_lbl.setText("Hovering")
            self.chord_badge_lbl.setText("")
            hover_txt = "  |  ".join(
                f"Str {n.string_number} · Fret {n.fret} ({n.note_name})"
                for n in fused.hover_notes
            )
            self.fret_detail_lbl.setText(f"Finger position: {hover_txt}")
            self.cents_lbl.setText("0¢")
            self.tuner_gauge.set_cents(0.0, False)
        else:
            self.note_hero_lbl.setText("—")
            self.chord_badge_lbl.setText("")
            self.fret_detail_lbl.setText("String / Fret: —")
            self.cents_lbl.setText("0¢")
            self.tuner_gauge.set_cents(0.0, False)

        # RMS bar (-72..0 dB -> 0..100)
        rms_pct = int(max(0.0, min(100.0, ((fused.rms_db + 72.0) / 72.0) * 100.0)))
        self.rms_bar.setValue(rms_pct)

        hand_state = "OK"
        if snap.hand_result is not None and not snap.hand_result.model_ready:
            hand_state = "Missing Model"
            self.download_model_btn.setVisible(True)
        elif snap.hand_result is not None and snap.hand_result.hand_present:
            hand_state = "Hand Detected"
        else:
            hand_state = "Searching Hand"

        mode_lbl = "DEMO" if snap.demo_mode else f"{snap.video_fps:.0f} FPS"
        self.diag_lbl.setText(
            f"Neck: {snap.neck_status.value} ({snap.tracked_points} pts)  |  "
            f"{hand_state}  |  {mode_lbl}  |  Audio: {snap.audio_latency_ms:.1f} ms"
        )

    def _export_ascii_tab(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ASCII Guitar Tab", "guitar_tab.txt", "Text Files (*.txt)"
        )
        if path:
            out = self.worker.export_ascii_tab(path, self.tuning_combo.currentText())
            self.statusBar().showMessage(f"Saved ASCII tab to {out}")

    def _export_json_tab(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON Tab", "guitar_tab.json", "JSON Files (*.json)"
        )
        if path:
            out = self.worker.export_json_tab(path, self.tuning_combo.currentText())
            self.statusBar().showMessage(f"Saved JSON tab to {out}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.worker.stop()
        self.worker.wait(2500)
        if self._download_worker is not None and self._download_worker.isRunning():
            self._download_worker.wait(1000)
        super().closeEvent(event)
