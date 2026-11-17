"""Main desktop window for FretVision (Stage 3 Studio UI)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
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
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fretvision.audio.audio_source import enumerate_audio_devices
from fretvision.config import AppConfig, CalibrationProfile, ThemeName, UiLayoutState
from fretvision.theory.music import TUNING_PRESETS, pc_to_name
from fretvision.vision.model_assets import default_model_path
from fretvision.vision.neck_tracker import NeckTrackingStatus
from fretvision.vision.video_source import enumerate_cameras

from .camera_widget import CameraViewWidget
from .fretboard_widget import VirtualFretboardWidget
from .setup_drawer import SetupDrawer
from .tab_widget import LiveTabWidget
from .telemetry_widgets import (
    ChordHistoryStrip,
    MiniSpectrumWidget,
    PrecisionTunerGauge,
)
from .theme import get_stylesheet
from .workers import EngineSnapshot, ModelDownloadWorker, ProcessingWorker


class MainWindow(QMainWindow):
    """Primary PySide6 desktop application window with collapsible setup, splitters, and Stage Mode."""

    def __init__(
        self,
        config: AppConfig | None = None,
        start_workers: bool = True,
        ui_state_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("FretVision — Live Guitar Key, Note & Tab Studio")
        self._ui_state_path = ui_state_path
        self._persist_ui_state = ui_state_path is not None or start_workers
        self.ui_state = (
            UiLayoutState.load(ui_state_path)
            if self._persist_ui_state
            else UiLayoutState()
        )
        self.resize(self.ui_state.window_width, self.ui_state.window_height)

        self.config = config or AppConfig(profile=CalibrationProfile.load("default"))
        self.config.theme_name = self.ui_state.theme_name
        self.config.fretboard_label_mode = self.ui_state.fretboard_label_mode
        self.config.setup_drawer_open = self.ui_state.setup_drawer_open
        self.config.stage_mode = self.ui_state.stage_mode

        self.worker = ProcessingWorker(self.config)
        self._download_worker: ModelDownloadWorker | None = None
        self._latest_snapshot: EngineSnapshot | None = None
        self._drawer_state_before_stage: bool = self.ui_state.setup_drawer_open

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._populate_devices()
        self._apply_theme(self.config.theme_name)
        self._restore_ui_state()

        if start_workers:
            self.worker.start()
            QTimer.singleShot(500, self._check_hand_model)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # --- Compact Studio Header Bar ---
        header_card = QFrame()
        header_card.setObjectName("HeaderBar")
        header = QHBoxLayout(header_card)
        header.setContentsMargins(12, 7, 12, 7)
        header.setSpacing(8)

        brand = QLabel("FretVision")
        brand.setObjectName("AppTitle")
        header.addWidget(brand)

        self.neck_pill = QLabel("● Neck: Uncalibrated")
        self.neck_pill.setObjectName("StatusPillIdle")
        header.addWidget(self.neck_pill)

        self.hand_pill = QLabel("● Hand")
        self.hand_pill.setObjectName("StatusPillIdle")
        header.addWidget(self.hand_pill)

        self.perf_pill = QLabel("0 FPS")
        self.perf_pill.setObjectName("StatusPillIdle")
        header.addWidget(self.perf_pill)

        header.addStretch(1)

        header.addWidget(QLabel("Tuning:"))
        self.tuning_combo = QComboBox()
        self.tuning_combo.addItems(list(TUNING_PRESETS.keys()))
        idx_t = self.tuning_combo.findText(self.config.profile.guitar.tuning_name)
        if idx_t >= 0:
            self.tuning_combo.setCurrentIndex(idx_t)
        header.addWidget(self.tuning_combo)

        header.addWidget(QLabel("Capo:"))
        self.capo_spin = QSpinBox()
        self.capo_spin.setRange(0, 12)
        self.capo_spin.setValue(self.config.profile.guitar.capo)
        header.addWidget(self.capo_spin)

        self.reset_key_btn = QPushButton("Reset Key")
        header.addWidget(self.reset_key_btn)

        self.open_clip_btn = QPushButton("Open Clip…")
        header.addWidget(self.open_clip_btn)

        self.demo_btn = QPushButton("Demo Mode")
        self.demo_btn.setCheckable(True)
        header.addWidget(self.demo_btn)

        self.stage_mode_btn = QPushButton("Stage Mode")
        self.stage_mode_btn.setCheckable(True)
        self.stage_mode_btn.setToolTip("Hide setup/camera panels and enlarge musical telemetry (F)")
        header.addWidget(self.stage_mode_btn)

        self.setup_drawer_btn = QPushButton("Setup ⚙")
        self.setup_drawer_btn.setCheckable(True)
        self.setup_drawer_btn.setObjectName("PrimaryBtn")
        header.addWidget(self.setup_drawer_btn)

        root.addWidget(header_card)

        # --- Body Row: Resizable Studio Splitters + Collapsible Setup Drawer ---
        body_row = QHBoxLayout()
        body_row.setContentsMargins(0, 0, 0, 0)
        body_row.setSpacing(8)
        root.addLayout(body_row, 1)

        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_splitter.setChildrenCollapsible(False)
        body_row.addWidget(self.v_splitter, 1)

        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.h_splitter.setChildrenCollapsible(False)
        self.v_splitter.addWidget(self.h_splitter)

        # Left Panel: AR Camera Card + Contextual Calibration Bar
        self.camera_card = QFrame()
        self.camera_card.setObjectName("Card")
        cam_lay = QVBoxLayout(self.camera_card)
        cam_lay.setContentsMargins(8, 8, 8, 8)
        cam_lay.setSpacing(6)

        cam_bar = QHBoxLayout()
        cam_bar.setSpacing(6)
        cam_title = QLabel("LIVE AR FRETBOARD")
        cam_title.setObjectName("CardTitle")
        cam_bar.addWidget(cam_title)

        self.provider_lbl = QLabel("CV")
        self.provider_lbl.setObjectName("StatusPillIdle")
        cam_bar.addWidget(self.provider_lbl)
        cam_bar.addStretch(1)

        self.calibrate_btn = QPushButton("Calibrate (4-Click)")
        self.calibrate_btn.setObjectName("PrimaryBtn")
        cam_bar.addWidget(self.calibrate_btn)

        self.adjust_corners_btn = QPushButton("Adjust Corners")
        self.adjust_corners_btn.setCheckable(True)
        cam_bar.addWidget(self.adjust_corners_btn)

        self.auto_guess_btn = QPushButton("Auto-Guess")
        cam_bar.addWidget(self.auto_guess_btn)

        self.snap_anchor_btn = QPushButton("Re-Snap")
        cam_bar.addWidget(self.snap_anchor_btn)
        cam_lay.addLayout(cam_bar)

        self.camera_widget = CameraViewWidget()
        cam_lay.addWidget(self.camera_widget, 1)

        # Clip replay transport bar (hidden unless an offline .mp4/.wav clip is open)
        self.clip_bar = QWidget()
        clip_lay = QHBoxLayout(self.clip_bar)
        clip_lay.setContentsMargins(0, 2, 0, 0)
        clip_lay.setSpacing(6)
        self.clip_play_btn = QPushButton("Pause")
        self.clip_play_btn.setCheckable(True)
        self.clip_scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_scrub_slider.setRange(0, 1000)
        self.clip_time_lbl = QLabel("00:00 / 00:00")
        self.clip_exit_btn = QPushButton("Exit Clip")
        clip_lay.addWidget(self.clip_play_btn)
        clip_lay.addWidget(self.clip_scrub_slider, 1)
        clip_lay.addWidget(self.clip_time_lbl)
        clip_lay.addWidget(self.clip_exit_btn)
        self.clip_bar.setVisible(False)
        cam_lay.addWidget(self.clip_bar)
        self.h_splitter.addWidget(self.camera_card)

        # Right Panel: Musician Telemetry Sidebar
        self.telemetry_panel = QWidget()
        self.telemetry_panel.setStyleSheet("background: transparent;")
        side_col = QVBoxLayout(self.telemetry_panel)
        side_col.setContentsMargins(0, 0, 0, 0)
        side_col.setSpacing(8)

        # 1. Key, Scale & Chord Progression Card
        key_card = QFrame()
        key_card.setObjectName("Card")
        kl = QVBoxLayout(key_card)
        kl.setContentsMargins(14, 10, 14, 10)
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

        prog_hdr = QLabel("CHORD PROGRESSION")
        prog_hdr.setObjectName("CardTitle")
        kl.addSpacing(4)
        kl.addWidget(prog_hdr)
        self.chord_history = ChordHistoryStrip()
        kl.addWidget(self.chord_history)
        side_col.addWidget(key_card)

        # 2. Pressed Note / Chord & Precision Tuner Card
        note_card = QFrame()
        note_card.setObjectName("Card")
        nl = QVBoxLayout(note_card)
        nl.setContentsMargins(14, 10, 14, 10)
        nl.setSpacing(5)
        n_title = QLabel("PRESSED NOTE, CHORD & TUNER")
        n_title.setObjectName("CardTitle")
        nl.addWidget(n_title)

        chord_note_row = QHBoxLayout()
        self.note_hero_lbl = QLabel("—")
        self.note_hero_lbl.setObjectName("HeroValue")
        chord_note_row.addWidget(self.note_hero_lbl, 1)

        self.chord_badge_lbl = QLabel("")
        self.chord_badge_lbl.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
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

        self.tuner_gauge = PrecisionTunerGauge()
        nl.addWidget(self.tuner_gauge)
        side_col.addWidget(note_card)

        # 3. Audio Spectrum & Signal Diagnostics Card
        diag_card = QFrame()
        diag_card.setObjectName("Card")
        dl = QVBoxLayout(diag_card)
        dl.setContentsMargins(14, 10, 14, 10)
        dl.setSpacing(6)
        d_title = QLabel("AUDIO SPECTRUM & SIGNAL")
        d_title.setObjectName("CardTitle")
        dl.addWidget(d_title)

        self.spectrum_widget = MiniSpectrumWidget()
        dl.addWidget(self.spectrum_widget)

        rms_row = QHBoxLayout()
        rms_row.addWidget(QLabel("Level:"))
        self.rms_bar = QProgressBar()
        self.rms_bar.setRange(0, 100)
        self.rms_bar.setValue(0)
        self.rms_bar.setTextVisible(False)
        rms_row.addWidget(self.rms_bar, 1)
        dl.addLayout(rms_row)

        self.diag_lbl = QLabel("Neck: Uncalibrated  |  Searching Hand")
        self.diag_lbl.setObjectName("SubtleLabel")
        self.diag_lbl.setWordWrap(True)
        dl.addWidget(self.diag_lbl)

        side_col.addWidget(diag_card)
        side_col.addStretch(1)
        self.h_splitter.addWidget(self.telemetry_panel)
        self.h_splitter.setStretchFactor(0, 6)
        self.h_splitter.setStretchFactor(1, 4)

        # Bottom Virtual Fretboard & Live Tablature inside Vertical Splitter
        self.fb_card = QFrame()
        self.fb_card.setObjectName("Card")
        fb_lay = QVBoxLayout(self.fb_card)
        fb_lay.setContentsMargins(4, 2, 4, 4)
        self.fretboard_widget = VirtualFretboardWidget()
        self.fretboard_widget.set_guitar_setup(
            self.config.profile.guitar.tuning_name,
            self.config.profile.guitar.capo,
            self.config.profile.guitar.handedness,
            self.config.profile.guitar.max_fret,
        )
        self.fretboard_widget.set_label_mode(self.config.fretboard_label_mode)
        fb_lay.addWidget(self.fretboard_widget)
        self.v_splitter.addWidget(self.fb_card)

        self.tab_card = QFrame()
        self.tab_card.setObjectName("Card")
        tab_lay = QVBoxLayout(self.tab_card)
        tab_lay.setContentsMargins(2, 2, 2, 2)
        self.tab_widget = LiveTabWidget()
        tab_lay.addWidget(self.tab_widget)
        self.v_splitter.addWidget(self.tab_card)

        self.v_splitter.setStretchFactor(0, 5)
        self.v_splitter.setStretchFactor(1, 2)
        self.v_splitter.setStretchFactor(2, 2)

        # Collapsible Setup Drawer on right edge
        self.setup_drawer = SetupDrawer(self.config)
        body_row.addWidget(self.setup_drawer)

        # Alias drawer widgets directly onto MainWindow for clean access & compatibility
        self.camera_combo = self.setup_drawer.camera_combo
        self.phone_url_btn = self.setup_drawer.phone_url_btn
        self.mirror_chk = self.setup_drawer.mirror_chk
        self.rotate_combo = self.setup_drawer.rotate_combo
        self.av_sync_spin = self.setup_drawer.av_sync_spin
        self.audio_combo = self.setup_drawer.audio_combo
        self.refresh_dev_btn = self.setup_drawer.refresh_dev_btn
        self.gate_slider = self.setup_drawer.gate_slider
        self.gate_val_lbl = self.setup_drawer.gate_val_lbl
        self.download_model_btn = self.setup_drawer.download_model_btn
        self.start_fret_spin = self.setup_drawer.start_fret_spin
        self.end_fret_spin = self.setup_drawer.end_fret_spin
        self.handed_combo = self.setup_drawer.handed_combo
        self.lock_flow_chk = self.setup_drawer.lock_flow_chk
        self.scale_overlay_chk = self.setup_drawer.scale_overlay_chk
        self.save_sample_btn = self.setup_drawer.save_sample_btn
        self.theme_combo = self.setup_drawer.theme_combo

        self.statusBar().showMessage("Ready")

    def _connect_signals(self) -> None:
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.status_message.connect(self.statusBar().showMessage)

        self.setup_drawer_btn.toggled.connect(self._on_drawer_toggled)
        self.setup_drawer.close_btn.clicked.connect(lambda: self.setup_drawer_btn.setChecked(False))
        self.stage_mode_btn.toggled.connect(self._on_stage_mode_toggled)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)

        self.camera_combo.activated.connect(self._on_camera_selected)
        self.phone_url_btn.clicked.connect(self._on_phone_url_clicked)
        self.open_clip_btn.clicked.connect(self._on_open_clip)
        self.clip_play_btn.clicked.connect(self._on_clip_play_toggled)
        self.clip_scrub_slider.sliderMoved.connect(self._on_clip_scrub)
        self.clip_exit_btn.clicked.connect(self._on_clip_exit)
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
        self.save_sample_btn.clicked.connect(self._on_save_sample)
        self.av_sync_spin.valueChanged.connect(lambda val: self.worker.set_av_sync_offset(float(val)))
        self.reset_key_btn.clicked.connect(self._on_reset_key)
        self.scale_overlay_chk.toggled.connect(self.fretboard_widget.set_show_scale_overlay)
        self.fretboard_widget.label_mode_changed.connect(self._on_fretboard_mode_changed)
        self.demo_btn.toggled.connect(self._on_demo_toggled)

        self.gate_slider.valueChanged.connect(self._on_gate_changed)
        self.download_model_btn.clicked.connect(self._start_model_download)

        self.tab_widget.bpm_spin.setValue(self.config.tab_bpm)
        idx_ts = self.tab_widget.timesig_combo.findText(self.config.time_signature)
        if idx_ts >= 0:
            self.tab_widget.timesig_combo.setCurrentIndex(idx_ts)
        self.tab_widget.quantize_chk.setChecked(self.config.use_quantized_tab)

        self.tab_widget.bpm_spin.valueChanged.connect(self._on_tab_timing_changed)
        self.tab_widget.timesig_combo.currentTextChanged.connect(self._on_tab_timing_changed)
        self.tab_widget.quantize_chk.toggled.connect(self._on_tab_timing_changed)
        self.tab_widget.metronome_btn.toggled.connect(self._on_metronome_toggled)

        self.tab_widget.pause_btn.toggled.connect(self.worker.set_tab_paused)
        self.tab_widget.clear_btn.clicked.connect(self.worker.clear_tab)
        self.tab_widget.export_txt_btn.clicked.connect(self._export_ascii_tab)
        self.tab_widget.export_midi_btn.clicked.connect(self._export_midi_tab)
        self.tab_widget.export_xml_btn.clicked.connect(self._export_musicxml_tab)
        self.tab_widget.export_json_btn.clicked.connect(self._export_json_tab)

    def _setup_shortcuts(self) -> None:
        self._shortcuts: list[QShortcut] = []
        bindings = (
            ("Space", self._on_space_shortcut),
            ("F", self.stage_mode_btn.toggle),
            ("C", self._on_start_calibration),
            ("R", self._on_reset_key),
            ("M", self.tab_widget.metronome_btn.toggle),
            ("D", self.demo_btn.toggle),
            ("Escape", self._on_escape_shortcut),
        )
        for key, handler in bindings:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

    def _restore_ui_state(self) -> None:
        idx_th = self.theme_combo.findData(self.ui_state.theme_name)
        if idx_th >= 0:
            self.theme_combo.setCurrentIndex(idx_th)

        self.setup_drawer_btn.blockSignals(True)
        self.setup_drawer_btn.setChecked(self.ui_state.setup_drawer_open)
        self.setup_drawer_btn.blockSignals(False)
        self.setup_drawer.setVisible(self.ui_state.setup_drawer_open)

        if self.ui_state.h_splitter_sizes:
            self.h_splitter.setSizes(self.ui_state.h_splitter_sizes)
        if self.ui_state.v_splitter_sizes:
            self.v_splitter.setSizes(self.ui_state.v_splitter_sizes)

        if self.ui_state.stage_mode:
            self.stage_mode_btn.setChecked(True)

    def _apply_theme(self, theme_name: ThemeName) -> None:
        self.config.theme_name = theme_name
        self.setStyleSheet(get_stylesheet(theme_name))

    def _on_theme_changed(self) -> None:
        data = self.theme_combo.currentData()
        if data in ("dark_studio", "high_contrast"):
            self._apply_theme(data)

    def _on_drawer_toggled(self, open_flag: bool) -> None:
        if self.stage_mode_btn.isChecked() and open_flag:
            self.stage_mode_btn.setChecked(False)
        self.setup_drawer.setVisible(open_flag)
        self.config.setup_drawer_open = open_flag

    def _on_stage_mode_toggled(self, enabled: bool) -> None:
        self.config.stage_mode = enabled
        if enabled:
            self._drawer_state_before_stage = self.setup_drawer_btn.isChecked()
            self.setup_drawer.setVisible(False)
            self.camera_card.setVisible(False)
            self.key_hero_lbl.setObjectName("StageHeroValue")
            self.note_hero_lbl.setObjectName("StageHeroValue")
        else:
            self.camera_card.setVisible(True)
            self.setup_drawer.setVisible(self.setup_drawer_btn.isChecked())
            self.key_hero_lbl.setObjectName("HeroValue")
            self.note_hero_lbl.setObjectName("HeroValue")

        self.fretboard_widget.set_stage_mode(enabled)
        # Re-polish labels so QSS objectName change updates typography immediately
        for lbl in (self.key_hero_lbl, self.note_hero_lbl):
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)
            lbl.update()

    def _on_fretboard_mode_changed(self, mode: str) -> None:
        if mode in ("notes", "degrees"):
            self.config.fretboard_label_mode = mode  # type: ignore[assignment]

    def _on_space_shortcut(self) -> None:
        if self._latest_snapshot is not None and self._latest_snapshot.clip_active:
            self._on_clip_play_toggled()
        else:
            self.tab_widget.pause_btn.toggle()

    def _on_escape_shortcut(self) -> None:
        if self.camera_widget._calibrating:
            self.camera_widget.cancel_calibration()
        elif self.adjust_corners_btn.isChecked():
            self.adjust_corners_btn.setChecked(False)
        elif self.stage_mode_btn.isChecked():
            self.stage_mode_btn.setChecked(False)

    def _on_reset_key(self) -> None:
        self.worker.reset_key()
        self.chord_history.clear()

    def _on_demo_toggled(self, checked: bool) -> None:
        if checked:
            self.chord_history.clear()
        self.worker.set_demo_mode(checked)

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
                "MediaPipe hand_landmarker.task not found — open Setup ⚙ to download."
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
        if self.stage_mode_btn.isChecked():
            self.stage_mode_btn.setChecked(False)
        self.adjust_corners_btn.setChecked(False)
        self.camera_widget.start_calibration(s_fret, e_fret)
        self.statusBar().showMessage("Click 4 fretboard corners on the camera view (Right-click or Esc cancels).")

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
                "Auto-Guess couldn't isolate fretboard — use 4-Click Calibrate."
            )

    def _on_save_sample(self) -> None:
        out = self.worker.save_training_sample()
        if out is None:
            self.statusBar().showMessage("Calibrate on a live/loaded frame before saving a training sample.")

    def _on_gate_changed(self, value: int) -> None:
        self.gate_val_lbl.setText(f"{value} dB")
        self.worker.set_noise_gate(float(value))

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 6000)

    @staticmethod
    def _set_pill_state(pill: QLabel, text: str, state: str) -> None:
        target_name = {
            "ok": "StatusPillOk",
            "warn": "StatusPillWarn",
            "idle": "StatusPillIdle",
        }.get(state, "StatusPillIdle")
        pill.setText(text)
        if pill.objectName() != target_name:
            pill.setObjectName(target_name)
            pill.style().unpolish(pill)
            pill.style().polish(pill)

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
        self.spectrum_widget.set_bands(snap.spectrum_bands)
        self.provider_lbl.setText(snap.neck_provider)

        # Update Key Card & Chord Progression Strip
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

        if fused.chord is not None:
            self.chord_history.ingest_chord(fused.chord.symbol, is_onset=fused.is_onset)

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
            self.tuner_gauge.set_state(cents, True, names)
        elif fused.hover_notes:
            self.note_hero_lbl.setText("Hovering")
            self.chord_badge_lbl.setText("")
            hover_txt = "  |  ".join(
                f"Str {n.string_number} · Fret {n.fret} ({n.note_name})"
                for n in fused.hover_notes
            )
            self.fret_detail_lbl.setText(f"Finger position: {hover_txt}")
            self.cents_lbl.setText("0¢")
            self.tuner_gauge.set_state(0.0, False)
        else:
            self.note_hero_lbl.setText("—")
            self.chord_badge_lbl.setText("")
            self.fret_detail_lbl.setText("String / Fret: —")
            self.cents_lbl.setText("0¢")
            self.tuner_gauge.set_state(0.0, False)

        # RMS bar & header status pills
        rms_pct = int(max(0.0, min(100.0, ((fused.rms_db + 72.0) / 72.0) * 100.0)))
        self.rms_bar.setValue(rms_pct)

        if snap.neck_status in (NeckTrackingStatus.LOCKED, NeckTrackingStatus.STATIC):
            self._set_pill_state(self.neck_pill, f"● Neck: {snap.neck_status.value}", "ok")
        elif snap.neck_status == NeckTrackingStatus.DRIFTING:
            self._set_pill_state(self.neck_pill, "● Neck: Drifting", "warn")
        else:
            self._set_pill_state(self.neck_pill, "● Neck: Uncalibrated", "idle")

        hand_state = "Searching Hand"
        if snap.hand_result is not None and not snap.hand_result.model_ready:
            hand_state = "Missing Model"
            self.download_model_btn.setVisible(True)
            self._set_pill_state(self.hand_pill, "● Model Missing", "warn")
        elif snap.hand_result is not None and snap.hand_result.hand_present:
            hand_state = "Hand Detected"
            self._set_pill_state(self.hand_pill, "● Hand OK", "ok")
        else:
            self._set_pill_state(self.hand_pill, "● Searching Hand", "idle")

        mode_lbl = "CLIP" if snap.clip_active else ("DEMO" if snap.demo_mode else f"{snap.video_fps:.0f} FPS")
        self._set_pill_state(self.perf_pill, f"{mode_lbl} · {snap.audio_latency_ms:.1f} ms", "ok" if snap.demo_mode or snap.video_fps > 1 else "idle")
        self.diag_lbl.setText(
            f"Neck: {snap.neck_status.value} ({snap.tracked_points} pts)  |  {hand_state}"
        )

        self.clip_bar.setVisible(snap.clip_active)
        if snap.clip_active:
            self.clip_play_btn.setText("Pause" if snap.clip_playing else "Play")
            if not self.clip_scrub_slider.isSliderDown() and snap.clip_duration_s > 0.0:
                pos_val = int(round((snap.clip_position_s / snap.clip_duration_s) * 1000.0))
                self.clip_scrub_slider.setValue(max(0, min(1000, pos_val)))
            cur_m, cur_s = divmod(int(snap.clip_position_s), 60)
            tot_m, tot_s = divmod(int(snap.clip_duration_s), 60)
            self.clip_time_lbl.setText(f"{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}")

    def _on_tab_timing_changed(self) -> None:
        self.worker.set_tab_timing(
            self.tab_widget.quantize_chk.isChecked(),
            self.tab_widget.bpm_spin.value(),
            self.tab_widget.timesig_combo.currentText(),
        )

    def _on_metronome_toggled(self, checked: bool) -> None:
        self.worker.set_metronome(
            checked,
            self.tab_widget.bpm_spin.value(),
            self.tab_widget.timesig_combo.currentText(),
        )

    def _on_open_clip(self) -> None:
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Recorded Guitar Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        if not video_path:
            return
        audio_path: str | None = None
        if not Path(video_path).with_suffix(".wav").exists():
            wav_choice, _ = QFileDialog.getOpenFileName(
                self,
                "Optional Paired Audio (.wav) — Cancel to use video audio/ffmpeg",
                str(Path(video_path).parent),
                "WAV Audio (*.wav)",
            )
            if wav_choice:
                audio_path = wav_choice
        self.demo_btn.setChecked(False)
        self.chord_history.clear()
        self.worker.load_clip(video_path, audio_path)

    def _on_clip_play_toggled(self) -> None:
        if self._latest_snapshot is None or not self._latest_snapshot.clip_active:
            return
        self.worker.set_clip_playing(not self._latest_snapshot.clip_playing)

    def _on_clip_scrub(self, slider_val: int) -> None:
        if self._latest_snapshot is None or not self._latest_snapshot.clip_active:
            return
        target_s = (slider_val / 1000.0) * self._latest_snapshot.clip_duration_s
        self.worker.seek_clip(target_s)

    def _on_clip_exit(self) -> None:
        self.worker.close_clip()
        self.clip_bar.setVisible(False)

    def _export_ascii_tab(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ASCII Guitar Tab", "guitar_tab.txt", "Text Files (*.txt)"
        )
        if path:
            out = self.worker.export_ascii_tab(path, self.tuning_combo.currentText())
            self.statusBar().showMessage(f"Saved ASCII tab to {out}")

    def _export_midi_tab(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDI File", "guitar_tab.mid", "MIDI Files (*.mid)"
        )
        if path:
            out = self.worker.export_midi_tab(
                path, self.tuning_combo.currentText(), self.tab_widget.bpm_spin.value()
            )
            self.statusBar().showMessage(f"Saved MIDI file to {out}")

    def _export_musicxml_tab(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MusicXML", "guitar_tab.musicxml", "MusicXML Files (*.musicxml *.xml)"
        )
        if path:
            out = self.worker.export_musicxml_tab(
                path,
                self.tuning_combo.currentText(),
                self.tab_widget.bpm_spin.value(),
                self.tab_widget.timesig_combo.currentText(),
            )
            self.statusBar().showMessage(f"Saved MusicXML to {out}")

    def _export_json_tab(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON Tab", "guitar_tab.json", "JSON Files (*.json)"
        )
        if path:
            out = self.worker.export_json_tab(path, self.tuning_combo.currentText())
            self.statusBar().showMessage(f"Saved JSON tab to {out}")

    def save_ui_state(self) -> None:
        state = UiLayoutState(
            window_width=self.width(),
            window_height=self.height(),
            v_splitter_sizes=self.v_splitter.sizes(),
            h_splitter_sizes=self.h_splitter.sizes(),
            theme_name=self.config.theme_name,
            fretboard_label_mode=self.config.fretboard_label_mode,
            setup_drawer_open=self.setup_drawer_btn.isChecked(),
            stage_mode=self.stage_mode_btn.isChecked(),
        )
        state.save(self._ui_state_path)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._persist_ui_state:
            self.save_ui_state()
        self.worker.stop()
        self.worker.wait(2500)
        if self._download_worker is not None and self._download_worker.isRunning():
            self._download_worker.wait(1000)
        super().closeEvent(event)
