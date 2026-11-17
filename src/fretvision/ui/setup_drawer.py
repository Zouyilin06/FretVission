"""Collapsible studio setup drawer organizing hardware, calibration, and appearance controls."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fretvision.config import AppConfig


class SetupDrawer(QFrame):
    """Collapsible right-side settings panel for video, audio, calibration, and theme."""

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Drawer")
        self.setFixedWidth(332)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        hdr = QHBoxLayout()
        title = QLabel("STUDIO SETUP & CALIBRATION")
        title.setObjectName("CardTitle")
        hdr.addWidget(title)
        hdr.addStretch(1)
        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("CompactPill")
        self.close_btn.setFixedWidth(28)
        hdr.addWidget(self.close_btn)
        outer.addLayout(hdr)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 4, 0)
        body_lay.setSpacing(10)
        scroll.setWidget(body)

        def _compact_combo(cb: QComboBox, min_chars: int = 8) -> QComboBox:
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(min_chars)
            cb.setMinimumWidth(0)
            cb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return cb

        # --- Section 1: Video & Phone Stream ---
        vid_box = QFrame()
        vid_box.setObjectName("SectionBox")
        vl = QVBoxLayout(vid_box)
        vl.setContentsMargins(10, 8, 10, 10)
        vl.setSpacing(6)
        v_hdr = QLabel("VIDEO & PHONE STREAM")
        v_hdr.setObjectName("CardTitle")
        vl.addWidget(v_hdr)

        cam_row = QHBoxLayout()
        self.camera_combo = _compact_combo(QComboBox())
        self.phone_url_btn = QPushButton("Phone URL…")
        cam_row.addWidget(self.camera_combo, 1)
        cam_row.addWidget(self.phone_url_btn)
        vl.addLayout(cam_row)

        ori_row = QHBoxLayout()
        self.mirror_chk = QCheckBox("Mirror")
        self.mirror_chk.setChecked(config.camera.mirror)
        ori_row.addWidget(self.mirror_chk)
        ori_row.addStretch(1)
        ori_row.addWidget(QLabel("Rotate:"))
        self.rotate_combo = _compact_combo(QComboBox(), min_chars=4)
        self.rotate_combo.setMinimumWidth(72)
        self.rotate_combo.addItems(["0°", "90°", "180°", "270°"])
        ori_row.addWidget(self.rotate_combo)
        vl.addLayout(ori_row)

        sync_row = QHBoxLayout()
        sync_row.addWidget(QLabel("AV Sync Offset:"))
        sync_row.addStretch(1)
        self.av_sync_spin = QSpinBox()
        self.av_sync_spin.setRange(-300, 300)
        self.av_sync_spin.setSingleStep(15)
        self.av_sync_spin.setSuffix(" ms")
        self.av_sync_spin.setValue(int(config.av_sync_offset_ms))
        self.av_sync_spin.setToolTip("Positive if phone/camera video lags behind microphone audio")
        sync_row.addWidget(self.av_sync_spin)
        vl.addLayout(sync_row)
        body_lay.addWidget(vid_box)

        # --- Section 2: Audio & Hand Model ---
        aud_box = QFrame()
        aud_box.setObjectName("SectionBox")
        al = QVBoxLayout(aud_box)
        al.setContentsMargins(10, 8, 10, 10)
        al.setSpacing(6)
        a_hdr = QLabel("AUDIO INPUT & MODEL")
        a_hdr.setObjectName("CardTitle")
        al.addWidget(a_hdr)

        aud_row = QHBoxLayout()
        self.audio_combo = _compact_combo(QComboBox())
        self.refresh_dev_btn = QPushButton("Refresh")
        aud_row.addWidget(self.audio_combo, 1)
        aud_row.addWidget(self.refresh_dev_btn)
        al.addLayout(aud_row)

        gate_row = QHBoxLayout()
        gate_row.addWidget(QLabel("Noise Gate:"))
        self.gate_slider = QSlider(Qt.Orientation.Horizontal)
        self.gate_slider.setRange(-72, -20)
        self.gate_slider.setValue(int(config.audio.noise_gate_db))
        self.gate_val_lbl = QLabel(f"{int(config.audio.noise_gate_db)} dB")
        self.gate_val_lbl.setMinimumWidth(46)
        self.gate_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        gate_row.addWidget(self.gate_slider, 1)
        gate_row.addWidget(self.gate_val_lbl)
        al.addLayout(gate_row)

        self.download_model_btn = QPushButton("Download Hand Tracking Model (~8 MB)")
        self.download_model_btn.setVisible(False)
        al.addWidget(self.download_model_btn)
        body_lay.addWidget(aud_box)

        # --- Section 3: Guitar & Calibration ---
        cal_box = QFrame()
        cal_box.setObjectName("SectionBox")
        cl = QVBoxLayout(cal_box)
        cl.setContentsMargins(10, 8, 10, 10)
        cl.setSpacing(6)
        c_hdr = QLabel("GUITAR & NECK CALIBRATION")
        c_hdr.setObjectName("CardTitle")
        cl.addWidget(c_hdr)

        fret_row = QHBoxLayout()
        fret_row.addWidget(QLabel("Frets:"))
        self.start_fret_spin = QSpinBox()
        self.start_fret_spin.setRange(0, 15)
        self.start_fret_spin.setValue(config.profile.start_fret)
        self.end_fret_spin = QSpinBox()
        self.end_fret_spin.setRange(3, 24)
        self.end_fret_spin.setValue(config.profile.end_fret)
        fret_row.addWidget(self.start_fret_spin)
        fret_row.addWidget(QLabel("to"))
        fret_row.addWidget(self.end_fret_spin)
        fret_row.addStretch(1)
        cl.addLayout(fret_row)

        hand_row = QHBoxLayout()
        hand_row.addWidget(QLabel("Orientation:"))
        self.handed_combo = _compact_combo(QComboBox())
        self.handed_combo.addItems(["Right-Handed", "Left-Handed"])
        if config.profile.guitar.handedness == "left":
            self.handed_combo.setCurrentIndex(1)
        hand_row.addWidget(self.handed_combo, 1)
        cl.addLayout(hand_row)

        self.lock_flow_chk = QCheckBox("Optical Flow Lock")
        self.lock_flow_chk.setChecked(config.profile.tracking_mode == "optical_flow")
        cl.addWidget(self.lock_flow_chk)

        self.scale_overlay_chk = QCheckBox("Key Scale Fretboard Overlay")
        self.scale_overlay_chk.setChecked(True)
        cl.addWidget(self.scale_overlay_chk)

        self.save_sample_btn = QPushButton("Save YOLO-Pose Sample")
        self.save_sample_btn.setToolTip(
            "Save current calibrated frame & 4 corners for RTX 4060 training"
        )
        cl.addWidget(self.save_sample_btn)
        body_lay.addWidget(cal_box)

        # --- Section 4: Appearance & Shortcuts ---
        app_box = QFrame()
        app_box.setObjectName("SectionBox")
        apl = QVBoxLayout(app_box)
        apl.setContentsMargins(10, 8, 10, 10)
        apl.setSpacing(6)
        ap_hdr = QLabel("APPEARANCE & SHORTCUTS")
        ap_hdr.setObjectName("CardTitle")
        apl.addWidget(ap_hdr)

        th_row = QHBoxLayout()
        th_row.addWidget(QLabel("Theme:"))
        self.theme_combo = _compact_combo(QComboBox())
        self.theme_combo.addItem("Dark Studio", "dark_studio")
        self.theme_combo.addItem("High-Contrast Stage", "high_contrast")
        th_row.addWidget(self.theme_combo, 1)
        apl.addLayout(th_row)

        hints = QLabel(
            "Shortcuts:\n"
            "Space: Pause Tab/Clip  •  F: Stage Mode\n"
            "C: Calibrate  •  M: Metronome\n"
            "R: Reset Key  •  D: Demo  •  Esc: Cancel"
        )
        hints.setObjectName("SubtleLabel")
        hints.setWordWrap(True)
        hints.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        apl.addWidget(hints)
        body_lay.addWidget(app_box)

        body_lay.addStretch(1)
