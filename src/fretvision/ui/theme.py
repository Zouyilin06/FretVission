"""Dark studio theme stylesheet and semantic color palette for FretVision."""

from __future__ import annotations

PALETTE = {
    "bg_root": "#0D1117",
    "bg_card": "#161B22",
    "bg_elevated": "#1F2631",
    "border": "#2A3241",
    "text_primary": "#F0F6FC",
    "text_secondary": "#8B949E",
    "accent_blue": "#3B82F6",
    "accent_emerald": "#10B981",
    "accent_amber": "#F59E0B",
    "accent_cyan": "#06B6D4",
    "accent_red": "#EF4444",
    "fretboard_wood": "#1A1614",
    "fret_wire": "#8C96A5",
    "string_steel": "#C9D1D9",
}

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0D1117;
    color: #F0F6FC;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QFrame#Card {
    background-color: #161B22;
    border: 1px solid #2A3241;
    border-radius: 10px;
}

QLabel#CardTitle {
    color: #8B949E;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    background: transparent;
}

QLabel#HeroValue {
    color: #F0F6FC;
    font-size: 28px;
    font-weight: 700;
    background: transparent;
}

QLabel#SubtleLabel {
    color: #8B949E;
    font-size: 12px;
    background: transparent;
}

QPushButton {
    background-color: #1F2631;
    color: #F0F6FC;
    border: 1px solid #30394A;
    border-radius: 6px;
    padding: 5px 11px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #283140;
    border-color: #4B5870;
}

QPushButton:pressed {
    background-color: #171D26;
}

QPushButton:checked {
    background-color: #1D4ED8;
    border-color: #3B82F6;
    color: #FFFFFF;
}

QPushButton#PrimaryBtn {
    background-color: #2563EB;
    border-color: #3B82F6;
    color: #FFFFFF;
}

QPushButton#PrimaryBtn:hover {
    background-color: #1D4ED8;
}

QPushButton#DangerBtn {
    background-color: #7F1D1D;
    border-color: #EF4444;
}

QComboBox, QSpinBox, QLineEdit {
    background-color: #161B22;
    color: #F0F6FC;
    border: 1px solid #30394A;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
}

QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
    border-color: #4B5870;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QProgressBar {
    background-color: #0D1117;
    border: 1px solid #2A3241;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #10B981;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    height: 5px;
    background: #2A3241;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #3B82F6;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QCheckBox {
    spacing: 6px;
    background: transparent;
}
"""
