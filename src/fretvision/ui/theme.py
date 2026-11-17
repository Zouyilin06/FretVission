"""Dark Studio and High-Contrast Stage theme stylesheets and semantic color palette."""

from __future__ import annotations

from fretvision.config import ThemeName

PALETTE = {
    "bg_root": "#0D1117",
    "bg_card": "#161B22",
    "bg_elevated": "#1F2631",
    "border": "#2A3241",
    "text_primary": "#F0F6FC",
    "text_secondary": "#8B949E",
    "accent_blue": "#3B82F6",
    "audio_only": "#A855F7",
    "accent_emerald": "#10B981",
    "accent_amber": "#F59E0B",
    "accent_cyan": "#06B6D4",
    "accent_red": "#EF4444",
    "fretboard_wood_top": "#1F1916",
    "fretboard_wood_bot": "#14100E",
    "fret_wire": "#94A3B8",
    "string_steel": "#CBD5E1",
}

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0D1117;
    color: #F0F6FC;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QFrame#HeaderBar {
    background-color: #121720;
    border: 1px solid #262F3F;
    border-radius: 10px;
}

QFrame#Card {
    background-color: #161B22;
    border: 1px solid #2A3241;
    border-radius: 10px;
}

QFrame#Drawer {
    background-color: #131820;
    border: 1px solid #2D3748;
    border-radius: 10px;
}

QFrame#SectionBox {
    background-color: #161B22;
    border: 1px solid #252D3B;
    border-radius: 8px;
}

QLabel#AppTitle {
    color: #F0F6FC;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.4px;
    background: transparent;
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

QLabel#StageHeroValue {
    color: #FFFFFF;
    font-size: 38px;
    font-weight: 800;
    background: transparent;
}

QLabel#SubtleLabel {
    color: #8B949E;
    font-size: 12px;
    background: transparent;
}

QLabel#StatusPillOk {
    background-color: rgba(16, 185, 129, 0.16);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.45);
    border-radius: 10px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 600;
}

QLabel#StatusPillWarn {
    background-color: rgba(245, 158, 11, 0.16);
    color: #FCD34D;
    border: 1px solid rgba(245, 158, 11, 0.45);
    border-radius: 10px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 600;
}

QLabel#StatusPillIdle {
    background-color: rgba(139, 148, 158, 0.14);
    color: #9CA3AF;
    border: 1px solid rgba(139, 148, 158, 0.35);
    border-radius: 10px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 600;
}

QPushButton {
    background-color: #1F2631;
    color: #F0F6FC;
    border: 1px solid #30394A;
    border-radius: 6px;
    padding: 5px 10px;
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

QPushButton#CompactPill {
    background-color: #1A212C;
    border: 1px solid #2E3848;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}

QPushButton#CompactPill:checked {
    background-color: #1D4ED8;
    border-color: #3B82F6;
    color: #FFFFFF;
}

QPushButton#DangerBtn {
    background-color: #7F1D1D;
    border-color: #EF4444;
}

QComboBox, QSpinBox, QLineEdit {
    background-color: #11161E;
    color: #F0F6FC;
    border: 1px solid #30394A;
    border-radius: 6px;
    padding: 3px 8px;
    min-height: 22px;
}

QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
    border-color: #4B5870;
}

QComboBox::drop-down {
    border: none;
    width: 18px;
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

QSplitter::handle {
    background-color: #1B212C;
    border-radius: 2px;
}

QSplitter::handle:hover {
    background-color: #3B82F6;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #0D1117;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #2E3747;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4B5870;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QCheckBox {
    spacing: 6px;
    background: transparent;
}
"""

HIGH_CONTRAST_STYLESHEET = DARK_STYLESHEET + """
QMainWindow, QWidget {
    background-color: #05070A;
    color: #FFFFFF;
}

QFrame#HeaderBar, QFrame#Card, QFrame#Drawer, QFrame#SectionBox {
    background-color: #0E131B;
    border: 1px solid #475569;
}

QLabel#CardTitle {
    color: #CBD5E1;
}

QLabel#SubtleLabel {
    color: #CBD5E1;
}

QLabel#HeroValue {
    color: #FFFFFF;
    font-size: 32px;
}

QPushButton {
    background-color: #1E293B;
    border: 1px solid #64748B;
    color: #FFFFFF;
}

QPushButton:checked {
    background-color: #2563EB;
    border-color: #60A5FA;
}
"""


def get_stylesheet(theme_name: ThemeName = "dark_studio") -> str:
    """Return QSS stylesheet for the active theme."""
    if theme_name == "high_contrast":
        return HIGH_CONTRAST_STYLESHEET
    return DARK_STYLESHEET
