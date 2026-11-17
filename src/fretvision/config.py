"""Application configuration dataclasses and profile persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal


Handedness = Literal["right", "left"]
NeckTrackingMode = Literal["optical_flow", "static"]
ThemeName = Literal["dark_studio", "high_contrast"]
FretboardLabelMode = Literal["notes", "degrees"]


def default_profiles_dir() -> Path:
    """Return the user config directory for saved FretVision profiles."""
    return Path.home() / ".fretvision" / "profiles"


def default_ui_state_path() -> Path:
    """Return path to persisted window/splitter UI state."""
    return Path.home() / ".fretvision" / "ui_state.json"


@dataclass
class CameraConfig:
    """Video capture source and orientation configuration."""

    source: int | str = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    mirror: bool = False
    rotation_deg: int = 0


@dataclass
class AudioConfig:
    """Audio input stream and DSP threshold settings."""

    device_index: int | None = None
    preferred_sample_rate: int = 44100
    hop_size: int = 512
    noise_gate_db: float = -48.0
    onset_flux_threshold: float = 0.25


@dataclass
class GuitarConfig:
    """Guitar tuning, capo, neck geometry, and handedness settings."""

    tuning_name: str = "Standard (EADGBE)"
    capo: int = 0
    max_fret: int = 22
    handedness: Handedness = "right"
    calibration_start_fret: int = 0
    calibration_end_fret: int = 12


@dataclass
class CalibrationProfile:
    """Saved 4-corner fretboard calibration and guitar settings."""

    name: str = "default"
    corners_px: list[list[float]] | None = None
    start_fret: int = 0
    end_fret: int = 12
    tracking_mode: NeckTrackingMode = "optical_flow"
    guitar: GuitarConfig = field(default_factory=GuitarConfig)

    def save(self, directory: Path | None = None) -> Path:
        """Persist profile to disk as JSON."""
        target_dir = directory or default_profiles_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, name: str = "default", directory: Path | None = None) -> CalibrationProfile:
        """Load a saved profile or return defaults if missing or malformed."""
        target_dir = directory or default_profiles_dir()
        path = target_dir / f"{name}.json"
        if not path.exists():
            return cls(name=name)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls(name=name)
        guitar_raw = raw.pop("guitar", {}) if isinstance(raw, dict) else {}
        valid_guitar_keys = {f.name for f in fields(GuitarConfig)}
        valid_prof_keys = {f.name for f in fields(cls)} - {"guitar"}
        guitar_cfg = GuitarConfig(
            **{k: v for k, v in guitar_raw.items() if k in valid_guitar_keys}
        )
        prof_kwargs = {k: v for k, v in raw.items() if k in valid_prof_keys}
        return cls(guitar=guitar_cfg, **prof_kwargs)


@dataclass
class UiLayoutState:
    """Persisted splitter sizes, window geometry, and UI visual preferences."""

    window_width: int = 1360
    window_height: int = 900
    v_splitter_sizes: list[int] | None = None
    h_splitter_sizes: list[int] | None = None
    theme_name: ThemeName = "dark_studio"
    fretboard_label_mode: FretboardLabelMode = "notes"
    setup_drawer_open: bool = False
    stage_mode: bool = False

    def save(self, path: Path | None = None) -> Path:
        target = path or default_ui_state_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> UiLayoutState:
        target = path or default_ui_state_path()
        if not target.exists():
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(raw, dict):
            return cls()
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in valid})


@dataclass
class AppConfig:
    """Top-level configuration bundle for FretVision."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    profile: CalibrationProfile = field(default_factory=CalibrationProfile)
    key_half_life_s: float = 6.0
    av_sync_offset_ms: float = 0.0
    tab_bpm: int = 100
    use_quantized_tab: bool = False
    time_signature: str = "4/4"
    tab_subdivision: int = 8
    theme_name: ThemeName = "dark_studio"
    fretboard_label_mode: FretboardLabelMode = "notes"
    setup_drawer_open: bool = False
    stage_mode: bool = False
