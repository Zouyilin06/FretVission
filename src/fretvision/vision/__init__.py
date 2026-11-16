"""Computer vision pipeline: camera streams, fretboard geometry, neck tracking, and hand tracking."""

from .fretboard_model import (
    FretboardModel,
    NeckMapping,
    fret_to_canonical_u,
    string_to_canonical_v,
    u_to_continuous_fret,
)
from .hand_tracker import FingerObservation, HandTracker, HandTrackingResult
from .model_assets import default_model_path, ensure_hand_landmarker_model
from .neck_tracker import NeckTracker, NeckTrackingStatus, auto_guess_neck_corners
from .video_source import CameraDevice, VideoSource, enumerate_cameras

__all__ = [
    "CameraDevice",
    "FingerObservation",
    "FretboardModel",
    "HandTracker",
    "HandTrackingResult",
    "NeckMapping",
    "NeckTracker",
    "NeckTrackingStatus",
    "VideoSource",
    "auto_guess_neck_corners",
    "default_model_path",
    "ensure_hand_landmarker_model",
    "enumerate_cameras",
    "fret_to_canonical_u",
    "string_to_canonical_v",
    "u_to_continuous_fret",
]
