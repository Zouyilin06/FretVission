"""Audio-visual sensor fusion and live guitar tablature recording."""

from .av_fusion import (
    ActiveNote,
    AudioVisualFusion,
    FusedFrameResult,
    build_visual_pitch_priors,
)
from .tab_recorder import TabColumn, TabRecorder

__all__ = [
    "ActiveNote",
    "AudioVisualFusion",
    "FusedFrameResult",
    "TabColumn",
    "TabRecorder",
    "build_visual_pitch_priors",
]
