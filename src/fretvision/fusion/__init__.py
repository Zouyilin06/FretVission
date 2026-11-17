"""Audio-visual sensor fusion and live guitar tablature recording."""

from .av_fusion import (
    ActiveNote,
    AudioVisualFusion,
    FusedFrameResult,
    build_visual_pitch_priors,
)
from .clip_replay import ClipReplaySource
from .tab_recorder import TabColumn, TabRecorder

__all__ = [
    "ActiveNote",
    "AudioVisualFusion",
    "ClipReplaySource",
    "FusedFrameResult",
    "TabColumn",
    "TabRecorder",
    "build_visual_pitch_priors",
]
