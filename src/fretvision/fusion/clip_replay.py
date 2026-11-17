"""Synchronized video + audio clip replay source for offline transcription and tuning."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from fretvision.audio.audio_source import load_wav_mono


class ClipReplaySource:
    """Synchronizes a recorded video file (.mp4/.avi) with a paired audio track (.wav or ffmpeg)."""

    def __init__(
        self,
        video_path: str | Path,
        audio_path: str | Path | None = None,
        mirror: bool = False,
        rotation_deg: int = 0,
    ) -> None:
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path) if audio_path else None
        self.mirror = bool(mirror)
        self.rotation_deg = int(rotation_deg) % 360

        self._cap: cv2.VideoCapture | None = None
        self._temp_wav: Path | None = None
        self._audio_samples: np.ndarray = np.zeros(0, dtype=np.float32)
        self._sample_rate: int = 44100

        self.fps: float = 30.0
        self.frame_count: int = 0
        self.duration_s: float = 0.0
        self.position_s: float = 0.0
        self.playing: bool = False

        self._cached_frame_idx: int = -1
        self._cached_frame: np.ndarray | None = None
        self._last_error: str | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def open(self) -> bool:
        """Open the video file and load/extract paired audio."""
        self.close()
        self._last_error = None

        if not self.video_path.exists():
            self._last_error = f"Video file not found: {self.video_path}"
            return False

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            self._last_error = f"Could not open video clip: {self.video_path}"
            return False

        self._cap = cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.fps = fps if 1.0 <= fps <= 120.0 else 30.0
        self.frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        video_dur = self.frame_count / self.fps

        # Resolve paired WAV audio or extract via ffmpeg if available
        wav_to_load: Path | None = self.audio_path
        if wav_to_load is None:
            sibling = self.video_path.with_suffix(".wav")
            if sibling.exists():
                wav_to_load = sibling
            else:
                wav_to_load = self._extract_audio_with_ffmpeg()

        if wav_to_load is not None and wav_to_load.exists():
            try:
                self._audio_samples, self._sample_rate = load_wav_mono(wav_to_load)
            except Exception as exc:
                self._last_error = f"Audio load warning: {exc}"
                self._audio_samples = np.zeros(0, dtype=np.float32)

        audio_dur = (
            len(self._audio_samples) / float(self._sample_rate)
            if len(self._audio_samples) > 0
            else 0.0
        )
        self.duration_s = max(video_dur, audio_dur, 0.1)
        self.position_s = 0.0
        self.playing = True
        self._cached_frame_idx = -1
        self._cached_frame = None
        return True

    def _extract_audio_with_ffmpeg(self) -> Path | None:
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe is None:
            return None
        tmp_dir = Path(tempfile.mkdtemp(prefix="fretvision_clip_"))
        out_wav = tmp_dir / "extracted.wav"
        try:
            subprocess.run(
                [
                    ffmpeg_exe,
                    "-y",
                    "-i",
                    str(self.video_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-acodec",
                    "pcm_s16le",
                    str(out_wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            self._temp_wav = out_wav
            return out_wav
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

    def seek(self, time_s: float) -> None:
        self.position_s = float(max(0.0, min(self.duration_s, time_s)))

    def advance(self, dt_s: float, loop: bool = True) -> None:
        if not self.playing:
            return
        self.position_s += float(dt_s)
        if self.position_s >= self.duration_s:
            if loop:
                self.position_s = self.position_s % max(0.1, self.duration_s)
            else:
                self.position_s = self.duration_s
                self.playing = False

    def read_at_current_position(
        self, n_audio_samples: int = 4096
    ) -> tuple[np.ndarray | None, np.ndarray, int, float]:
        """Return (frame_bgr, audio_window, sample_rate, position_s) at current clip clock."""
        frame = self._read_frame_idx(int(round(self.position_s * self.fps)))

        out_audio = np.zeros(n_audio_samples, dtype=np.float32)
        if len(self._audio_samples) > 0:
            end_idx = int(round(self.position_s * self._sample_rate))
            end_idx = max(0, min(len(self._audio_samples), end_idx))
            start_idx = max(0, end_idx - n_audio_samples)
            seg = self._audio_samples[start_idx:end_idx]
            if len(seg) > 0:
                out_audio[n_audio_samples - len(seg) :] = seg

        return frame, out_audio, self._sample_rate, self.position_s

    def _read_frame_idx(self, frame_idx: int) -> np.ndarray | None:
        if self._cap is None:
            return None
        clamped = max(0, min(self.frame_count - 1, int(frame_idx)))
        if clamped != self._cached_frame_idx or self._cached_frame is None:
            if clamped != self._cached_frame_idx + 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
            ok, frame = self._cap.read()
            if ok:
                self._cached_frame = self._apply_orientation(frame)
                self._cached_frame_idx = clamped
        return None if self._cached_frame is None else self._cached_frame.copy()

    def _apply_orientation(self, frame: np.ndarray) -> np.ndarray:
        if self.rotation_deg == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_deg == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_deg == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._temp_wav is not None:
            shutil.rmtree(self._temp_wav.parent, ignore_errors=True)
            self._temp_wav = None
        self._cached_frame = None
        self._cached_frame_idx = -1
