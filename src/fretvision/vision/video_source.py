"""Thread-safe camera, phone IP stream, and replay video capture."""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraDevice:
    """Available local camera device."""

    index: int
    name: str


def enumerate_cameras(max_probes: int = 4) -> list[CameraDevice]:
    """Probe local camera indices safely without blocking indefinitely."""
    devices: list[CameraDevice] = []
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    prev_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    try:
        for idx in range(max_probes):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap.release()
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    devices.append(CameraDevice(index=idx, name=f"Camera {idx}"))
            cap.release()
    finally:
        if prev_level is not None and hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(prev_level)
    return devices


class VideoSource:
    """Non-blocking latest-frame reader for webcams, phone HTTP/RTSP streams, and video files."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        mirror: bool = False,
        rotation_deg: int = 0,
        loop_file: bool = True,
    ) -> None:
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.target_fps = int(target_fps)
        self.mirror = bool(mirror)
        self.rotation_deg = int(rotation_deg) % 360
        self.loop_file = bool(loop_file)

        self._cap: cv2.VideoCapture | None = None
        self._is_file = isinstance(source, str) and not source.startswith(("http://", "https://", "rtsp://"))
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None

    def open(self) -> bool:
        """Open the capture device or stream and start the background reader thread."""
        self.close()
        self._last_error = None

        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap.release()
                cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(str(self.source))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            self._last_error = f"Could not open video source: {self.source}"
            return False

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def _reader_loop(self) -> None:
        assert self._cap is not None
        file_fps = self._cap.get(cv2.CAP_PROP_FPS) or float(self.target_fps)
        file_period = 1.0 / max(1.0, min(60.0, file_fps))

        while self._running and self._cap is not None:
            t0 = time.monotonic()
            ok, frame = self._cap.read()
            if not ok:
                if self._is_file and self.loop_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.02)
                continue

            frame = self._apply_orientation(frame)
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = time.monotonic()

            if self._is_file:
                elapsed = time.monotonic() - t0
                if elapsed < file_period:
                    time.sleep(file_period - elapsed)

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

    def read_latest(self) -> tuple[np.ndarray | None, float]:
        """Return a copy of the latest available BGR frame and its monotonic timestamp."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_ts

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def close(self) -> None:
        """Stop reader thread and release capture resource."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest_frame = None
            self._latest_ts = 0.0
