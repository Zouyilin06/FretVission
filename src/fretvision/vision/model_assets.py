"""Model asset management for MediaPipe HandLandmarker."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Callable

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def default_model_path() -> Path:
    """Return path to the bundled/cached MediaPipe hand_landmarker.task asset."""
    # <repo>/src/fretvision/vision/model_assets.py -> <repo>/assets/models/
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "assets" / "models" / "hand_landmarker.task"


def ensure_hand_landmarker_model(
    target_path: Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
    timeout_s: float = 30.0,
) -> Path:
    """Download hand_landmarker.task if missing and return its local Path."""
    path = target_path or default_model_path()
    if path.exists() and path.stat().st_size > 1_000_000:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".task.tmp")
    if progress_cb:
        progress_cb("Downloading MediaPipe hand_landmarker.task...")

    try:
        with urllib.request.urlopen(HAND_LANDMARKER_URL, timeout=timeout_s) as resp:
            with tmp_path.open("wb") as out:
                shutil.copyfileobj(resp, out)
        if tmp_path.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded hand_landmarker.task appears incomplete.")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    if progress_cb:
        progress_cb("MediaPipe HandLandmarker model ready.")
    return path
