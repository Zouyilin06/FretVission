"""GPU-accelerated ONNX YOLO-Pose neck detector and fret-periodic CV detector."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from .fretboard_model import FretboardModel


def default_neck_model_path() -> Path:
    """Return expected path for an optional YOLO-Pose ONNX guitar neck model."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "assets" / "models" / "fretboard_pose.onnx"


def _letterbox(
    image_bgr: np.ndarray,
    new_shape: int = 640,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize and pad image to square new_shape x new_shape preserving aspect ratio."""
    h, w = image_bgr.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = (new_shape - nw) / 2.0
    dh = (new_shape - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, (dw, dh)


def decode_yolo_pose_output(
    output: np.ndarray,
    scale: float,
    pad_xy: tuple[float, float],
    orig_hw: tuple[int, int],
    conf_threshold: float = 0.35,
) -> np.ndarray | None:
    """Decode standard YOLOv8/YOLO11-Pose output (1, 17, N) or (1, N, 17) into 4 corners.

    Expected layout per anchor: [cx, cy, w, h, conf, k1x, k1y, k1c, ..., k4x, k4y, k4c].
    Returns (4, 2) float32 corners in original pixel coordinates or None.
    """
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 17 and arr.shape[1] != 17:
        arr = arr.T  # -> (N, 17)
    elif arr.shape[0] < arr.shape[1] and arr.shape[0] >= 17:
        arr = arr.T  # -> (N, C)
    if arr.shape[1] < 17:
        return None

    scores = arr[:, 4]
    best_idx = int(np.argmax(scores))
    best_conf = float(scores[best_idx])
    if best_conf < conf_threshold:
        return None

    row = arr[best_idx]
    kpts = row[5:17].reshape(4, 3)[:, :2].copy()
    dw, dh = pad_xy
    kpts[:, 0] = (kpts[:, 0] - dw) / max(1e-6, scale)
    kpts[:, 1] = (kpts[:, 1] - dh) / max(1e-6, scale)

    h, w = orig_hw
    kpts[:, 0] = np.clip(kpts[:, 0], 0.0, float(w - 1))
    kpts[:, 1] = np.clip(kpts[:, 1], 0.0, float(h - 1))
    try:
        FretboardModel._validate_corners(kpts)
    except ValueError:
        return None
    return kpts.astype(np.float32)


def _score_fret_periodicity(gray: np.ndarray, corners: np.ndarray) -> float:
    """Score how strongly perpendicular fret wires inside a quad follow 12-TET spacing."""
    strip_w, strip_h = 260, 48
    dst = np.array(
        [[0, 0], [strip_w - 1, 0], [strip_w - 1, strip_h - 1], [0, strip_h - 1]],
        dtype=np.float32,
    )
    try:
        FretboardModel._validate_corners(corners)
    except ValueError:
        return -1.0

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(gray, M, (strip_w, strip_h))
    grad_x = np.abs(cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3))
    # Average across middle 60% of neck width to ignore outer neck borders
    profile = np.mean(grad_x[10:-10, :], axis=0)
    profile = profile - float(np.median(profile))
    max_p = float(np.max(profile))
    if max_p <= 1.0:
        return 0.0

    peaks, _ = find_peaks(profile, height=max_p * 0.25, distance=7)
    if len(peaks) < 4:
        return 0.05 * len(peaks)

    # Check monotone or 12-TET spacing ratios between consecutive fret wire peaks
    diffs = np.diff(peaks.astype(np.float64))
    if len(diffs) < 3:
        return 0.15
    ratios = diffs[1:] / np.maximum(1.0, diffs[:-1])
    # In standard left->right or right->left orientation, ratios cluster near 2^(-1/12) (~0.944) or 2^(1/12) (~1.059)
    err_fwd = np.median(np.abs(ratios - (2.0 ** (-1.0 / 12.0))))
    err_rev = np.median(np.abs(ratios - (2.0 ** (1.0 / 12.0))))
    best_err = float(min(err_fwd, err_rev))
    periodicity_score = math.exp(-6.0 * best_err) * min(1.0, len(peaks) / 8.0)
    return float(periodicity_score)


def detect_fretboard_cv(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Detect guitar neck corners using pair-wise edge clustering + 12-TET fret periodicity."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 25, 90)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=45,
        minLineLength=max(45, int(w * 0.14)),
        maxLineGap=32,
    )
    if lines is None or len(lines) < 2:
        return None

    # (angle_deg, ymid, x1, y1, x2, y2)
    candidates: list[tuple[float, float, float, float, float, float]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        dy = y2 - y1
        if dx < 20.0:
            continue
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle_deg) > 40.0:
            continue
        ymid = 0.5 * (y1 + y2)
        candidates.append((angle_deg, ymid, x1, y1, x2, y2))

    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda c: c[1])
    min_neck_h = max(24.0, h * 0.05)
    max_neck_h = h * 0.36

    best_corners: np.ndarray | None = None
    best_score = -1.0

    def _y_at(line_t: tuple[float, float, float, float, float, float], x_q: float) -> float:
        _, _, lx1, ly1, lx2, ly2 = line_t
        slope_l = (ly2 - ly1) / max(1e-6, lx2 - lx1)
        return ly1 + slope_l * (x_q - lx1)

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            # Pair any near-parallel lines regardless of distracting background angles
            if abs(candidates[i][0] - candidates[j][0]) > 10.0:
                continue
            dy_mid = candidates[j][1] - candidates[i][1]
            if not (min_neck_h <= dy_mid <= max_neck_h):
                continue
            x_left = max(candidates[i][2], candidates[j][2])
            x_right = min(candidates[i][4], candidates[j][4])
            overlap = x_right - x_left
            if overlap <= w * 0.16:
                continue

            x_start = max(12.0, min(candidates[i][2], candidates[j][2]))
            x_end = min(w - 12.0, max(candidates[i][4], candidates[j][4]))
            corners = np.array(
                [
                    [x_start, np.clip(_y_at(candidates[i], x_start), 4, h - 4)],
                    [x_end, np.clip(_y_at(candidates[i], x_end), 4, h - 4)],
                    [x_end, np.clip(_y_at(candidates[j], x_end), 4, h - 4)],
                    [x_start, np.clip(_y_at(candidates[j], x_start), 4, h - 4)],
                ],
                dtype=np.float32,
            )
            fret_score = _score_fret_periodicity(gray, corners)
            if fret_score < 0.0:
                continue
            total_score = 0.85 * fret_score + 0.15 * (overlap / float(w))
            if total_score > best_score:
                best_score = total_score
                best_corners = corners

    return best_corners


class NeuralNeckDetector:
    """ONNX Runtime (DirectML/CUDA/CPU) YOLO-Pose 4-corner fretboard detector with CV fallback."""

    def __init__(self, model_path: Path | None = None, input_size: int = 640) -> None:
        self.model_path = model_path or default_neck_model_path()
        self.input_size = int(input_size)
        self._session = None
        self._input_name: str = ""
        self._provider_name: str = "CV-FretPeriodic"
        self._load_attempted: bool = False

    @property
    def model_loaded(self) -> bool:
        return self._session is not None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def initialize(self) -> bool:
        """Load ONNX session with GPU acceleration (DirectML / CUDA) if model exists."""
        if self._session is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True

        if not self.model_path.exists():
            return False

        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError:
            return False

        avail = set(ort.get_available_providers())
        preferred = [
            p
            for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
            if p in avail
        ]
        if not preferred:
            preferred = ["CPUExecutionProvider"]

        try:
            sess = ort.InferenceSession(str(self.model_path), providers=preferred)
            self._session = sess
            self._input_name = sess.get_inputs()[0].name
            active = sess.get_providers()
            self._provider_name = active[0] if active else "ONNX"
            return True
        except Exception:
            self._session = None
            return False

    def detect_corners(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Detect 4 fretboard corners using GPU ONNX-YOLO-Pose or fret-periodic CV fallback."""
        if self.initialize() and self._session is not None:
            padded, scale, pad_xy = _letterbox(frame_bgr, self.input_size)
            rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
            blob = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1, 3, H, W)
            outputs = self._session.run(None, {self._input_name: blob})
            if outputs:
                corners = decode_yolo_pose_output(
                    outputs[0],
                    scale=scale,
                    pad_xy=pad_xy,
                    orig_hw=frame_bgr.shape[:2],
                )
                if corners is not None:
                    return corners

        return detect_fretboard_cv(frame_bgr)


def export_yolo_pose_sample(
    frame_bgr: np.ndarray,
    corners_px: np.ndarray,
    start_fret: int,
    end_fret: int,
    output_dir: Path | None = None,
    sample_id: str | None = None,
) -> Path:
    """Save a calibrated frame and YOLO-Pose 4-corner keypoint label for local GPU training."""
    import time

    base_dir = output_dir or (Path(__file__).resolve().parents[3] / "assets" / "dataset" / "neck_pose")
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    corners = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    h, w = frame_bgr.shape[:2]
    stem = sample_id or f"neck_{int(time.time() * 1000)}"
    img_path = images_dir / f"{stem}.jpg"
    lbl_path = labels_dir / f"{stem}.txt"
    meta_path = labels_dir / f"{stem}.json"

    cv2.imwrite(str(img_path), frame_bgr)

    xs = np.clip(corners[:, 0] / float(w), 0.0, 1.0)
    ys = np.clip(corners[:, 1] / float(h), 0.0, 1.0)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    bw = max(1e-4, x_max - x_min)
    bh = max(1e-4, y_max - y_min)

    kpt_tokens = " ".join(f"{float(xs[i]):.6f} {float(ys[i]):.6f} 2" for i in range(4))
    lbl_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpt_tokens}\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "start_fret": int(start_fret),
                "end_fret": int(end_fret),
                "corners_px": corners.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return img_path
