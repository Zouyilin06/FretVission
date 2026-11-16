"""MediaPipe HandLandmarker wrapper and fretboard finger/barre probability mapping."""

from __future__ import annotations

# Import mediapipe before cv2 on Windows to prevent DLL initialization issues.
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .fretboard_model import FretboardModel
from .model_assets import default_model_path

FINGERTIP_LANDMARKS: dict[str, int] = {
    "T": 4,   # Thumb
    "1": 8,   # Index
    "2": 12,  # Middle
    "3": 16,  # Ring
    "4": 20,  # Pinky
}

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


@dataclass(frozen=True)
class FingerObservation:
    """Tracked fingertip projected onto fretboard coordinates."""

    finger_label: str  # "T", "1", "2", "3", "4"
    pixel_xy: tuple[float, float]
    string_float: float
    fret_float: float
    string_idx: int | None
    fret_idx: int | None
    in_bounds: bool
    confidence: float


@dataclass
class HandTrackingResult:
    """Per-frame hand detection, mapped fingertips, barre candidates, and mask."""

    hand_present: bool
    landmarks_px: list[tuple[float, float]]
    hand_bboxes: list[tuple[int, int, int, int]]
    fingers: list[FingerObservation]
    press_prob_matrix: np.ndarray  # shape (6, max_fret + 1)
    model_ready: bool
    error_message: str | None = None


class HandTracker:
    """Tracks the fretting hand via MediaPipe Tasks HandLandmarker (RunningMode.VIDEO)."""

    def __init__(
        self,
        model_path: Path | None = None,
        max_fret: int = 24,
        smoothing_alpha: float = 0.55,
    ) -> None:
        self.model_path = model_path or default_model_path()
        self.max_fret = int(max_fret)
        self.smoothing_alpha = float(smoothing_alpha)
        self._landmarker: mp_vision.HandLandmarker | None = None
        self._init_error: str | None = None
        self._last_timestamp_ms: int = -1
        self._smoothed_px: dict[str, tuple[float, float]] = {}

    def initialize(self) -> bool:
        """Load the MediaPipe HandLandmarker task if model file exists."""
        if self._landmarker is not None:
            return True
        if not self.model_path.exists():
            self._init_error = (
                f"HandLandmarker model missing at {self.model_path}. "
                "Download it via Settings or setup."
            )
            return False

        try:
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.45,
                min_hand_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._init_error = None
            return True
        except Exception as exc:
            self._init_error = f"Failed to initialize MediaPipe HandLandmarker: {exc}"
            self._landmarker = None
            return False

    def close(self) -> None:
        """Release MediaPipe native landmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        fretboard: FretboardModel | None,
    ) -> HandTrackingResult:
        """Detect hands and project fretting fingertips onto the fretboard model."""
        empty_matrix = np.zeros((6, self.max_fret + 1), dtype=np.float32)

        if self._landmarker is None and not self.initialize():
            return HandTrackingResult(
                hand_present=False,
                landmarks_px=[],
                hand_bboxes=[],
                fingers=[],
                press_prob_matrix=empty_matrix,
                model_ready=False,
                error_message=self._init_error,
            )

        assert self._landmarker is not None
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            self._smoothed_px.clear()
            return HandTrackingResult(
                hand_present=False,
                landmarks_px=[],
                hand_bboxes=[],
                fingers=[],
                press_prob_matrix=empty_matrix,
                model_ready=True,
            )

        # Convert all hands to pixel coordinates & bounding boxes (for neck tracker masking)
        all_hands_px: list[list[tuple[float, float]]] = []
        bboxes: list[tuple[int, int, int, int]] = []
        for lm_list in result.hand_landmarks:
            pts = [(lm.x * w, lm.y * h) for lm in lm_list]
            all_hands_px.append(pts)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bboxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))

        fretting_hand_idx = self._select_fretting_hand(all_hands_px, fretboard)
        fretting_pts = all_hands_px[fretting_hand_idx]

        smoothed_tips: dict[str, tuple[float, float]] = {}
        for label, lm_idx in FINGERTIP_LANDMARKS.items():
            raw_x, raw_y = fretting_pts[lm_idx]
            prev = self._smoothed_px.get(label)
            if prev is not None and math.hypot(raw_x - prev[0], raw_y - prev[1]) < 90.0:
                sx = self.smoothing_alpha * raw_x + (1.0 - self.smoothing_alpha) * prev[0]
                sy = self.smoothing_alpha * raw_y + (1.0 - self.smoothing_alpha) * prev[1]
            else:
                sx, sy = raw_x, raw_y
            self._smoothed_px[label] = (sx, sy)
            smoothed_tips[label] = (sx, sy)

        base_result = HandTrackingResult(
            hand_present=True,
            landmarks_px=fretting_pts,
            hand_bboxes=bboxes,
            fingers=[
                FingerObservation(
                    finger_label=label,
                    pixel_xy=xy,
                    string_float=-1.0,
                    fret_float=-1.0,
                    string_idx=None,
                    fret_idx=None,
                    in_bounds=False,
                    confidence=0.9,
                )
                for label, xy in smoothed_tips.items()
            ],
            press_prob_matrix=empty_matrix,
            model_ready=True,
        )
        if fretboard is not None:
            return self.project_to_fretboard(base_result, fretboard)
        return base_result

    def project_to_fretboard(
        self,
        result: HandTrackingResult,
        fretboard: FretboardModel,
    ) -> HandTrackingResult:
        """Project hand landmarks and barre likelihood onto an updated FretboardModel."""
        if not result.hand_present:
            return result

        press_matrix = np.zeros((6, self.max_fret + 1), dtype=np.float32)
        fingers: list[FingerObservation] = []

        for prev_obs in result.fingers:
            sx, sy = prev_obs.pixel_xy
            label = prev_obs.finger_label
            mapped = fretboard.map_image_point(sx, sy)
            if label == "T" and mapped.in_bounds and mapped.continuous_string > 1.2:
                in_bounds_effective = False
            else:
                in_bounds_effective = mapped.in_bounds

            obs = FingerObservation(
                finger_label=label,
                pixel_xy=(sx, sy),
                string_float=mapped.continuous_string,
                fret_float=mapped.continuous_fret,
                string_idx=mapped.string_idx if in_bounds_effective else None,
                fret_idx=mapped.fret_idx if in_bounds_effective else None,
                in_bounds=in_bounds_effective,
                confidence=prev_obs.confidence,
            )
            fingers.append(obs)

            if in_bounds_effective and mapped.fret_idx is not None and mapped.fret_idx >= 1:
                self._splat_gaussian(
                    press_matrix,
                    mapped.continuous_string,
                    float(mapped.fret_idx),
                    peak=1.0 if label != "T" else 0.7,
                )

        if len(result.landmarks_px) >= 21:
            self._apply_barre_likelihood(press_matrix, result.landmarks_px, fretboard)

        return HandTrackingResult(
            hand_present=True,
            landmarks_px=result.landmarks_px,
            hand_bboxes=result.hand_bboxes,
            fingers=fingers,
            press_prob_matrix=np.clip(press_matrix, 0.0, 1.0),
            model_ready=result.model_ready,
            error_message=result.error_message,
        )

    @staticmethod
    def _select_fretting_hand(
        hands_px: list[list[tuple[float, float]]],
        fretboard: FretboardModel | None,
    ) -> int:
        if len(hands_px) == 1 or fretboard is None:
            return 0
        best_idx = 0
        best_hits = -1
        best_dist = 1e9
        poly = fretboard.get_visible_neck_polygon(0, 18)
        neck_center = np.mean(poly, axis=0)

        for idx, pts in enumerate(hands_px):
            hits = 0
            for lm_idx in (8, 12, 16, 20):
                mapped = fretboard.map_image_point(*pts[lm_idx])
                if mapped.in_bounds:
                    hits += 1
            palm_center = np.mean(np.array([pts[0], pts[5], pts[9], pts[17]]), axis=0)
            dist = float(np.linalg.norm(palm_center - neck_center))
            if hits > best_hits or (hits == best_hits and dist < best_dist):
                best_hits = hits
                best_dist = dist
                best_idx = idx
        return best_idx

    def _splat_gaussian(
        self,
        matrix: np.ndarray,
        cont_string: float,
        target_fret: float,
        peak: float = 1.0,
    ) -> None:
        """Distribute soft press likelihood around (cont_string, target_fret)."""
        s_center = int(round(cont_string))
        f_center = int(round(target_fret))
        for ds in (-1, 0, 1):
            s = s_center + ds
            if not (0 <= s < 6):
                continue
            for df in (-1, 0, 1):
                f = f_center + df
                if not (1 <= f <= self.max_fret):
                    continue
                d2 = ((s - cont_string) / 0.65) ** 2 + ((f - target_fret) / 0.55) ** 2
                val = peak * math.exp(-0.5 * d2)
                if val > matrix[s, f]:
                    matrix[s, f] = val

    def _apply_barre_likelihood(
        self,
        matrix: np.ndarray,
        pts_px: list[tuple[float, float]],
        fretboard: FretboardModel,
    ) -> None:
        """If index finger joints 5..8 span multiple strings at nearly the same fret, boost barre cells."""
        mapped_joints = [fretboard.map_image_point(*pts_px[i]) for i in (6, 7, 8)]
        valid = [m for m in mapped_joints if m.in_bounds and m.fret_idx is not None and m.fret_idx >= 1]
        if len(valid) < 2:
            return
        frets = [m.fret_idx for m in valid if m.fret_idx is not None]
        strings = [m.continuous_string for m in valid]
        if max(frets) - min(frets) <= 1 and max(strings) - min(strings) >= 1.8:
            barre_fret = int(round(float(np.median(frets))))
            s_lo = max(0, int(math.floor(min(strings))))
            s_hi = min(5, int(math.ceil(max(strings))))
            for s in range(s_lo, s_hi + 1):
                matrix[s, barre_fret] = max(float(matrix[s, barre_fret]), 0.72)
