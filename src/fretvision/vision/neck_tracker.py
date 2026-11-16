"""Hybrid 4-corner calibration + hand-masked Lucas-Kanade optical-flow fretboard tracker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .fretboard_model import FretboardModel


class NeckTrackingStatus(str, Enum):
    """Health state of fretboard homography tracking."""

    UNCALIBRATED = "Uncalibrated"
    LOCKED = "Locked"
    STATIC = "Static Lock"
    DRIFTING = "Tracking Low — Re-center Neck"


@dataclass
class NeckTrackResult:
    """Per-frame fretboard tracking output."""

    status: NeckTrackingStatus
    model: FretboardModel | None
    tracked_point_count: int


def auto_guess_neck_corners(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Heuristically propose a 4-corner fretboard quad from dominant long parallel edges.

    Returns corners ordered as [start-top, end-top, end-bottom, start-bottom] or None.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 160)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=max(60, int(w * 0.22)),
        maxLineGap=25,
    )
    if lines is None or len(lines) < 2:
        return None

    # Keep near-horizontal / mild-angled lines (guitar neck typically within +-35 deg)
    candidates: list[tuple[float, float, float, float, float]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        dy = y2 - y1
        if dx < 20.0:
            continue
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle_deg) > 38.0:
            continue
        ymid = 0.5 * (y1 + y2)
        candidates.append((angle_deg, ymid, x1, y1, x2))

    if len(candidates) < 2:
        return None

    # Find median orientation and cluster parallel upper/lower edge candidates
    angles = np.array([c[0] for c in candidates])
    med_angle = float(np.median(angles))
    aligned = [c for c in candidates if abs(c[0] - med_angle) <= 7.0]
    if len(aligned) < 2:
        return None

    aligned.sort(key=lambda c: c[1])
    best_pair: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    best_span = 0.0
    min_neck_h = max(24.0, h * 0.05)
    max_neck_h = h * 0.30

    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            dy_mid = aligned[j][1] - aligned[i][1]
            if min_neck_h <= dy_mid <= max_neck_h:
                x_left = max(aligned[i][2], aligned[j][2])
                x_right = min(aligned[i][4], aligned[j][4])
                overlap = x_right - x_left
                if overlap > w * 0.20 and overlap > best_span:
                    best_span = overlap
                    best_pair = (aligned[i], aligned[j])

    if best_pair is None:
        return None

    top_l, bot_l = best_pair
    slope = float(np.tan(np.radians(med_angle)))
    x_start = max(20.0, min(top_l[2], bot_l[2]))
    x_end = min(w - 20.0, max(top_l[4], bot_l[4]))
    dx_half = 0.5 * (x_end - x_start)
    x_mid = 0.5 * (x_start + x_end)

    top_y_start = top_l[1] - slope * dx_half
    top_y_end = top_l[1] + slope * dx_half
    bot_y_start = bot_l[1] - slope * dx_half
    bot_y_end = bot_l[1] + slope * dx_half

    # Clamp inside frame
    corners = np.array(
        [
            [x_start, np.clip(top_y_start, 5, h - 5)],
            [x_end, np.clip(top_y_end, 5, h - 5)],
            [x_end, np.clip(bot_y_end, 5, h - 5)],
            [x_start, np.clip(bot_y_start, 5, h - 5)],
        ],
        dtype=np.float32,
    )
    _ = x_mid
    return corners


class NeckTracker:
    """Tracks a calibrated guitar fretboard using hand-masked optical flow + RANSAC."""

    def __init__(self, tracking_mode: str = "optical_flow") -> None:
        self.tracking_mode = tracking_mode
        self._anchor_model: FretboardModel | None = None
        self._current_model: FretboardModel | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._min_track_points = 14

    @property
    def model(self) -> FretboardModel | None:
        return self._current_model

    @property
    def is_calibrated(self) -> bool:
        return self._current_model is not None

    def set_calibration(
        self,
        corners_px: np.ndarray,
        start_fret: int = 0,
        end_fret: int = 12,
        max_fret: int = 24,
    ) -> FretboardModel:
        """Set or reset the fretboard calibration from 4 image corners."""
        model = FretboardModel(
            corners_px=corners_px,
            start_fret=start_fret,
            end_fret=end_fret,
            max_fret=max_fret,
        )
        self._anchor_model = model
        self._current_model = model
        self._prev_gray = None
        self._prev_pts = None
        return model

    def clear(self) -> None:
        """Clear calibration and optical flow state."""
        self._anchor_model = None
        self._current_model = None
        self._prev_gray = None
        self._prev_pts = None

    def reset_to_anchor(self) -> None:
        """Snap back to initial calibrated position if optical flow drifted."""
        if self._anchor_model is not None:
            self._current_model = self._anchor_model
            self._prev_gray = None
            self._prev_pts = None

    def update(
        self,
        frame_bgr: np.ndarray,
        hand_masks: list[tuple[int, int, int, int]] | None = None,
    ) -> NeckTrackResult:
        """Update fretboard pose for the current frame while ignoring moving hands."""
        if self._current_model is None:
            return NeckTrackResult(
                status=NeckTrackingStatus.UNCALIBRATED,
                model=None,
                tracked_point_count=0,
            )

        if self.tracking_mode == "static":
            return NeckTrackResult(
                status=NeckTrackingStatus.STATIC,
                model=self._current_model,
                tracked_point_count=0,
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Build mask covering neck neighborhood minus hand bounding boxes
        mask = self._build_neck_feature_mask(
            (h, w),
            self._current_model,
            hand_masks or [],
        )

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < self._min_track_points:
            self._prev_pts = self._detect_features(gray, mask)
            self._prev_gray = gray
            status = (
                NeckTrackingStatus.LOCKED
                if self._prev_pts is not None and len(self._prev_pts) >= 8
                else NeckTrackingStatus.DRIFTING
            )
            cnt = 0 if self._prev_pts is None else len(self._prev_pts)
            return NeckTrackResult(status=status, model=self._current_model, tracked_point_count=cnt)

        next_pts, st, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._prev_pts,
            None,
            winSize=(19, 19),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        valid_mask = (st.reshape(-1) == 1) if st is not None else np.zeros(0, dtype=bool)
        if next_pts is None or int(np.sum(valid_mask)) < 8:
            self._prev_pts = self._detect_features(gray, mask)
            self._prev_gray = gray
            return NeckTrackResult(
                status=NeckTrackingStatus.DRIFTING,
                model=self._current_model,
                tracked_point_count=int(np.sum(valid_mask)),
            )

        src = self._prev_pts[valid_mask].reshape(-1, 2)
        dst = next_pts[valid_mask].reshape(-1, 2)

        # Remove points that entered a hand mask in the new frame
        if hand_masks:
            keep = []
            for idx, (px, py) in enumerate(dst):
                inside_hand = any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in hand_masks)
                keep.append(not inside_hand)
            keep_arr = np.asarray(keep, dtype=bool)
            if int(np.sum(keep_arr)) >= 8:
                src = src[keep_arr]
                dst = dst[keep_arr]

        # Estimate constrained similarity/homography (prefer affine partial 2D if few points)
        if len(src) >= 12:
            H_delta, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
        else:
            M_aff, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            H_delta = np.vstack([M_aff, [0.0, 0.0, 1.0]]) if M_aff is not None else None

        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        if H_delta is not None and inlier_count >= 8 and self._is_sane_update(H_delta, (w, h)):
            self._current_model = self._current_model.with_delta_homography(H_delta)
            status = NeckTrackingStatus.LOCKED
        else:
            status = NeckTrackingStatus.DRIFTING

        # Refresh feature set inside current updated neck region
        updated_mask = self._build_neck_feature_mask((h, w), self._current_model, hand_masks or [])
        if inlier_count < 24:
            self._prev_pts = self._detect_features(gray, updated_mask)
        else:
            self._prev_pts = dst.reshape(-1, 1, 2)
        self._prev_gray = gray

        cnt = 0 if self._prev_pts is None else len(self._prev_pts)
        return NeckTrackResult(status=status, model=self._current_model, tracked_point_count=cnt)

    def _is_sane_update(self, H_delta: np.ndarray, frame_wh: tuple[int, int]) -> bool:
        """Reject degenerate homographies or large single-frame jumps."""
        if self._current_model is None:
            return False
        det = float(np.linalg.det(H_delta[:2, :2]))
        if not (0.85 <= det <= 1.18):
            return False

        old_corners = self._current_model.get_calibrated_corners_px()
        new_corners = cv2.perspectiveTransform(
            old_corners.reshape(-1, 1, 2).astype(np.float32),
            H_delta.astype(np.float32),
        ).reshape(-1, 2)

        max_shift = float(np.max(np.linalg.norm(new_corners - old_corners, axis=1)))
        if max_shift > 35.0:
            return False

        # Ensure corners stay finite and reasonably near frame
        w, h = frame_wh
        if not np.all(np.isfinite(new_corners)):
            return False
        if np.min(new_corners[:, 0]) < -w * 0.5 or np.max(new_corners[:, 0]) > w * 1.5:
            return False
        if np.min(new_corners[:, 1]) < -h * 0.5 or np.max(new_corners[:, 1]) > h * 1.5:
            return False
        return True

    def _build_neck_feature_mask(
        self,
        shape_hw: tuple[int, int],
        model: FretboardModel,
        hand_masks: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = model.get_visible_neck_polygon(model.start_fret, model.end_fret).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        # Dilate slightly to capture crisp neck boundary edges & fret ends
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.dilate(mask, kernel)
        for x0, y0, x1, y1 in hand_masks:
            cx0 = max(0, x0 - 12)
            cy0 = max(0, y0 - 12)
            cx1 = min(w, x1 + 12)
            cy1 = min(h, y1 + 12)
            mask[cy0:cy1, cx0:cx1] = 0
        return mask

    @staticmethod
    def _detect_features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=65,
            qualityLevel=0.02,
            minDistance=8,
            mask=mask,
            blockSize=7,
        )
