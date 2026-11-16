"""Hybrid 4-corner calibration + hand-masked Lucas-Kanade optical-flow fretboard tracker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .fretboard_model import FretboardModel
from .neural_neck import NeuralNeckDetector, detect_fretboard_cv


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


def auto_guess_neck_corners(
    frame_bgr: np.ndarray,
    neural_detector: NeuralNeckDetector | None = None,
) -> np.ndarray | None:
    """Detect 4 fretboard corners using GPU ONNX-YOLO-Pose or fret-periodic CV fallback."""
    if neural_detector is not None:
        return neural_detector.detect_corners(frame_bgr)
    return detect_fretboard_cv(frame_bgr)


class NeckTracker:
    """Tracks a calibrated guitar fretboard using hand-masked optical flow + ORB re-lock."""

    def __init__(self, tracking_mode: str = "optical_flow") -> None:
        self.tracking_mode = tracking_mode
        self._anchor_model: FretboardModel | None = None
        self._current_model: FretboardModel | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._min_track_points = 14

        self._orb = cv2.ORB_create(nfeatures=500)
        self._anchor_kp: tuple[cv2.KeyPoint, ...] | None = None
        self._anchor_des: np.ndarray | None = None

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
        self._anchor_kp = None
        self._anchor_des = None
        return model

    def clear(self) -> None:
        """Clear calibration and optical flow state."""
        self._anchor_model = None
        self._current_model = None
        self._prev_gray = None
        self._prev_pts = None
        self._anchor_kp = None
        self._anchor_des = None

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

        # Capture ORB anchor keypoints strictly inside neck region on first valid frame
        if self._anchor_des is None and self._anchor_model is not None:
            self._capture_anchor_orb(gray, hand_masks or [])

        # Build mask covering neck neighborhood minus hand bounding boxes
        mask = self._build_neck_feature_mask(
            (h, w),
            self._current_model,
            hand_masks or [],
        )

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < self._min_track_points:
            # Try ORB re-localization first if we previously lost points
            if self._prev_gray is not None and self._relocalize_orb(gray, hand_masks or []):
                updated_mask = self._build_neck_feature_mask((h, w), self._current_model, hand_masks or [])
                self._prev_pts = self._detect_features(gray, updated_mask)
                self._prev_gray = gray
                cnt = 0 if self._prev_pts is None else len(self._prev_pts)
                return NeckTrackResult(
                    status=NeckTrackingStatus.LOCKED,
                    model=self._current_model,
                    tracked_point_count=cnt,
                )

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
            if self._relocalize_orb(gray, hand_masks or []):
                updated_mask = self._build_neck_feature_mask((h, w), self._current_model, hand_masks or [])
                self._prev_pts = self._detect_features(gray, updated_mask)
                self._prev_gray = gray
                cnt = 0 if self._prev_pts is None else len(self._prev_pts)
                return NeckTrackResult(
                    status=NeckTrackingStatus.LOCKED,
                    model=self._current_model,
                    tracked_point_count=cnt,
                )
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
            try:
                self._current_model = self._current_model.with_delta_homography(H_delta)
                status = NeckTrackingStatus.LOCKED
            except ValueError:
                status = NeckTrackingStatus.DRIFTING
        else:
            if self._relocalize_orb(gray, hand_masks or []):
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

    def _capture_anchor_orb(
        self,
        gray: np.ndarray,
        hand_masks: list[tuple[int, int, int, int]],
    ) -> None:
        """Index ORB descriptors strictly on the calibrated guitar neck region."""
        if self._anchor_model is None:
            return
        h, w = gray.shape[:2]
        mask = self._build_neck_feature_mask((h, w), self._anchor_model, hand_masks)
        kp, des = self._orb.detectAndCompute(gray, mask)
        if des is not None and len(kp) >= 12:
            self._anchor_kp = kp
            self._anchor_des = des

    def _relocalize_orb(
        self,
        gray: np.ndarray,
        hand_masks: list[tuple[int, int, int, int]],
    ) -> bool:
        """Recover neck homography from anchor ORB features after drift or occlusion."""
        if self._anchor_model is None or self._anchor_kp is None or self._anchor_des is None:
            return False

        h, w = gray.shape[:2]
        search_mask = np.full((h, w), 255, dtype=np.uint8)
        for x0, y0, x1, y1 in hand_masks:
            search_mask[max(0, y0 - 12) : min(h, y1 + 12), max(0, x0 - 12) : min(w, x1 + 12)] = 0

        cur_kp, cur_des = self._orb.detectAndCompute(gray, search_mask)
        if cur_des is None or len(cur_kp) < 12:
            return False

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(self._anchor_des, cur_des, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            return False

        src = np.float32([self._anchor_kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst = np.float32([cur_kp[m.trainIdx].pt for m in good]).reshape(-1, 2)
        H_orb, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H_orb is None or inliers is None or int(np.sum(inliers)) < 9:
            return False

        det = float(np.linalg.det(H_orb[:2, :2]))
        if not (0.55 <= det <= 1.85):
            return False

        anchor_corners = self._anchor_model.get_calibrated_corners_px()
        new_corners = cv2.perspectiveTransform(
            anchor_corners.reshape(-1, 1, 2).astype(np.float32),
            H_orb.astype(np.float32),
        ).reshape(-1, 2)
        if (
            np.min(new_corners[:, 0]) < -w * 0.25
            or np.max(new_corners[:, 0]) > w * 1.25
            or np.min(new_corners[:, 1]) < -h * 0.25
            or np.max(new_corners[:, 1]) > h * 1.25
        ):
            return False

        try:
            FretboardModel._validate_corners(new_corners)
            self._current_model = self._anchor_model.with_delta_homography(H_orb)
            return True
        except ValueError:
            return False
