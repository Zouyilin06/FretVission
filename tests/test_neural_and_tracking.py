"""Tests for Phase 2B: GPU ONNX YOLO-Pose decoder, fret-periodic CV, ORB re-lock, and sample export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fretvision.vision.fretboard_model import FretboardModel
from fretvision.vision.neck_tracker import NeckTracker, NeckTrackingStatus
from fretvision.vision.neural_neck import (
    NeuralNeckDetector,
    _letterbox,
    decode_yolo_pose_output,
    detect_fretboard_cv,
    export_yolo_pose_sample,
)


def _draw_textured_guitar_scene(
    corners: np.ndarray,
    w: int = 640,
    h: int = 360,
    distractor_lines: bool = True,
) -> np.ndarray:
    img = np.full((h, w, 3), 28, dtype=np.uint8)
    if distractor_lines:
        # Parallel background desk / shelf lines that lack 12-TET fret spacing
        cv2.line(img, (20, 40), (610, 40), (190, 190, 190), 2)
        cv2.line(img, (20, 82), (610, 82), (190, 190, 190), 2)

    cv2.fillPoly(img, [corners.astype(np.int32)], (75, 55, 45))
    fb = FretboardModel(corners, start_fret=0, end_fret=12)
    for f in range(0, 13):
        p0, p1 = fb.fret_wire_segment(f)
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (235, 235, 235), 2)
    for f in (3, 5, 7, 9, 12):
        cx, cy = fb.fret_string_center_px(2, f)
        cv2.circle(img, (int(cx), int(cy)), 4, (240, 240, 240), -1)
    # Crisp edge inlays / tuning marker dots along neck boundaries
    for s in (0, 5):
        for f in range(1, 13):
            px, py = fb.fret_string_center_px(s, f)
            cv2.circle(img, (int(px), int(py)), 3, (215, 200, 175), -1)
    return img


def test_yolo_pose_letterbox_decode_round_trip() -> None:
    orig = np.zeros((360, 640, 3), dtype=np.uint8)
    _, scale, pad_xy = _letterbox(orig, 640)
    expected_corners = np.array([[70.0, 110.0], [560.0, 95.0], [565.0, 225.0], [65.0, 220.0]], dtype=np.float32)

    # Build synthetic YOLOv8-Pose output (1, 17, 10)
    out = np.zeros((1, 17, 10), dtype=np.float32)
    anchor_idx = 4
    out[0, 4, anchor_idx] = 0.92  # confidence
    dw, dh = pad_xy
    for k in range(4):
        kx = expected_corners[k, 0] * scale + dw
        ky = expected_corners[k, 1] * scale + dh
        out[0, 5 + k * 3 + 0, anchor_idx] = kx
        out[0, 5 + k * 3 + 1, anchor_idx] = ky
        out[0, 5 + k * 3 + 2, anchor_idx] = 0.99

    decoded = decode_yolo_pose_output(out, scale=scale, pad_xy=pad_xy, orig_hw=orig.shape[:2])
    assert decoded is not None
    assert np.allclose(decoded, expected_corners, atol=1e-2)


def test_fret_periodic_cv_rejects_desk_distractor() -> None:
    true_corners = np.array([[80, 175], [540, 155], [545, 255], [80, 260]], dtype=np.float32)
    scene = _draw_textured_guitar_scene(true_corners, distractor_lines=True)
    detected = detect_fretboard_cv(scene)
    assert detected is not None
    # Detected vertical center must land on the guitar neck (~210px), NOT the upper shelf lines (~61px)
    mean_y = float(np.mean(detected[:, 1]))
    assert 165.0 <= mean_y <= 255.0

    # NeuralNeckDetector cleanly falls back to CV when ONNX model file doesn't exist
    det = NeuralNeckDetector(model_path=Path("nonexistent_neck_pose.onnx"))
    fb_c = det.detect_corners(scene)
    assert fb_c is not None
    assert det.provider_name == "CV-FretPeriodic"


def test_neck_tracker_orb_relocalizes_after_large_jump() -> None:
    corners0 = np.array([[90, 120], [500, 115], [505, 215], [90, 210]], dtype=np.float32)
    frame0 = _draw_textured_guitar_scene(corners0, distractor_lines=False)

    tracker = NeckTracker(tracking_mode="optical_flow")
    tracker.set_calibration(corners0, start_fret=0, end_fret=12)

    # First frame locks optical flow and captures ORB anchor descriptors on the neck
    res0 = tracker.update(frame0, hand_masks=[])
    assert res0.status == NeckTrackingStatus.LOCKED

    # Simulate guitar moving suddenly by +55 px right and +35 px down (exceeds LK 35 px jump gate)
    shift = np.array([[55.0, 35.0]], dtype=np.float32)
    corners_moved = corners0 + shift
    frame_moved = _draw_textured_guitar_scene(corners_moved, distractor_lines=False)

    # First blank occlusion frame forces LK failure, then moved frame triggers ORB re-lock
    blank = np.full_like(frame0, 25)
    res_blank = tracker.update(blank, hand_masks=[])
    assert res_blank.status == NeckTrackingStatus.DRIFTING

    res_reloc = tracker.update(frame_moved, hand_masks=[])
    assert res_reloc.status == NeckTrackingStatus.LOCKED
    assert tracker.model is not None
    reloc_corners = tracker.model.get_calibrated_corners_px()
    err = float(np.mean(np.linalg.norm(reloc_corners - corners_moved, axis=1)))
    assert err < 6.0


def test_export_yolo_pose_sample_writes_valid_label(tmp_path: Path) -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    corners = np.array([[64, 90], [576, 90], [576, 216], [64, 216]], dtype=np.float32)
    img_path = export_yolo_pose_sample(
        frame, corners, start_fret=0, end_fret=12, output_dir=tmp_path, sample_id="test_01"
    )
    assert img_path.exists()
    lbl_path = tmp_path / "labels" / "test_01.txt"
    assert lbl_path.exists()
    tokens = lbl_path.read_text(encoding="utf-8").strip().split()
    # class + 4 bbox + 4 * 3 keypoints = 17 tokens
    assert len(tokens) == 17
    assert tokens[0] == "0"
