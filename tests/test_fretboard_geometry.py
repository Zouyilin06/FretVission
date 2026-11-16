"""Tests for 12-TET fretboard homography geometry and hand-masked optical flow tracking."""

from __future__ import annotations

import cv2
import numpy as np

from fretvision.vision.fretboard_model import (
    FretboardModel,
    fret_to_canonical_u,
    u_to_continuous_fret,
)
from fretvision.vision.neck_tracker import NeckTracker, NeckTrackingStatus


def test_12tet_fret_canonical_round_trip() -> None:
    assert fret_to_canonical_u(0) == 0.0
    assert abs(fret_to_canonical_u(12) - 0.5) < 1e-6
    assert abs(fret_to_canonical_u(24) - 0.75) < 1e-6
    for f in (1, 3, 5, 7, 12, 15, 22):
        assert abs(u_to_continuous_fret(fret_to_canonical_u(f)) - f) < 1e-4


def test_perspective_homography_mapping_subneck() -> None:
    # Tilted + tapered neck covering frets 3..15
    corners = np.array(
        [
            [120.0, 140.0],  # fret 3, Low-E top edge
            [560.0, 110.0],  # fret 15, Low-E top edge
            [575.0, 225.0],  # fret 15, High-E bottom edge
            [115.0, 230.0],  # fret 3, High-E bottom edge
        ],
        dtype=np.float32,
    )
    model = FretboardModel(corners_px=corners, start_fret=3, end_fret=15, max_fret=22)

    for s_idx in range(6):
        for fret in (1, 4, 7, 12, 15, 19):
            px, py = model.fret_string_center_px(s_idx, fret)
            mapped = model.map_image_point(px, py)
            assert mapped.in_bounds
            assert mapped.string_idx == s_idx
            assert mapped.fret_idx == fret


def test_neck_tracker_optical_flow_ignores_moving_hand() -> None:
    rng = np.random.default_rng(42)
    h, w = 360, 640

    # Create textured background + crisp neck block with fret wires & texture
    base = np.full((h, w, 3), 25, dtype=np.uint8)
    corners0 = np.array([[80, 120], [520, 115], [525, 220], [80, 215]], dtype=np.float32)

    def render_neck(shift_x: int, shift_y: int, hand_box: tuple[int, int, int, int]) -> np.ndarray:
        img = base.copy()
        shifted = corners0 + np.array([[shift_x, shift_y]], dtype=np.float32)
        poly = shifted.astype(np.int32)
        cv2.fillPoly(img, [poly], (80, 60, 50))
        # Add high-contrast fret dots & grain inside neck
        fb = FretboardModel(shifted, start_fret=0, end_fret=12)
        for f in range(1, 13):
            p0, p1 = fb.fret_wire_segment(f)
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (230, 230, 230), 2)
        for f in (3, 5, 7, 9, 12):
            cx, cy = fb.fret_string_center_px(2, f)
            cv2.circle(img, (int(cx), int(cy)), 4, (240, 240, 240), -1)
        # Draw stationary textured pegs near neck edges
        for px_o in range(95, 510, 28):
            cv2.rectangle(
                img,
                (px_o + shift_x, 122 + shift_y),
                (px_o + shift_x + 6, 128 + shift_y),
                (200, 180, 150),
                -1,
            )
            cv2.rectangle(
                img,
                (px_o + shift_x, 206 + shift_y),
                (px_o + shift_x + 6, 212 + shift_y),
                (200, 180, 150),
                -1,
            )
        # Overlay a large moving hand patch that shifts dramatically in opposite direction
        hx0, hy0, hx1, hy1 = hand_box
        noise_patch = rng.integers(90, 220, size=(hy1 - hy0, hx1 - hx0, 3), dtype=np.uint8)
        img[hy0:hy1, hx0:hx1] = noise_patch
        return img

    tracker = NeckTracker(tracking_mode="optical_flow")
    tracker.set_calibration(corners0, start_fret=0, end_fret=12)

    hand0 = (160, 130, 240, 260)
    f0 = render_neck(0, 0, hand0)
    res0 = tracker.update(f0, hand_masks=[hand0])
    assert res0.status == NeckTrackingStatus.LOCKED

    # Shift neck smoothly by +12 px right, +6 px down over 3 steps while hand sweeps left!
    for step, (sx, sy, hx) in enumerate([(4, 2, 145), (8, 4, 130), (12, 6, 115)]):
        hand_box = (hx, 130, hx + 80, 260)
        frame = render_neck(sx, sy, hand_box)
        res = tracker.update(frame, hand_masks=[hand_box])
        assert res.model is not None

    assert tracker.model is not None
    tracked_corners = tracker.model.get_calibrated_corners_px()
    expected_corners = corners0 + np.array([[12.0, 6.0]], dtype=np.float32)
    err = float(np.mean(np.linalg.norm(tracked_corners - expected_corners, axis=1)))
    assert err < 4.5
