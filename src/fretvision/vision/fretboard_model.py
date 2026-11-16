"""12-TET projective fretboard geometry and image<->neck coordinate transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def fret_to_canonical_u(fret: float) -> float:
    """Return equal-temperament normalized position u in [0, 1) from nut (0) toward bridge (1)."""
    if fret <= 0.0:
        return 0.0
    return 1.0 - (2.0 ** (-float(fret) / 12.0))


def u_to_continuous_fret(u: float) -> float:
    """Inverse of fret_to_canonical_u: map normalized u to continuous fret number."""
    if u <= 0.0:
        return 0.0
    clamped = min(float(u), 0.96)  # ~fret 55 safety bound
    return -12.0 * math.log2(1.0 - clamped)


def string_to_canonical_v(string_idx: float, edge_margin: float = 0.085) -> float:
    """Map string index 0..5 (Low E..High E) to canonical fretboard v in [edge_margin, 1-edge_margin]."""
    span = 1.0 - 2.0 * edge_margin
    return edge_margin + (float(string_idx) / 5.0) * span


def v_to_continuous_string(v: float, edge_margin: float = 0.085) -> float:
    """Map canonical v coordinate back to continuous string index 0..5."""
    span = 1.0 - 2.0 * edge_margin
    return 5.0 * (float(v) - edge_margin) / max(1e-6, span)


@dataclass(frozen=True)
class NeckMapping:
    """Projected neck location for an image point."""

    u: float
    v: float
    continuous_fret: float
    continuous_string: float
    fret_idx: int | None
    string_idx: int | None
    in_bounds: bool


class FretboardModel:
    """Projective homography between canonical 12-TET fretboard coordinates and image pixels.

    Canonical coordinates:
        u = 1 - 2^(-fret / 12)  along string length (0 = nut, 1 = bridge)
        v in [0, 1]              across fretboard width (0 = Low-E edge, 1 = High-E edge)
    """

    def __init__(
        self,
        corners_px: np.ndarray,
        start_fret: int = 0,
        end_fret: int = 12,
        max_fret: int = 24,
        edge_margin: float = 0.085,
    ) -> None:
        corners = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
        if end_fret <= start_fret:
            raise ValueError(f"end_fret ({end_fret}) must be > start_fret ({start_fret})")

        self.start_fret = int(start_fret)
        self.end_fret = int(end_fret)
        self.max_fret = int(max_fret)
        self.edge_margin = float(edge_margin)

        u0 = fret_to_canonical_u(self.start_fret)
        u1 = fret_to_canonical_u(self.end_fret)
        canonical_quad = np.array(
            [
                [u0, 0.0],
                [u1, 0.0],
                [u1, 1.0],
                [u0, 1.0],
            ],
            dtype=np.float32,
        )

        # H maps canonical (u, v) -> image (x, y)
        self.H: np.ndarray = cv2.getPerspectiveTransform(canonical_quad, corners)
        self.H_inv: np.ndarray = np.linalg.inv(self.H)

    @classmethod
    def from_homography(
        cls,
        H: np.ndarray,
        start_fret: int = 0,
        end_fret: int = 12,
        max_fret: int = 24,
        edge_margin: float = 0.085,
    ) -> FretboardModel:
        """Construct a FretboardModel directly from an updated homography matrix H."""
        obj = cls.__new__(cls)
        obj.start_fret = int(start_fret)
        obj.end_fret = int(end_fret)
        obj.max_fret = int(max_fret)
        obj.edge_margin = float(edge_margin)
        obj.H = np.asarray(H, dtype=np.float64).copy()
        obj.H_inv = np.linalg.inv(obj.H)
        return obj

    def with_delta_homography(self, delta_H_img: np.ndarray) -> FretboardModel:
        """Apply an image-space frame-to-frame homography update and return a new model."""
        new_H = np.asarray(delta_H_img, dtype=np.float64) @ self.H.astype(np.float64)
        if abs(new_H[2, 2]) > 1e-9:
            new_H = new_H / new_H[2, 2]
        return FretboardModel.from_homography(
            new_H,
            start_fret=self.start_fret,
            end_fret=self.end_fret,
            max_fret=self.max_fret,
            edge_margin=self.edge_margin,
        )

    def neck_to_image(self, points_uv: np.ndarray) -> np.ndarray:
        """Project canonical (u, v) points of shape (N, 2) into image (x, y) pixels."""
        pts = np.asarray(points_uv, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, self.H.astype(np.float32))
        return projected.reshape(-1, 2)

    def image_to_neck_uv(self, points_xy: np.ndarray) -> np.ndarray:
        """Project image (x, y) points of shape (N, 2) back into canonical (u, v)."""
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, self.H_inv.astype(np.float32))
        return projected.reshape(-1, 2)

    def get_calibrated_corners_px(self) -> np.ndarray:
        """Return the 4 image pixel corners corresponding to [start_fret, end_fret]."""
        u0 = fret_to_canonical_u(self.start_fret)
        u1 = fret_to_canonical_u(self.end_fret)
        quad = np.array([[u0, 0.0], [u1, 0.0], [u1, 1.0], [u0, 1.0]], dtype=np.float32)
        return self.neck_to_image(quad)

    def get_visible_neck_polygon(self, fret_min: int = 0, fret_max: int | None = None) -> np.ndarray:
        """Return the 4-point polygon in image space spanning [fret_min, fret_max]."""
        hi = self.max_fret if fret_max is None else min(self.max_fret, fret_max)
        u0 = fret_to_canonical_u(fret_min)
        u1 = fret_to_canonical_u(hi)
        quad = np.array([[u0, 0.0], [u1, 0.0], [u1, 1.0], [u0, 1.0]], dtype=np.float32)
        return self.neck_to_image(quad)

    def fret_wire_segment(self, fret: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return image pixel endpoints ((x0, y0), (x1, y1)) for a fret wire."""
        u = fret_to_canonical_u(fret)
        pts = self.neck_to_image(np.array([[u, 0.0], [u, 1.0]], dtype=np.float32))
        return (float(pts[0, 0]), float(pts[0, 1])), (float(pts[1, 0]), float(pts[1, 1]))

    def string_segment(
        self, string_idx: int, fret_min: int = 0, fret_max: int | None = None
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return image pixel endpoints for string_idx (0=Low E .. 5=High E)."""
        hi = self.max_fret if fret_max is None else min(self.max_fret, fret_max)
        u0 = fret_to_canonical_u(fret_min)
        u1 = fret_to_canonical_u(hi)
        v = string_to_canonical_v(string_idx, self.edge_margin)
        pts = self.neck_to_image(np.array([[u0, v], [u1, v]], dtype=np.float32))
        return (float(pts[0, 0]), float(pts[0, 1])), (float(pts[1, 0]), float(pts[1, 1]))

    def fret_string_center_px(self, string_idx: int, fret: int) -> tuple[float, float]:
        """Return expected finger-press center pixel for (string_idx, fret)."""
        if fret <= 0:
            u = fret_to_canonical_u(0.0)
        else:
            # Guitarists press ~35% behind the target fret wire (between f-1 and f)
            u_prev = fret_to_canonical_u(fret - 1)
            u_curr = fret_to_canonical_u(fret)
            u = u_prev + 0.65 * (u_curr - u_prev)
        v = string_to_canonical_v(string_idx, self.edge_margin)
        pt = self.neck_to_image(np.array([[u, v]], dtype=np.float32))[0]
        return float(pt[0]), float(pt[1])

    def map_image_point(self, x: float, y: float) -> NeckMapping:
        """Map an image fingertip pixel (x, y) to continuous and discrete (string, fret)."""
        uv = self.image_to_neck_uv(np.array([[x, y]], dtype=np.float32))[0]
        u, v = float(uv[0]), float(uv[1])

        u_max = fret_to_canonical_u(self.max_fret + 0.5)
        cont_fret = u_to_continuous_fret(max(0.0, u))
        cont_string = v_to_continuous_string(v, self.edge_margin)

        in_u = -0.02 <= u <= u_max
        in_v = -0.08 <= v <= 1.08
        in_bounds = in_u and in_v

        if not in_bounds:
            return NeckMapping(
                u=u,
                v=v,
                continuous_fret=cont_fret,
                continuous_string=cont_string,
                fret_idx=None,
                string_idx=None,
                in_bounds=False,
            )

        # Finger pressing fret f lies between fret wire f-1 and f (+0.10 tolerance past wire f)
        if cont_fret <= 0.08:
            fret_idx = 0
        else:
            fret_idx = max(1, min(self.max_fret, int(math.ceil(cont_fret - 0.10))))

        string_idx = max(0, min(5, int(round(cont_string))))
        return NeckMapping(
            u=u,
            v=v,
            continuous_fret=cont_fret,
            continuous_string=cont_string,
            fret_idx=fret_idx,
            string_idx=string_idx,
            in_bounds=True,
        )
