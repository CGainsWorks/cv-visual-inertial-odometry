"""Pinhole camera models with the two distortion families the datasets use.

EuRoC ships ``radial-tangential`` intrinsics, TUM VI ships ``equidistant``
(Kannala-Brandt).  Both are here.

The estimator never works in pixels internally.  The front end tracks on the
raw distorted image, because that is what the pixels actually look like and
resampling a whole 752x480 frame per iteration to avoid a four-term polynomial
is a poor trade.  The tracked pixel coordinates are then lifted to *undistorted
normalised* coordinates once, and everything downstream, triangulation,
reprojection residuals, PnP, works there.  Residuals are scaled back by the
focal length at the end so the numbers printed are pixels, which is the only
unit anyone can judge by eye.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Newton on the distortion model.  Three steps is enough for EuRoC radtan and
# TUM VI equidistant; the cap is a guard against a pathological coefficient set.
_UNDISTORT_ITERS = 12


@dataclass
class PinholeCamera:
    """Intrinsics plus a distortion model.

    Parameters
    ----------
    width, height:
        Image size, used only for the in-bounds check.
    fx, fy, cx, cy:
        Pinhole intrinsics in pixels.
    distortion:
        Four coefficients.  ``[k1, k2, p1, p2]`` for radial-tangential,
        ``[k1, k2, k3, k4]`` for equidistant.
    model:
        ``radtan``, ``equidistant`` or ``none``.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(4))
    model: str = "radtan"

    def __post_init__(self) -> None:
        self.distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        if self.distortion.size < 4:
            self.distortion = np.pad(self.distortion, (0, 4 - self.distortion.size))
        if self.model not in ("radtan", "equidistant", "none"):
            raise ValueError(f"unknown distortion model {self.model!r}")

    # -- basic accessors ---------------------------------------------------
    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def focal(self) -> float:
        """One number to scale normalised residuals into pixels."""
        return 0.5 * (self.fx + self.fy)

    # -- distortion --------------------------------------------------------
    def distort(self, xn: np.ndarray) -> np.ndarray:
        """Normalised undistorted ``(N, 2)`` to normalised distorted ``(N, 2)``."""
        xn = np.atleast_2d(np.asarray(xn, dtype=float))
        if self.model == "none":
            return xn.copy()
        x, y = xn[:, 0], xn[:, 1]
        k = self.distortion
        if self.model == "radtan":
            r2 = x * x + y * y
            radial = 1.0 + k[0] * r2 + k[1] * r2 * r2
            dx = 2.0 * k[2] * x * y + k[3] * (r2 + 2.0 * x * x)
            dy = k[2] * (r2 + 2.0 * y * y) + 2.0 * k[3] * x * y
            return np.stack([x * radial + dx, y * radial + dy], axis=1)
        # equidistant
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        t2 = theta * theta
        theta_d = theta * (
            1.0 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4
        )
        scale = np.where(r > 1e-9, theta_d / np.maximum(r, 1e-12), 1.0)
        return np.stack([x * scale, y * scale], axis=1)

    def project(self, xn: np.ndarray) -> np.ndarray:
        """Normalised undistorted ``(N, 2)`` to pixels ``(N, 2)``."""
        d = self.distort(xn)
        return np.stack(
            [self.fx * d[:, 0] + self.cx, self.fy * d[:, 1] + self.cy], axis=1
        )

    def project_points(self, p_cam: np.ndarray):
        """3-D camera-frame points ``(N, 3)`` to pixels, plus a validity mask.

        Points at or behind the image plane are marked invalid rather than
        silently projected to a mirrored position somewhere in frame, which is
        how a bad landmark turns into a plausible-looking outlier.
        """
        p_cam = np.atleast_2d(np.asarray(p_cam, dtype=float))
        z = p_cam[:, 2]
        ok = z > 1e-6
        xn = np.zeros((p_cam.shape[0], 2))
        xn[ok] = p_cam[ok, :2] / z[ok, None]
        uv = self.project(xn)
        ok &= (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < self.height)
        )
        return uv, ok

    def distort_jacobian(self, xn: np.ndarray) -> np.ndarray:
        """``d(distorted) / d(undistorted)``, an ``(N, 2, 2)`` stack.

        Used to invert the distortion by Newton iteration, and by the
        calibration module when intrinsics are being refined.
        """
        xn = np.atleast_2d(np.asarray(xn, dtype=float))
        n = xn.shape[0]
        if self.model == "none":
            return np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
        x, y = xn[:, 0], xn[:, 1]
        k = self.distortion
        J = np.empty((n, 2, 2))
        if self.model == "radtan":
            r2 = x * x + y * y
            radial = 1.0 + k[0] * r2 + k[1] * r2 * r2
            dradial = 2.0 * (k[0] + 2.0 * k[1] * r2)
            J[:, 0, 0] = radial + x * x * dradial + 2.0 * k[2] * y + 6.0 * k[3] * x
            J[:, 0, 1] = x * y * dradial + 2.0 * k[2] * x + 2.0 * k[3] * y
            J[:, 1, 0] = x * y * dradial + 2.0 * k[2] * x + 2.0 * k[3] * y
            J[:, 1, 1] = radial + y * y * dradial + 6.0 * k[2] * y + 2.0 * k[3] * x
            return J
        # equidistant: d = xn * s(r), s = theta_d(theta(r)) / r
        r = np.sqrt(x * x + y * y)
        rs = np.maximum(r, 1e-12)
        theta = np.arctan(r)
        t2 = theta * theta
        poly = 1.0 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4
        dpoly = (
            1.0
            + 3.0 * k[0] * t2
            + 5.0 * k[1] * t2 ** 2
            + 7.0 * k[2] * t2 ** 3
            + 9.0 * k[3] * t2 ** 4
        )
        theta_d = theta * poly
        s = np.where(r > 1e-9, theta_d / rs, 1.0)
        # ds/dr = (dtheta_d/dtheta * dtheta/dr) / r - theta_d / r^2
        dtheta_dr = 1.0 / (1.0 + r * r)
        ds_dr = np.where(r > 1e-9, (dpoly * dtheta_dr) / rs - theta_d / (rs * rs), 0.0)
        J[:, 0, 0] = s + x * ds_dr * (x / rs)
        J[:, 0, 1] = x * ds_dr * (y / rs)
        J[:, 1, 0] = y * ds_dr * (x / rs)
        J[:, 1, 1] = s + y * ds_dr * (y / rs)
        return J

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """Pixels ``(N, 2)`` to normalised undistorted ``(N, 2)``.

        Written out rather than handed to ``cv2.undistortPoints`` for two
        reasons.  OpenCV 5 dropped the Python binding that let you set the
        iteration count, and its default five fixed-point steps leave about a
        tenth of a pixel of error at the corners of the EuRoC images, larger
        than the feature-tracking noise the estimator is trying to model, which
        would put a systematic bias into every residual.  The Newton iteration
        below uses the analytic Jacobian and converges to machine precision in
        three or four steps.
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        if uv.size == 0:
            return uv.reshape(0, 2)
        target = np.stack(
            [(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy], axis=1
        )
        if self.model == "none":
            return target
        xn = target.copy()
        for _ in range(_UNDISTORT_ITERS):
            err = self.distort(xn) - target
            if np.abs(err).max() < 1e-14:
                break
            J = self.distort_jacobian(xn)
            # Closed-form 2x2 solve; a batched np.linalg.solve here is about
            # four times slower and this runs once per feature per frame.
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            dx = (J[:, 1, 1] * err[:, 0] - J[:, 0, 1] * err[:, 1]) / det
            dy = (-J[:, 1, 0] * err[:, 0] + J[:, 0, 0] * err[:, 1]) / det
            xn -= np.stack([dx, dy], axis=1)
        return xn

    def bearings(self, uv: np.ndarray) -> np.ndarray:
        """Pixels to unit bearing vectors in the camera frame."""
        xn = self.undistort(uv)
        b = np.concatenate([xn, np.ones((xn.shape[0], 1))], axis=1)
        return b / np.linalg.norm(b, axis=1, keepdims=True)

    def undistort_maps(self, alpha: float = 0.0):
        """Remap tables to a distortion-free camera, for visualisation only.

        The estimator does not use these.  They exist so the tracking and
        loop-closure overlays in ``docs/`` are drawn on straight lines.
        """
        size = (self.width, self.height)
        if self.model == "equidistant":
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.distortion.reshape(4, 1), size, np.eye(3), balance=alpha
            )
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.distortion.reshape(4, 1), np.eye(3), new_K, size,
                cv2.CV_32FC1,
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K, self.distortion.reshape(1, 4), size, alpha, size
            )
            m1, m2 = cv2.initUndistortRectifyMap(
                self.K, self.distortion.reshape(1, 4), np.eye(3), new_K, size,
                cv2.CV_32FC1,
            )
        cam = PinholeCamera(
            self.width,
            self.height,
            float(new_K[0, 0]),
            float(new_K[1, 1]),
            float(new_K[0, 2]),
            float(new_K[1, 2]),
            np.zeros(4),
            "none",
        )
        return m1, m2, cam

    def to_dict(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "distortion": [float(v) for v in self.distortion],
            "model": self.model,
        }

    @staticmethod
    def from_dict(d: dict) -> "PinholeCamera":
        return PinholeCamera(
            int(d["width"]),
            int(d["height"]),
            float(d["fx"]),
            float(d["fy"]),
            float(d["cx"]),
            float(d["cy"]),
            np.asarray(d.get("distortion", np.zeros(4)), dtype=float),
            str(d.get("model", "radtan")),
        )


def projection_jacobian(p_cam: np.ndarray) -> np.ndarray:
    """d(x/z, y/z) / d(p_cam), the 2x3 block every reprojection factor needs."""
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    iz = 1.0 / z
    return np.array([[iz, 0.0, -x * iz * iz], [0.0, iz, -y * iz * iz]])
