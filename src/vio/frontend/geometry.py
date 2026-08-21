"""Triangulation, two-view initialisation and pose-from-points.

The linear parts are written out because they are short and because their
failure modes matter more than their speed: a triangulation that silently
returns a point behind the camera, or one 400 metres away from a two-pixel
baseline, is how a visual odometry front end poisons its own back end.  Every
function here returns an explicit validity mask rather than a point that has to
be sanity-checked by the caller.

The RANSAC-wrapped estimators, essential matrix and PnP, are OpenCV's.
Those are well-tested implementations of algorithms whose interesting content
is the minimal solver and swapping in a Python RANSAC loop would make the
front end ten times slower while making the *back end*, which is the part worth
judging, no more mine than it already is.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import Pose, so3


def triangulate_dlt(poses, bearings, min_angle_deg: float = 1.0):
    """Multi-view triangulation by the direct linear transform.

    Parameters
    ----------
    poses:
        ``T_world_cam`` for each observation.
    bearings:
        Undistorted normalised coordinates ``(N, 2)``, one per pose.
    min_angle_deg:
        Reject the point if the widest angle between any two observing rays is
        below this.  A small parallax angle is exactly the case where the
        linear solution is dominated by noise: the depth error grows as
        ``1 / sin(angle)``, so at 0.1 degrees a half-pixel of noise is tens of
        metres.  Rejecting is better than returning a number.

    Returns
    -------
    ``(point, ok)``.  ``point`` is meaningless when ``ok`` is False.
    """
    n = len(poses)
    if n < 2:
        return np.zeros(3), False
    xn = np.atleast_2d(np.asarray(bearings, dtype=float))

    A = np.empty((2 * n, 4))
    dirs = np.empty((n, 3))
    for i, (T, x) in enumerate(zip(poses, xn)):
        T_cw = T.inverse()
        P = np.hstack([T_cw.R, T_cw.t.reshape(3, 1)])
        A[2 * i] = x[0] * P[2] - P[0]
        A[2 * i + 1] = x[1] * P[2] - P[1]
        d = T.R @ np.array([x[0], x[1], 1.0])
        dirs[i] = d / np.linalg.norm(d)

    # Widest pairwise ray angle.  For two views this is the parallax; for many
    # it is the best-conditioned pair, which is what actually fixes the depth.
    cosines = dirs @ dirs.T
    min_cos = float(np.min(cosines))
    angle = np.degrees(np.arccos(np.clip(min_cos, -1.0, 1.0)))
    if angle < min_angle_deg:
        return np.zeros(3), False

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    if abs(h[3]) < 1e-12:
        return np.zeros(3), False
    p = h[:3] / h[3]

    # Cheirality: the point has to be in front of every camera that saw it.
    for T in poses:
        if (T.R.T @ (p - T.t))[2] <= 1e-6:
            return p, False
    return p, True


def triangulate_two_view(T0: Pose, T1: Pose, x0, x1, min_angle_deg: float = 1.0):
    return triangulate_dlt([T0, T1], np.stack([np.asarray(x0), np.asarray(x1)]),
                           min_angle_deg)


def refine_point(poses, bearings, point, iterations: int = 5):
    """Gauss-Newton refinement of a triangulated point.

    The DLT minimises an algebraic error, not reprojection error.  Three or
    four Newton steps on the geometric cost typically move the point by a few
    millimetres at 5 m range, which is small but free and it removes the
    algebraic-versus-geometric objection to seeding a bundle adjustment with a
    linear solution.
    """
    p = np.asarray(point, dtype=float).copy()
    xn = np.atleast_2d(np.asarray(bearings, dtype=float))
    for _ in range(iterations):
        H = np.zeros((3, 3))
        g = np.zeros(3)
        for T, x in zip(poses, xn):
            R_cw = T.R.T
            pc = R_cw @ (p - T.t)
            if pc[2] <= 1e-6:
                return p
            iz = 1.0 / pc[2]
            r = np.array([pc[0] * iz - x[0], pc[1] * iz - x[1]])
            Jp = np.array([[iz, 0.0, -pc[0] * iz * iz], [0.0, iz, -pc[1] * iz * iz]])
            J = Jp @ R_cw
            H += J.T @ J
            g += J.T @ r
        try:
            step = np.linalg.solve(H + np.eye(3) * 1e-9, -g)
        except np.linalg.LinAlgError:
            return p
        p = p + step
        if np.linalg.norm(step) < 1e-9:
            break
    return p


@dataclass
class TwoViewResult:
    """Relative pose from a point correspondence set, up to scale."""

    T_0_1: Pose
    inliers: np.ndarray
    parallax_deg: float
    ok: bool
    reason: str = ""


def relative_pose(x0, x1, threshold_norm: float = 2.0e-3,
                  min_inliers: int = 20) -> TwoViewResult:
    """Essential-matrix relative pose between two views, translation unit norm.

    ``x0`` and ``x1`` are undistorted normalised coordinates, so the identity
    is passed as the camera matrix and ``threshold_norm`` is in normalised
    units, roughly ``pixels / focal_length``.  The default 2e-3 is about one
    pixel on a 460-pixel focal length.

    Monocular scale is not observable here and no attempt is made to invent it.
    The translation comes back as a unit vector and it is the initialiser's job
    to give it a length, which on this system is the IMU's job.
    """
    x0 = np.atleast_2d(np.asarray(x0, dtype=float))
    x1 = np.atleast_2d(np.asarray(x1, dtype=float))
    if len(x0) < 8:
        return TwoViewResult(Pose.identity(), np.zeros(0, bool), 0.0, False,
                             "fewer than 8 correspondences")

    E, mask = cv2.findEssentialMat(
        x0, x1, np.eye(3), method=cv2.RANSAC, prob=0.9999, threshold=threshold_norm
    )
    if E is None or E.shape != (3, 3):
        return TwoViewResult(Pose.identity(), np.zeros(len(x0), bool), 0.0, False,
                             "essential matrix failed")
    mask = mask.reshape(-1).astype(bool)
    if mask.sum() < min_inliers:
        return TwoViewResult(Pose.identity(), mask, 0.0,
                             False, f"only {int(mask.sum())} inliers")

    n_good, R, t, pose_mask = cv2.recoverPose(E, x0[mask], x1[mask], np.eye(3))
    inliers = np.zeros(len(x0), bool)
    idx = np.nonzero(mask)[0]
    inliers[idx[pose_mask.reshape(-1).astype(bool)]] = True

    # recoverPose returns R, t taking view 0 into view 1, so the pose of camera
    # 1 expressed in camera 0's frame is the inverse.
    T_0_1 = Pose(R, t.reshape(3)).inverse()

    b0 = np.concatenate([x0[inliers], np.ones((int(inliers.sum()), 1))], axis=1)
    b1 = np.concatenate([x1[inliers], np.ones((int(inliers.sum()), 1))], axis=1)
    if len(b0) == 0:
        return TwoViewResult(T_0_1, inliers, 0.0, False, "no cheirality inliers")
    b0 /= np.linalg.norm(b0, axis=1, keepdims=True)
    b1 = (b1 @ T_0_1.R.T)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    parallax = np.degrees(np.arccos(np.clip(np.sum(b0 * b1, axis=1), -1.0, 1.0)))
    median_parallax = float(np.median(parallax))

    return TwoViewResult(T_0_1, inliers, median_parallax, True)


@dataclass
class PnPResult:
    T_world_cam: Pose
    inliers: np.ndarray
    ok: bool
    reason: str = ""


def solve_pnp(points_world, bearings, initial: Pose | None = None,
              threshold_norm: float = 3.0e-3, min_inliers: int = 8) -> PnPResult:
    """Camera pose from 3-D points and their normalised image coordinates."""
    P = np.atleast_2d(np.asarray(points_world, dtype=float))
    x = np.atleast_2d(np.asarray(bearings, dtype=float))
    if len(P) < 4:
        return PnPResult(initial or Pose.identity(), np.zeros(len(P), bool), False,
                         "fewer than 4 points")

    rvec = tvec = None
    use_guess = False
    if initial is not None:
        T_cw = initial.inverse()
        rvec = so3.log(T_cw.R).reshape(3, 1)
        tvec = T_cw.t.reshape(3, 1)
        use_guess = True

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        P.reshape(-1, 1, 3), x.reshape(-1, 1, 2), np.eye(3), None,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=use_guess,
        iterationsCount=200, reprojectionError=threshold_norm,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inl is None or len(inl) < min_inliers:
        n = 0 if inl is None else len(inl)
        return PnPResult(initial or Pose.identity(), np.zeros(len(P), bool), False,
                         f"pnp failed with {n} inliers")

    inliers = np.zeros(len(P), bool)
    inliers[inl.reshape(-1)] = True
    T_cw = Pose(so3.exp(rvec.reshape(3)), tvec.reshape(3))
    return PnPResult(T_cw.inverse(), inliers, True)


def reprojection_errors(T_world_cam: Pose, points_world, bearings, focal: float = 1.0):
    """Per-point reprojection error, in pixels if ``focal`` is supplied."""
    P = np.atleast_2d(np.asarray(points_world, dtype=float))
    x = np.atleast_2d(np.asarray(bearings, dtype=float))
    pc = (P - T_world_cam.t) @ T_world_cam.R
    z = pc[:, 2]
    out = np.full(len(P), np.inf)
    ok = z > 1e-6
    out[ok] = focal * np.linalg.norm(pc[ok, :2] / z[ok, None] - x[ok], axis=1)
    return out
