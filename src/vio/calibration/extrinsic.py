"""Camera-IMU rotational and translational extrinsic calibration.

Rotation first, on its own, because it can be: rotation is the one part of
the extrinsic that is observable from *rotation alone*, with no scale, no
translation and no structure involved.  For any pair of instants the camera and
the body turn through related amounts::

    dR_body = R_bc  dR_cam  R_bc^T

Taking logarithms, ``theta_body = R_bc theta_cam`` for every pair, which is
Procrustes: stack the two sets of rotation vectors and take the SVD.  This is
the classical hand-eye result specialised to the case where only rotation is
wanted and it needs nothing more than a gyroscope and a visual rotation
estimate.  It is also what makes the rest tractable: with ``R_bc`` known to a
degree, the translation is a well-behaved refinement rather than a search.

Translation second, jointly, because it is not separately observable.
``t_bc`` only enters through the lever-arm acceleration that rotation induces,
so it needs the full visual-inertial problem: landmarks, poses, IMU factors and
:class:`vio.backend.factors.ExtrinsicReprojectionFactor` with the extrinsic as
a free variable.  Under pure translation it is not observable at all, which is
why :func:`calibrate` reports the rotational excitation alongside the answer
rather than returning a number with no indication of whether it means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend import factors as F
from ..backend.graph import FactorGraph
from ..backend.solver import SolverOptions, covariance, optimise
from ..backend.variables import PointVariable, PoseVariable, Values
from ..geometry import Pose, so3


@dataclass
class ExtrinsicResult:
    ok: bool
    reason: str = ""
    T_body_cam: Pose = field(default_factory=Pose.identity)
    #: Angle between the estimate and the starting value, degrees.
    rotation_change_deg: float = 0.0
    translation_change_m: float = 0.0
    #: Root-mean-square disagreement of the Procrustes fit, in radians.
    rotation_residual: float = 0.0
    #: Total rotation swept during the calibration window, radians.  Below
    #: about half a radian per axis the translation is not observable.
    excitation_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_pairs: int = 0
    #: 6x6 covariance of the extrinsic, if the joint refinement ran.
    covariance: np.ndarray | None = None


def rotation_from_increments(cam_increments, body_increments,
                             min_angle: float = 0.02) -> tuple:
    """Solve ``theta_body = R_bc theta_cam`` by orthogonal Procrustes.

    ``min_angle`` drops pairs whose rotation is too small to carry direction
    information: near zero the logarithm is dominated by noise and including
    those pairs makes the fit worse, not better, because Procrustes weights
    every pair by its magnitude squared and a noise-only pair contributes pure
    noise at full weight.
    """
    a, b = [], []
    for dR_c, dR_b in zip(cam_increments, body_increments):
        tc = so3.log(dR_c)
        tb = so3.log(dR_b)
        if np.linalg.norm(tc) < min_angle or np.linalg.norm(tb) < min_angle:
            continue
        a.append(tc)
        b.append(tb)
    if len(a) < 6:
        return None, float("nan"), len(a)

    A = np.stack(a)
    B = np.stack(b)
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    residual = float(np.sqrt(np.mean(np.sum((B - A @ R.T) ** 2, axis=1))))
    return R, residual, len(a)


def rotation_from_streams(camera_rotations, body_rotations, stride: int = 1):
    """Convenience wrapper taking two aligned sequences of absolute rotations."""
    cam = [
        np.asarray(camera_rotations[i]).T @ np.asarray(camera_rotations[i + stride])
        for i in range(len(camera_rotations) - stride)
    ]
    body = [
        np.asarray(body_rotations[i]).T @ np.asarray(body_rotations[i + stride])
        for i in range(len(body_rotations) - stride)
    ]
    return rotation_from_increments(cam, body)


def refine(
    poses_wb,
    landmarks,
    observations,
    camera,
    T_body_cam_initial: Pose,
    pixel_sigma: float = 1.0,
    huber_pixels: float = 2.0,
    fix_poses: bool = False,
    iterations: int = 30,
) -> ExtrinsicResult:
    """Joint refinement of the extrinsic against a set of observations.

    ``observations`` is a list of ``(pose_index, landmark_id, normalised_xy)``.
    ``fix_poses`` holds the trajectory and solves for the extrinsic and the
    structure only, which is the right thing when the poses came from a
    trusted source such as ground truth and the wrong thing when they came
    from the very estimator being calibrated.
    """
    values = Values()
    for i, T in enumerate(poses_wb):
        values[f"x{i}"] = PoseVariable(T)
    for lid, p in landmarks.items():
        values[f"l{lid}"] = PointVariable(p)
    values["ext"] = PoseVariable(T_body_cam_initial)

    g = FactorGraph()
    loss = F.HuberLoss(huber_pixels)
    used = 0
    for pi, lid, xn in observations:
        if f"l{lid}" not in values:
            continue
        g.add(F.ExtrinsicReprojectionFactor(
            f"x{pi}", f"l{lid}", "ext", xn, camera, pixel_sigma, loss
        ))
        used += 1
    if used < 50:
        return ExtrinsicResult(False, f"only {used} observations")

    fixed = set()
    if fix_poses:
        fixed = {f"x{i}" for i in range(len(poses_wb))}
    else:
        # Gauge: hold the first two poses so the solve cannot drift as a rigid
        # body or rescale itself into the extrinsic.
        g.add(F.PosePriorFactor("x0", poses_wb[0], np.eye(6) * 1e5))
        if len(poses_wb) > 1:
            g.add(F.PosePriorFactor("x1", poses_wb[1], np.eye(6) * 1e5))

    res = optimise(g, values, fixed=fixed,
                   options=SolverOptions(max_iterations=iterations))
    T = res.values.pose("ext")
    cov = covariance(g, res.values, ["ext"], fixed=fixed)
    return ExtrinsicResult(
        ok=res.converged,
        reason=res.reason,
        T_body_cam=T,
        rotation_change_deg=float(
            np.degrees(so3.angle_between(T_body_cam_initial.R, T.R))
        ),
        translation_change_m=float(np.linalg.norm(T.t - T_body_cam_initial.t)),
        covariance=None if cov is None else cov.get("ext"),
    )


def excitation(body_rotations) -> np.ndarray:
    """Total unsigned rotation about each body axis, radians.

    A calibration run that never rotates about an axis cannot see the
    translation component perpendicular to it and reporting the number is the
    difference between a calibration and a guess.
    """
    total = np.zeros(3)
    for a, b in zip(body_rotations[:-1], body_rotations[1:]):
        total += np.abs(so3.log(np.asarray(a).T @ np.asarray(b)))
    return total
