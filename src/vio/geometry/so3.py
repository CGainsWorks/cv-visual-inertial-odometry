"""SO(3) on the manifold.

Everything downstream depends on these seven functions being right, so they are
written out rather than imported and the tests check them against finite
differences and against scipy's rotation implementation.

Convention, fixed once and used everywhere in this repository:

    a rotation is a 3x3 matrix ``R`` mapping body coordinates to world
    coordinates and it is perturbed *on the right*::

        R (+) d  ==  R @ Exp(d)

Right perturbation is what makes the IMU preintegration Jacobians in
``vio.imu.preintegration`` come out in the form Forster et al. publish, so it is
not an arbitrary choice.  Mixing left and right conventions is the single most
common way to get a VIO back end that almost converges, so there is exactly one
convention here and no left-perturbation variants exist.
"""

from __future__ import annotations

import numpy as np

# Below this angle the Taylor expansions are more accurate than the closed
# forms, which divide by theta.  1e-7 rad is roughly where float64 cancellation
# in (1 - cos t) starts to bite.
_EPS = 1e-7


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a 3-vector, so that ``hat(a) @ b == cross(a, b)``."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def vee(S: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hat`.  Takes the skew part, so it tolerates small asymmetry."""
    S = np.asarray(S, dtype=float).reshape(3, 3)
    return 0.5 * np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]])


def exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector -> rotation matrix (Rodrigues)."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < _EPS:
        # R = I + K + K^2/2 + ..., kept to second order.
        return np.eye(3) + K + 0.5 * (K @ K)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def log(R: np.ndarray) -> np.ndarray:
    """Logarithm map: rotation matrix -> rotation vector, with ``|phi| <= pi``.

    Routed through a quaternion rather than through ``arccos(trace)``.  The
    trace form loses precision quadratically as the angle approaches 0 or pi.
    Near pi it was good to only about 1e-5 rad in testing, whereas the
    ``atan2`` form below holds a few ulps across the whole range.  Half-turns
    do occur in practice on the EuRoC Vicon Room sequences, so this matters.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    w, v = _quaternion(R)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        # theta ~ 2n/w and the direction v/n cancels against it.
        return 2.0 * v / w if w != 0.0 else np.zeros(3)
    theta = 2.0 * np.arctan2(n, w)
    return (theta / n) * v


def _quaternion(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Unit quaternion ``(w, v)`` of a rotation, with ``w >= 0``.

    Shepperd's method: pick whichever of the four components is largest so the
    square root is never taken of something near zero.
    """
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0)) * 2.0
        if s < 1e-12:
            return 1.0, np.zeros(3)
        w = (R[k, j] - R[j, k]) / s
        v = np.zeros(3)
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
    if w < 0.0:
        w, v = -w, -v
    return w, v


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian ``Jr(phi)`` of SO(3).

    Defined by ``Exp(phi + dphi) ~= Exp(phi) Exp(Jr(phi) dphi)``, which is what
    turns a perturbation of a rotation *vector* into a perturbation of the
    rotation it generates.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < _EPS:
        return np.eye(3) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    a = (1.0 - np.cos(theta)) / t2
    b = (theta - np.sin(theta)) / (t2 * theta)
    return np.eye(3) - a * K + b * (K @ K)


def right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of :func:`right_jacobian`."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < _EPS:
        return np.eye(3) + 0.5 * K + (1.0 / 12.0) * (K @ K)
    c = (1.0 / (theta * theta)) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * K + c * (K @ K)


def project(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix in the Frobenius sense.

    Repeated composition walks a rotation off the manifold at about 1e-16 per
    product.  That is harmless for hundreds of frames and not harmless over a
    whole EuRoC sequence, so the estimator reprojects after every update.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def angle_between(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle in radians between two rotations."""
    return float(np.linalg.norm(log(np.asarray(R_a).T @ np.asarray(R_b))))
