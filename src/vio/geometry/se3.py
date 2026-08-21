"""SE(3) as a rigid transform, plus the pose type the whole estimator passes about.

A :class:`Pose` is stored as a rotation matrix and a translation, never as a 4x4
matrix, because every operation the estimator performs wants ``R`` and ``t``
separately and packing them into a homogeneous matrix only to unpack it again
costs more than it saves.

The convention matches :mod:`vio.geometry.so3`: a pose maps body coordinates to
world coordinates, ``p_world = R @ p_body + t`` and it is perturbed on the
right with the translation part expressed in the *body* frame::

    T (+) xi  ==  T @ Exp(xi),      xi = [rotation (3), translation (3)]

Ordering rotation before translation is the GTSAM ordering rather than the
Sophus one.  Either works; getting them mixed does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import so3

_EPS = 1e-7


def exp(xi: np.ndarray) -> "Pose":
    """Exponential map of SE(3).  ``xi`` is ``[phi (3), rho (3)]``."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    phi, rho = xi[:3], xi[3:]
    R = so3.exp(phi)
    return Pose(R, left_jacobian_v(phi) @ rho)


def log(T: "Pose") -> np.ndarray:
    """Logarithm map of SE(3), inverse of :func:`exp`."""
    phi = so3.log(T.R)
    rho = np.linalg.solve(left_jacobian_v(phi), T.t)
    return np.concatenate([phi, rho])


def left_jacobian_v(phi: np.ndarray) -> np.ndarray:
    """The ``V`` block of the SE(3) exponential, ``t = V(phi) rho``."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = so3.hat(phi)
    if theta < _EPS:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    a = (1.0 - np.cos(theta)) / t2
    b = (theta - np.sin(theta)) / (t2 * theta)
    return np.eye(3) + a * K + b * (K @ K)


@dataclass
class Pose:
    """A rigid transform, body to world."""

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)

    # -- construction ------------------------------------------------------
    @staticmethod
    def identity() -> "Pose":
        return Pose(np.eye(3), np.zeros(3))

    @staticmethod
    def from_matrix(M: np.ndarray) -> "Pose":
        M = np.asarray(M, dtype=float).reshape(4, 4)
        return Pose(M[:3, :3], M[:3, 3])

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    # -- group operations --------------------------------------------------
    def inverse(self) -> "Pose":
        Rt = self.R.T
        return Pose(Rt, -Rt @ self.t)

    def __mul__(self, other: "Pose") -> "Pose":
        if not isinstance(other, Pose):
            return NotImplemented
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def act(self, p: np.ndarray) -> np.ndarray:
        """Apply to a point or to an ``(N, 3)`` array of points."""
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            return self.R @ p + self.t
        return p @ self.R.T + self.t

    def adjoint(self) -> np.ndarray:
        """Adjoint, in the ``[rotation, translation]`` ordering used here."""
        A = np.zeros((6, 6))
        A[:3, :3] = self.R
        A[3:, :3] = so3.hat(self.t) @ self.R
        A[3:, 3:] = self.R
        return A

    # -- manifold operations ----------------------------------------------
    def retract(self, xi: np.ndarray) -> "Pose":
        """Right-multiplicative update, the one the optimiser applies."""
        out = self * exp(xi)
        out.R = so3.project(out.R)
        return out

    def local(self, other: "Pose") -> np.ndarray:
        """Inverse of :meth:`retract`: ``a.local(b)`` gives ``xi`` with ``a.retract(xi) == b``."""
        return log(self.inverse() * other)

    def normalised(self) -> "Pose":
        return Pose(so3.project(self.R), self.t)

    def copy(self) -> "Pose":
        return Pose(self.R.copy(), self.t.copy())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        rpy = np.degrees(so3.log(self.R))
        return f"Pose(t={np.round(self.t, 4).tolist()}, rotvec_deg={np.round(rpy, 3).tolist()})"
