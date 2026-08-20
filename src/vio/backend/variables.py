"""The variable types the optimiser knows how to move.

A variable is a value plus a retraction.  The optimiser only ever sees the
tangent dimension and calls :meth:`Variable.retract`, so adding a new variable
type does not require touching the solver.

The pose retraction here is *not* the SE(3) exponential.  It treats a pose as a
point on the product manifold ``SO(3) x R^3``::

    R <- R Exp(d_phi)          rotation perturbed on the right
    p <- p + d_p               translation perturbed in the world frame

rather than ``T <- T Exp(xi)``, which couples the two.  Both are valid
retractions and both converge to the same minimum; the decoupled one gives
markedly simpler Jacobians for the IMU and reprojection factors, and the
translation block of the resulting Hessian is better scaled because a
translation step means the same thing regardless of the current attitude.
``vio.geometry.se3`` keeps the exponential form for the places that genuinely
need the group structure, such as pose composition and interpolation.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Pose, so3
from ..types import ImuBias


class Variable:
    """Base class.  ``dim`` is the *tangent* dimension, not the storage size."""

    dim = 0

    def __init__(self, value):
        self.value = value

    def retract(self, delta: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError

    def local(self, other):  # pragma: no cover - interface
        raise NotImplementedError

    def local_jacobian(self, other):
        """``d local(self, other) / d(other retraction)``.

        The identity for a Euclidean variable, and *not* the identity for a
        pose: displacing the target rotation by ``Exp(d)`` moves the local
        coordinates by ``Jr^-1(phi) d``, not by ``d``.  A marginal prior
        evaluated away from its linearisation point needs this to differentiate
        correctly, and dropping it, which is the usual shortcut, costs
        about three parts in ten thousand at a millimetre of displacement.
        """
        return np.eye(other.dim)

    def copy(self):  # pragma: no cover - interface
        raise NotImplementedError


class PoseVariable(Variable):
    """A body pose in the world frame.  Tangent order is ``[d_phi, d_p]``."""

    dim = 6

    def __init__(self, value: Pose):
        super().__init__(value.copy() if isinstance(value, Pose) else Pose(*value))

    def retract(self, delta: np.ndarray) -> "PoseVariable":
        delta = np.asarray(delta, dtype=float).reshape(6)
        R = so3.project(self.value.R @ so3.exp(delta[:3]))
        return PoseVariable(Pose(R, self.value.t + delta[3:]))

    def local(self, other: "PoseVariable") -> np.ndarray:
        return np.concatenate(
            [so3.log(self.value.R.T @ other.value.R), other.value.t - self.value.t]
        )

    def local_jacobian(self, other: "PoseVariable") -> np.ndarray:
        J = np.eye(6)
        J[:3, :3] = so3.right_jacobian_inv(
            so3.log(self.value.R.T @ other.value.R)
        )
        return J

    def copy(self) -> "PoseVariable":
        return PoseVariable(self.value.copy())


class VectorVariable(Variable):
    """A plain Euclidean variable: velocity, gravity direction, a time offset."""

    def __init__(self, value):
        v = np.atleast_1d(np.asarray(value, dtype=float)).astype(float)
        super().__init__(v)
        self.dim = int(v.size)

    def retract(self, delta: np.ndarray) -> "VectorVariable":
        return VectorVariable(self.value + np.asarray(delta, dtype=float).reshape(-1))

    def local(self, other: "VectorVariable") -> np.ndarray:
        return other.value - self.value

    def copy(self) -> "VectorVariable":
        return VectorVariable(self.value.copy())


class PointVariable(VectorVariable):
    """A 3-D landmark.

    Distinguished from a plain vector only so the solver can find the block it
    is allowed to eliminate by Schur complement.  In a bundle adjustment the
    landmarks outnumber the poses by two orders of magnitude and each one
    touches only its own observations, which is exactly the structure the
    complement exploits.
    """

    def __init__(self, value):
        super().__init__(np.asarray(value, dtype=float).reshape(3))

    def retract(self, delta: np.ndarray) -> "PointVariable":
        return PointVariable(self.value + np.asarray(delta, dtype=float).reshape(3))

    def copy(self) -> "PointVariable":
        return PointVariable(self.value.copy())


class BiasVariable(VectorVariable):
    """IMU bias, stored as ``[acc (3), gyro (3)]``."""

    def __init__(self, value):
        if isinstance(value, ImuBias):
            value = value.vector()
        super().__init__(np.asarray(value, dtype=float).reshape(6))

    @property
    def bias(self) -> ImuBias:
        return ImuBias.from_vector(self.value)

    def retract(self, delta: np.ndarray) -> "BiasVariable":
        return BiasVariable(self.value + np.asarray(delta, dtype=float).reshape(6))

    def copy(self) -> "BiasVariable":
        return BiasVariable(self.value.copy())


class Values(dict):
    """A mapping from key to :class:`Variable`, with manifold arithmetic."""

    def retract(self, deltas: dict) -> "Values":
        out = Values()
        for key, var in self.items():
            d = deltas.get(key)
            out[key] = var.retract(d) if d is not None else var.copy()
        return out

    def copy(self) -> "Values":
        return Values({k: v.copy() for k, v in self.items()})

    def pose(self, key) -> Pose:
        return self[key].value

    def vector(self, key) -> np.ndarray:
        return self[key].value

    def bias(self, key) -> ImuBias:
        return self[key].bias
