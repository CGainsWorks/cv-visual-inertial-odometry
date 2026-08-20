"""Analytic trajectories with exact derivatives.

The point of this module is that nothing in it is numerical.  Position,
velocity, acceleration, orientation and body angular rate all come out of
closed-form expressions, so when the preintegration tests compare against it
there is no integration error on the reference side to hide behind.  If the
comparison fails, the estimator is wrong.

Orientation is parameterised as ``R(t) = Exp(r(t))`` for a smooth rotation
vector ``r``.  The body angular rate then follows from the right Jacobian::

    omega_body(t) = Jr(r(t)) @ r_dot(t)

which is exact for any ``r``, including ones whose norm passes through pi.
That identity is checked directly in ``tests/test_trajectory.py`` against a
finite difference of ``R``, because it is the one step here that is not obvious
by inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Pose, so3


@dataclass
class Sinusoid:
    """``a * sin(w t + phi) + c`` and its first two derivatives."""

    amplitude: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    offset: np.ndarray

    def __post_init__(self) -> None:
        for name in ("amplitude", "omega", "phase", "offset"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float).reshape(3))

    def value(self, t: float) -> np.ndarray:
        return self.amplitude * np.sin(self.omega * t + self.phase) + self.offset

    def derivative(self, t: float) -> np.ndarray:
        return self.amplitude * self.omega * np.cos(self.omega * t + self.phase)

    def second_derivative(self, t: float) -> np.ndarray:
        return -self.amplitude * self.omega ** 2 * np.sin(self.omega * t + self.phase)


class Trajectory:
    """A smooth body trajectory: sinusoidal position, sinusoidal attitude.

    Parameters are deliberately incommensurate so the path does not close on
    itself every period, which would make a loop-closure test trivially easy
    and an initialisation test trivially hard (a perfectly periodic path leaves
    the accelerometer bias and gravity direction poorly separated).
    """

    def __init__(
        self,
        position: Sinusoid | None = None,
        rotation: Sinusoid | None = None,
        yaw_rate: float = 0.0,
    ) -> None:
        self.position = position or Sinusoid(
            amplitude=[3.0, 2.2, 0.45],
            omega=[2.0 * np.pi / 20.0, 2.0 * np.pi / 13.0, 2.0 * np.pi / 7.0],
            phase=[0.0, 0.7, 1.9],
            offset=[0.0, 0.0, 1.4],
        )
        self.rotation = rotation or Sinusoid(
            amplitude=[0.13, 0.10, 0.35],
            omega=[2.0 * np.pi / 17.0, 2.0 * np.pi / 13.0, 2.0 * np.pi / 23.0],
            phase=[0.4, 2.1, 0.0],
            offset=[0.0, 0.0, 0.0],
        )
        #: A constant yaw rate on top of the sinusoid, so the trajectory keeps
        #: turning rather than oscillating about one heading.  Set to zero for
        #: the tests that want a strictly bounded rotation vector.
        self.yaw_rate = float(yaw_rate)

    # -- translation -------------------------------------------------------
    def p(self, t: float) -> np.ndarray:
        return self.position.value(t)

    def v(self, t: float) -> np.ndarray:
        return self.position.derivative(t)

    def a(self, t: float) -> np.ndarray:
        """World-frame acceleration, gravity *not* included."""
        return self.position.second_derivative(t)

    # -- rotation ----------------------------------------------------------
    def rotvec(self, t: float) -> np.ndarray:
        r = self.rotation.value(t)
        if self.yaw_rate:
            r = r + np.array([0.0, 0.0, self.yaw_rate * t])
        return r

    def rotvec_dot(self, t: float) -> np.ndarray:
        d = self.rotation.derivative(t)
        if self.yaw_rate:
            d = d + np.array([0.0, 0.0, self.yaw_rate])
        return d

    def R(self, t: float) -> np.ndarray:
        return so3.exp(self.rotvec(t))

    def omega_body(self, t: float) -> np.ndarray:
        """Angular rate in the body frame, which is what a gyroscope reads."""
        return so3.right_jacobian(self.rotvec(t)) @ self.rotvec_dot(t)

    # -- combined ----------------------------------------------------------
    def pose(self, t: float) -> Pose:
        return Pose(self.R(t), self.p(t))

    def specific_force(self, t: float, gravity: np.ndarray) -> np.ndarray:
        """What an ideal, bias-free accelerometer reads at time ``t``.

        An accelerometer measures specific force, not acceleration: at rest it
        reads ``+9.81`` upwards, not zero.  Hence the subtraction rather than
        the addition that a first pass at this always writes.
        """
        return self.R(t).T @ (self.a(t) - np.asarray(gravity, dtype=float))

    def length(self, t0: float, t1: float, samples: int = 20000) -> float:
        """Path length, for the drift-per-metre figures."""
        ts = np.linspace(t0, t1, samples)
        ps = np.stack([self.p(float(t)) for t in ts])
        return float(np.linalg.norm(np.diff(ps, axis=0), axis=1).sum())
