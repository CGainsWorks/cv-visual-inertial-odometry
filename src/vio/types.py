"""The handful of dataclasses that every layer passes about.

They live here rather than in the module that happens to define them first so
that ``vio.sim`` can build an IMU stream without importing the estimator, and
the estimator can consume one without importing the simulator.  That separation
is what lets the preintegration tests compare against an independent generator
instead of against themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Pose

#: World gravity.  ``z`` is up, so gravity points down, and this is the value
#: EuRoC was recorded under (Zurich, ~9.81 m/s^2).  It is a module constant
#: rather than a magic number scattered through the preintegration because the
#: sign convention is the single easiest thing to get wrong in a VIO.
GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass
class ImuBias:
    """Accelerometer and gyroscope biases, both in the body frame."""

    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.acc = np.asarray(self.acc, dtype=float).reshape(3)
        self.gyro = np.asarray(self.gyro, dtype=float).reshape(3)

    def vector(self) -> np.ndarray:
        """``[acc (3), gyro (3)]``, the ordering the optimiser uses."""
        return np.concatenate([self.acc, self.gyro])

    @staticmethod
    def from_vector(v: np.ndarray) -> "ImuBias":
        v = np.asarray(v, dtype=float).reshape(6)
        return ImuBias(v[:3], v[3:])

    def __sub__(self, other: "ImuBias") -> np.ndarray:
        return self.vector() - other.vector()

    def copy(self) -> "ImuBias":
        return ImuBias(self.acc.copy(), self.gyro.copy())


@dataclass
class NavState:
    """Rotation, position and velocity at one instant, all in the world frame."""

    R: np.ndarray
    p: np.ndarray
    v: np.ndarray

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.p = np.asarray(self.p, dtype=float).reshape(3)
        self.v = np.asarray(self.v, dtype=float).reshape(3)

    @property
    def pose(self) -> Pose:
        return Pose(self.R, self.p)

    def copy(self) -> "NavState":
        return NavState(self.R.copy(), self.p.copy(), self.v.copy())


@dataclass
class ImuNoise:
    """Continuous-time IMU noise densities, in the units the datasets publish.

    EuRoC's ``imu0/sensor.yaml`` gives exactly these four numbers, so the
    loader can fill this in without anyone typing a constant.  The defaults are
    the EuRoC ADIS16448 values.

    Attributes
    ----------
    acc_noise:
        Accelerometer white noise, m/s^2 per sqrt(Hz).
    gyro_noise:
        Gyroscope white noise, rad/s per sqrt(Hz).
    acc_walk:
        Accelerometer bias random walk, m/s^3 per sqrt(Hz).
    gyro_walk:
        Gyroscope bias random walk, rad/s^2 per sqrt(Hz).
    """

    acc_noise: float = 2.0e-3
    gyro_noise: float = 1.6968e-4
    acc_walk: float = 3.0e-3
    gyro_walk: float = 1.9393e-5

    def discrete(self, dt: float) -> "tuple[np.ndarray, np.ndarray]":
        """Discrete-time covariances for one interval.

        White noise integrates as ``sigma^2 / dt`` and the random walk as
        ``sigma^2 * dt``.  Getting these the wrong way round produces a filter
        that looks fine on short sequences and diverges on long ones, so both
        directions are checked in ``tests/test_preintegration.py``.
        """
        meas = np.diag(
            np.concatenate(
                [
                    np.full(3, self.acc_noise ** 2 / dt),
                    np.full(3, self.gyro_noise ** 2 / dt),
                ]
            )
        )
        walk = np.diag(
            np.concatenate(
                [
                    np.full(3, self.acc_walk ** 2 * dt),
                    np.full(3, self.gyro_walk ** 2 * dt),
                ]
            )
        )
        return meas, walk


@dataclass
class ImuData:
    """A block of IMU samples, ordered by time.

    Stored column-wise rather than as a list of records because every consumer
    wants whole arrays: the preintegration walks them in a loop, the
    initialiser takes means over them, and the calibration shifts the timestamps
    wholesale.
    """

    t: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.acc = np.asarray(self.acc, dtype=float).reshape(-1, 3)
        self.gyro = np.asarray(self.gyro, dtype=float).reshape(-1, 3)
        if not (len(self.t) == len(self.acc) == len(self.gyro)):
            raise ValueError("IMU arrays disagree in length")

    def __len__(self) -> int:
        return len(self.t)

    def between(self, t0: float, t1: float) -> "ImuData":
        """Samples in ``[t0, t1)``, plus the sample bracketing each end.

        Returning the bracketing samples matters: an image timestamp almost
        never lands exactly on an IMU sample, and dropping the partial interval
        at each end loses up to 5 ms of motion per keyframe at 200 Hz, which
        integrates into a real bias.
        """
        i0 = int(np.searchsorted(self.t, t0, side="right")) - 1
        i1 = int(np.searchsorted(self.t, t1, side="left")) + 1
        i0 = max(i0, 0)
        i1 = min(i1, len(self.t))
        return ImuData(self.t[i0:i1], self.acc[i0:i1], self.gyro[i0:i1])
