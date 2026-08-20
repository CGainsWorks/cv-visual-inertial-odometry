"""On-manifold IMU preintegration.

This is the piece that makes visual-inertial odometry tractable.  Between two
keyframes there are typically ten to fifty IMU samples; re-integrating them
every time the optimiser nudges the pose at either end would dominate the cost
of the solve.  Preintegration collapses them into three quantities that depend
only on the measurements and on the *bias*, not on the states::

    dR_ij  ~  R_i^T R_j
    dv_ij  ~  R_i^T (v_j - v_i - g dt)
    dp_ij  ~  R_i^T (p_j - p_i - v_i dt - g dt^2 / 2)

together with a 9x9 covariance and a first-order correction for how they move
when the bias estimate changes.  Following Forster, Carlone, Dellaert and
Scaramuzza, *On-Manifold Preintegration for Real-Time Visual-Inertial
Odometry*, T-RO 2017, with right perturbation throughout to match
:mod:`vio.geometry.so3`.

Two integration schemes are implemented, and the difference between them is
measured rather than asserted: ``scripts/ablate.py --only integration``
regenerates the table in ``docs/results.md``.

``euler``
    The scheme written in the paper: every interval uses the measurement at its
    start.  First-order accurate in ``dt``.

``midpoint``
    Averages the *rotated* accelerations at the two ends of the interval and
    turns through the averaged gyro reading.  Second-order accurate.  This is
    the default.

The trap in the midpoint scheme is worth stating, because the first version of
this file fell into it: averaging the raw accelerometer readings while still
rotating them by the start-of-interval attitude makes velocity *worse* than
plain Euler, not better.  The two accelerations have to be rotated into the
reference frame separately, by the attitude at their own end of the interval,
before being averaged.  The bias Jacobians below are derived for that scheme
rather than borrowed from the Euler one, and ``tests/test_preintegration.py``
checks them against a finite difference of a full re-integration.

Bias handling deserves a note too.  The deltas are integrated once at a
*reference* bias and corrected to first order as the estimate moves.  That
correction is only good for small changes, so :meth:`Preintegrated.repropagate`
exists and the back end calls it when the bias has moved far enough to matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import so3
from ..types import GRAVITY, ImuBias, ImuData, ImuNoise, NavState

#: Beyond this much change from the reference bias, the first-order correction
#: is not trusted and the block is re-integrated.  Ordering is
#: ``[acc (3), gyro (3)]``.  0.01 rad/s of gyro bias over a 0.1 s interval is
#: 1 mrad of rotation, which is already past where linearising is free.
REPROPAGATE_THRESHOLD = np.array([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])


@dataclass
class Preintegrated:
    """Accumulated IMU deltas between two instants, at a fixed reference bias."""

    bias_ref: ImuBias = field(default_factory=ImuBias)
    noise: ImuNoise = field(default_factory=ImuNoise)
    scheme: str = "midpoint"

    dR: np.ndarray = field(default_factory=lambda: np.eye(3))
    dv: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dt: float = 0.0

    #: Covariance of ``[d_phi, d_v, d_p]``, in that order.
    cov: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))

    # First-order bias Jacobians.  ``J_v_ba`` is d(dv)/d(b_acc), and so on.
    J_R_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    #: The per-interval endpoint measurements actually used, kept so the block
    #: can be re-integrated at a new reference bias without going back to the
    #: raw stream.  This is what makes repropagation cheap and a block
    #: self-contained.
    steps: list = field(default_factory=list)
    span: tuple = (0.0, 0.0)

    n_samples: int = 0

    # -- accumulation ------------------------------------------------------
    def integrate_step(
        self,
        dt: float,
        acc0: np.ndarray,
        gyro0: np.ndarray,
        acc1: np.ndarray | None = None,
        gyro1: np.ndarray | None = None,
        record: bool = True,
    ) -> None:
        """Advance by one interval of length ``dt``.

        ``acc0``/``gyro0`` are the raw measurements at the start of the
        interval and ``acc1``/``gyro1`` those at the end.  Passing the same
        values for both, which is what the ``euler`` scheme does, reduces this
        to the propagation written in the paper.
        """
        if dt <= 0.0:
            return
        acc0 = np.asarray(acc0, dtype=float).reshape(3)
        gyro0 = np.asarray(gyro0, dtype=float).reshape(3)
        acc1 = acc0 if acc1 is None else np.asarray(acc1, dtype=float).reshape(3)
        gyro1 = gyro0 if gyro1 is None else np.asarray(gyro1, dtype=float).reshape(3)
        if record:
            self.steps.append((dt, acc0, gyro0, acc1, gyro1))

        a0 = acc0 - self.bias_ref.acc
        a1 = acc1 - self.bias_ref.acc
        w = 0.5 * (gyro0 + gyro1) - self.bias_ref.gyro

        phi = w * dt
        dR_k = so3.exp(phi)
        Jr_k = so3.right_jacobian(phi)

        R0 = self.dR
        R1 = R0 @ dR_k
        # The two accelerations, each rotated by the attitude at its own end of
        # the interval, then averaged.  Rotating both by R0 is the mistake.
        a_bar = 0.5 * (R0 @ a0 + R1 @ a1)

        # -- covariance ----------------------------------------------------
        # Propagated with the Forster structure evaluated at the averaged
        # measurement.  The midpoint rule reduces the *variance* of the
        # effective interval measurement, and consecutive intervals share an
        # endpoint sample so that reduction is correlated; both effects are
        # ignored here, which leaves the covariance mildly conservative.  How
        # mildly is measured against a Monte Carlo in
        # ``tests/test_preintegration.py`` rather than guessed at.
        a_hat_bar = so3.hat(0.5 * (a0 + a1))
        R_a_hat = R0 @ a_hat_bar
        A = np.eye(9)
        A[0:3, 0:3] = dR_k.T
        A[3:6, 0:3] = -R_a_hat * dt
        A[6:9, 0:3] = -0.5 * R_a_hat * dt * dt
        A[6:9, 3:6] = np.eye(3) * dt

        # Noise ordering is [acc, gyro], matching ImuNoise.discrete().
        B = np.zeros((9, 6))
        B[0:3, 3:6] = Jr_k * dt
        B[3:6, 0:3] = R0 * dt
        B[6:9, 0:3] = 0.5 * R0 * dt * dt
        meas_cov, _ = self.noise.discrete(dt)
        self.cov = A @ self.cov @ A.T + B @ meas_cov @ B.T

        # -- bias Jacobians ------------------------------------------------
        # d(a_bar)/d(b_gyro) and d(a_bar)/d(b_acc) for the midpoint update.
        # Perturbing b_gyro by delta rotates R0 by Exp(J_R_bg delta) and turns
        # dR_k by Exp(-Jr_k delta dt), so both terms appear.
        d_abar_dbg = -0.5 * R0 @ so3.hat(a0 + dR_k @ a1) @ self.J_R_bg
        d_abar_dbg += 0.5 * R1 @ so3.hat(a1) @ Jr_k * dt
        d_abar_dba = -0.5 * (R0 + R1)

        # Position uses the *previous* velocity Jacobian, so these four lines
        # are order dependent.
        self.J_p_ba = self.J_p_ba + self.J_v_ba * dt + 0.5 * d_abar_dba * dt * dt
        self.J_p_bg = self.J_p_bg + self.J_v_bg * dt + 0.5 * d_abar_dbg * dt * dt
        self.J_v_ba = self.J_v_ba + d_abar_dba * dt
        self.J_v_bg = self.J_v_bg + d_abar_dbg * dt
        self.J_R_bg = dR_k.T @ self.J_R_bg - Jr_k * dt

        # -- the deltas ----------------------------------------------------
        self.dp = self.dp + self.dv * dt + 0.5 * a_bar * dt * dt
        self.dv = self.dv + a_bar * dt
        self.dR = so3.project(R1)
        self.dt += dt
        self.n_samples += 1

    def integrate(self, imu: ImuData, t0: float, t1: float) -> "Preintegrated":
        """Accumulate every sample between ``t0`` and ``t1``.

        The first and last intervals are partial: an image timestamp does not
        land on an IMU sample, so the measurements are interpolated to the
        exact boundary rather than the interval being rounded to the nearest
        sample.  At 200 Hz that is up to 5 ms of motion per keyframe, and it
        does not average out, it biases every interval the same way.
        """
        t = np.asarray(imu.t, dtype=float)
        if len(t) < 2 or t1 <= t0:
            return self

        acc, gyro = imu.acc, imu.gyro
        # searchsorted rather than a boolean mask: the mask scans the whole
        # stream on every call, which is invisible for a 50 ms block and is not
        # for a long one on a 30000-sample sequence.
        lo = int(np.searchsorted(t, t0, side="right"))
        hi = int(np.searchsorted(t, t1, side="left"))
        edges = np.concatenate([[t0], t[lo:hi], [t1]])
        self.span = (float(t0), float(t1))

        for k in range(len(edges) - 1):
            ta, tb = float(edges[k]), float(edges[k + 1])
            if tb - ta <= 1e-12:
                continue
            a0, w0 = _interp(t, acc, ta), _interp(t, gyro, ta)
            if self.scheme == "midpoint":
                a1, w1 = _interp(t, acc, tb), _interp(t, gyro, tb)
            else:
                a1, w1 = a0, w0
            self.integrate_step(tb - ta, a0, w0, a1, w1)
        return self

    # -- use ---------------------------------------------------------------
    def corrected(self, bias: ImuBias):
        """The deltas moved to a new bias by the first-order correction."""
        db = np.asarray(bias.vector() - self.bias_ref.vector(), dtype=float)
        dba, dbg = db[:3], db[3:]
        dR = so3.project(self.dR @ so3.exp(self.J_R_bg @ dbg))
        dv = self.dv + self.J_v_ba @ dba + self.J_v_bg @ dbg
        dp = self.dp + self.J_p_ba @ dba + self.J_p_bg @ dbg
        return dR, dv, dp

    def predict(
        self, state_i: NavState, bias: ImuBias | None = None, gravity=GRAVITY
    ) -> NavState:
        """Propagate a state forward through this block."""
        if bias is None:
            dR, dv, dp = self.dR, self.dv, self.dp
        else:
            dR, dv, dp = self.corrected(bias)
        g = np.asarray(gravity, dtype=float)
        R_j = so3.project(state_i.R @ dR)
        v_j = state_i.v + g * self.dt + state_i.R @ dv
        p_j = (
            state_i.p
            + state_i.v * self.dt
            + 0.5 * g * self.dt * self.dt
            + state_i.R @ dp
        )
        return NavState(R_j, p_j, v_j)

    def residual(
        self,
        state_i: NavState,
        bias_i: ImuBias,
        state_j: NavState,
        gravity=GRAVITY,
    ) -> np.ndarray:
        """The 9-vector ``[r_R, r_v, r_p]`` the back end minimises."""
        dR, dv, dp = self.corrected(bias_i)
        g = np.asarray(gravity, dtype=float)
        Rt = state_i.R.T
        r_R = so3.log(dR.T @ Rt @ state_j.R)
        r_v = Rt @ (state_j.v - state_i.v - g * self.dt) - dv
        r_p = (
            Rt
            @ (
                state_j.p
                - state_i.p
                - state_i.v * self.dt
                - 0.5 * g * self.dt * self.dt
            )
            - dp
        )
        return np.concatenate([r_R, r_v, r_p])

    def needs_repropagation(self, bias: ImuBias) -> bool:
        delta = np.abs(bias.vector() - self.bias_ref.vector())
        return bool(np.any(delta > REPROPAGATE_THRESHOLD))

    def repropagate(self, bias: ImuBias) -> "Preintegrated":
        """Re-integrate the stored intervals at a new reference bias.

        A new block is returned rather than the existing one mutated, so a
        half-updated object can never reach the solver.
        """
        out = Preintegrated(bias_ref=bias.copy(), noise=self.noise, scheme=self.scheme)
        out.span = self.span
        out.steps = self.steps
        for dt, a0, w0, a1, w1 in self.steps:
            out.integrate_step(dt, a0, w0, a1, w1, record=False)
        return out

    # -- weighting ---------------------------------------------------------
    def information(self) -> np.ndarray:
        """Inverse covariance, ridged so a one-sample block cannot blow up."""
        C = self.cov + np.eye(9) * 1e-12
        return np.linalg.inv(C)

    def sqrt_information(self) -> np.ndarray:
        """Upper-triangular ``L`` with ``L.T @ L == information()``."""
        return np.linalg.cholesky(self.information()).T

    def bias_walk_covariance(self) -> np.ndarray:
        """Covariance of the bias change across this block, for the walk factor."""
        _, walk = self.noise.discrete(max(self.dt, 1e-9))
        return walk


def _interp(t: np.ndarray, values: np.ndarray, query: float) -> np.ndarray:
    """Linear interpolation of a 3-vector series, clamped at the ends."""
    if query <= t[0]:
        return values[0].copy()
    if query >= t[-1]:
        return values[-1].copy()
    k = int(np.searchsorted(t, query, side="right")) - 1
    k = min(max(k, 0), len(t) - 2)
    span = t[k + 1] - t[k]
    if span <= 0.0:
        return values[k].copy()
    w = (query - t[k]) / span
    return (1.0 - w) * values[k] + w * values[k + 1]


def preintegrate(
    imu: ImuData,
    t0: float,
    t1: float,
    bias: ImuBias | None = None,
    noise: ImuNoise | None = None,
    scheme: str = "midpoint",
) -> Preintegrated:
    """Convenience wrapper: build a block covering ``[t0, t1]``."""
    pre = Preintegrated(
        bias_ref=(bias or ImuBias()).copy(),
        noise=noise or ImuNoise(),
        scheme=scheme,
    )
    return pre.integrate(imu, t0, t1)
