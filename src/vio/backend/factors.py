"""Factors: a residual, the variables it touches, and its analytic Jacobians.

Every Jacobian here is written out by hand.  Every one of them is also checked
against a central finite difference of its own residual in
``tests/test_factors.py``, at randomised linearisation points, to a relative
tolerance of 1e-6.  That test is the reason to trust this file; the derivations
in the comments are the reason it was worth writing rather than differentiating
numerically at run time, which would be roughly thirty times slower for the IMU
factor's fifteen columns.

Conventions, all inherited from :mod:`vio.backend.variables`:

* poses retract as ``R <- R Exp(d_phi)``, ``p <- p + d_p``;
* residuals are returned *unwhitened*, and the solver applies
  ``sqrt_info`` and the robust kernel;
* the IMU residual orders its nine rows as ``[rotation, velocity, position]``,
  matching :class:`vio.imu.preintegration.Preintegrated`.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Pose, so3
from ..geometry.camera import PinholeCamera, projection_jacobian
from ..types import GRAVITY, ImuBias, NavState


class Loss:
    """Base robust kernel.  ``weight`` is applied to residual and Jacobian alike."""

    def weight(self, sq_norm: float) -> float:
        return 1.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}()"


class TrivialLoss(Loss):
    pass


class HuberLoss(Loss):
    """Quadratic inside ``delta``, linear outside.

    Implemented as iteratively reweighted least squares: the square root of
    ``rho'`` multiplies the whitened residual and its Jacobian, which is what
    makes a single Gauss-Newton step on the reweighted system equal a step on
    the robustified cost.  It is not exact, the second-order term of ``rho``
    is dropped, but dropping it is what keeps the Hessian positive definite,
    and the alternative (Triggs correction) is fragile in exactly the
    high-residual regime the kernel exists to handle.
    """

    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def weight(self, sq_norm: float) -> float:
        if sq_norm <= self.delta * self.delta:
            return 1.0
        return float(self.delta / np.sqrt(sq_norm))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"HuberLoss({self.delta})"


class CauchyLoss(Loss):
    """Redescending kernel, for loop-closure candidates that may be wrong."""

    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def weight(self, sq_norm: float) -> float:
        return float(1.0 / (1.0 + sq_norm / (self.delta * self.delta)))


class Factor:
    """Base class.  Subclasses provide ``residual`` and ``jacobians``."""

    def __init__(self, keys, dim: int, sqrt_info=None, loss: Loss | None = None):
        self.keys = list(keys)
        self.dim = int(dim)
        self.sqrt_info = (
            np.eye(self.dim) if sqrt_info is None else np.asarray(sqrt_info, dtype=float)
        )
        if self.sqrt_info.shape == (self.dim,):
            self.sqrt_info = np.diag(self.sqrt_info)
        self.loss = loss or TrivialLoss()

    def residual(self, values) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def jacobians(self, values) -> list:  # pragma: no cover - interface
        raise NotImplementedError

    def linearise(self, values):
        """Whitened residual, whitened Jacobians and the robust weight.

        Returns ``(r, [J...], cost)`` where ``cost`` is the robustified
        one-half-squared-norm actually being minimised, so the solver can
        compare iterations on the same quantity it is stepping on.
        """
        r = self.sqrt_info @ self.residual(values)
        Js = [self.sqrt_info @ J for J in self.jacobians(values)]
        sq = float(r @ r)
        w = self.loss.weight(sq)
        if w != 1.0:
            s = np.sqrt(w)
            r = r * s
            Js = [J * s for J in Js]
        # For Huber, w * sq is the correct robustified cost up to the constant
        # that the kernel subtracts; the constant does not affect the step.
        return r, Js, 0.5 * w * sq

    def error(self, values) -> float:
        r = self.sqrt_info @ self.residual(values)
        sq = float(r @ r)
        return 0.5 * self.loss.weight(sq) * sq


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


class PosePriorFactor(Factor):
    """Anchors a pose.  Residual is the retraction that would reach the target."""

    def __init__(self, key, target: Pose, sqrt_info=None, loss=None):
        super().__init__([key], 6, sqrt_info, loss)
        self.target = target.copy()

    def residual(self, values) -> np.ndarray:
        T = values[self.keys[0]].value
        return np.concatenate(
            [so3.log(self.target.R.T @ T.R), T.t - self.target.t]
        )

    def jacobians(self, values) -> list:
        T = values[self.keys[0]].value
        J = np.zeros((6, 6))
        # d/d_phi Log(Rt^T R Exp(d)) = Jr^-1(residual rotation)
        r_phi = so3.log(self.target.R.T @ T.R)
        J[:3, :3] = so3.right_jacobian_inv(r_phi)
        J[3:, 3:] = np.eye(3)
        return [J]


class VectorPriorFactor(Factor):
    """Anchors a Euclidean variable: a velocity, a bias, a time offset."""

    def __init__(self, key, target, sqrt_info=None, loss=None):
        target = np.atleast_1d(np.asarray(target, dtype=float))
        super().__init__([key], int(target.size), sqrt_info, loss)
        self.target = target

    def residual(self, values) -> np.ndarray:
        return values[self.keys[0]].value - self.target

    def jacobians(self, values) -> list:
        return [np.eye(self.dim)]


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


class ReprojectionFactor(Factor):
    """One landmark seen in one frame.

    The measurement is an *undistorted normalised* image coordinate, and the
    residual is scaled by the focal length so its units are pixels.  Keeping
    the residual in pixels matters for the robust kernel: a Huber threshold in
    normalised units would mean a different number of pixels on every camera.
    """

    def __init__(
        self,
        pose_key,
        point_key,
        measurement,
        camera: PinholeCamera,
        T_body_cam: Pose,
        pixel_sigma: float = 1.0,
        loss: Loss | None = None,
    ):
        super().__init__(
            [pose_key, point_key], 2, np.eye(2) / float(pixel_sigma), loss
        )
        self.measurement = np.asarray(measurement, dtype=float).reshape(2)
        self.camera = camera
        self.T_body_cam = T_body_cam.copy()
        self.T_cam_body = T_body_cam.inverse()
        self.focal = camera.focal

    def _camera_point(self, values):
        T_wb = values[self.keys[0]].value
        p_w = values[self.keys[1]].value
        p_b = T_wb.R.T @ (p_w - T_wb.t)
        p_c = self.T_cam_body.R @ p_b + self.T_cam_body.t
        return T_wb, p_b, p_c

    def residual(self, values) -> np.ndarray:
        _, _, p_c = self._camera_point(values)
        z = p_c[2]
        if z <= 1e-6:
            # Behind the camera.  Returning a large but finite residual keeps
            # the solve well posed; the robust kernel then caps its influence.
            # Raising or returning NaN here would poison the whole system for
            # one bad landmark.
            return np.full(2, 1e3)
        return self.focal * (p_c[:2] / z - self.measurement)

    def jacobians(self, values) -> list:
        T_wb, p_b, p_c = self._camera_point(values)
        if p_c[2] <= 1e-6:
            return [np.zeros((2, 6)), np.zeros((2, 3))]
        Jp = self.focal * projection_jacobian(p_c)
        R_cb = self.T_cam_body.R
        R_cw = R_cb @ T_wb.R.T

        J_pose = np.zeros((2, 6))
        # p_b = R_wb^T (p_w - t);  R_wb <- R_wb Exp(d) gives p_b + hat(p_b) d
        J_pose[:, :3] = Jp @ (R_cb @ so3.hat(p_b))
        J_pose[:, 3:] = Jp @ (-R_cw)
        J_point = Jp @ R_cw
        return [J_pose, J_point]


class ExtrinsicReprojectionFactor(ReprojectionFactor):
    """A reprojection factor that also optimises the camera-IMU extrinsic.

    Used only by :mod:`vio.calibration`.  The estimator's own factors hold the
    extrinsic fixed, because leaving six extra degrees of freedom free inside a
    sliding window makes the whole thing slower and less stable for no gain
    once the calibration has converged.
    """

    def __init__(self, pose_key, point_key, extrinsic_key, measurement, camera,
                 pixel_sigma=1.0, loss=None):
        super().__init__(
            pose_key, point_key, measurement, camera, Pose.identity(), pixel_sigma, loss
        )
        self.keys = [pose_key, point_key, extrinsic_key]

    def _camera_point(self, values):
        T_wb = values[self.keys[0]].value
        p_w = values[self.keys[1]].value
        T_bc = values[self.keys[2]].value
        p_b = T_wb.R.T @ (p_w - T_wb.t)
        T_cb = T_bc.inverse()
        p_c = T_cb.R @ p_b + T_cb.t
        return T_wb, p_b, p_c, T_bc

    def residual(self, values) -> np.ndarray:
        _, _, p_c, _ = self._camera_point(values)
        z = p_c[2]
        if z <= 1e-6:
            return np.full(2, 1e3)
        return self.focal * (p_c[:2] / z - self.measurement)

    def jacobians(self, values) -> list:
        T_wb, p_b, p_c, T_bc = self._camera_point(values)
        if p_c[2] <= 1e-6:
            return [np.zeros((2, 6)), np.zeros((2, 3)), np.zeros((2, 6))]
        Jp = self.focal * projection_jacobian(p_c)
        R_cb = T_bc.R.T
        R_cw = R_cb @ T_wb.R.T

        J_pose = np.zeros((2, 6))
        J_pose[:, :3] = Jp @ (R_cb @ so3.hat(p_b))
        J_pose[:, 3:] = Jp @ (-R_cw)
        J_point = Jp @ R_cw

        # p_c = R_bc^T (p_b - t_bc).  With R_bc <- R_bc Exp(d_phi) the first
        # term becomes Exp(-d_phi) R_bc^T (p_b - t_bc) = p_c + hat(p_c) d_phi,
        # and t_bc <- t_bc + d_t contributes -R_bc^T d_t.
        J_ext = np.zeros((2, 6))
        J_ext[:, :3] = Jp @ so3.hat(p_c)
        J_ext[:, 3:] = Jp @ (-R_cb)
        return [J_pose, J_point, J_ext]


# ---------------------------------------------------------------------------
# Inertial
# ---------------------------------------------------------------------------


class ImuFactor(Factor):
    """A preintegrated IMU block between two keyframes.

    Touches five variables: ``pose_i, vel_i, bias_i, pose_j, vel_j``.  The bias
    at ``j`` is tied to the bias at ``i`` by a separate
    :class:`BiasRandomWalkFactor`, which is what Forster does and what keeps
    this factor's Jacobian block structure manageable.
    """

    def __init__(
        self,
        pose_i,
        vel_i,
        bias_i,
        pose_j,
        vel_j,
        preintegrated,
        gravity=GRAVITY,
        loss: Loss | None = None,
    ):
        super().__init__(
            [pose_i, vel_i, bias_i, pose_j, vel_j],
            9,
            preintegrated.sqrt_information(),
            loss,
        )
        self.pre = preintegrated
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)

    def _unpack(self, values):
        T_i = values[self.keys[0]].value
        v_i = values[self.keys[1]].value
        b_i = ImuBias.from_vector(values[self.keys[2]].value)
        T_j = values[self.keys[3]].value
        v_j = values[self.keys[4]].value
        return T_i, v_i, b_i, T_j, v_j

    def residual(self, values) -> np.ndarray:
        T_i, v_i, b_i, T_j, v_j = self._unpack(values)
        return self.pre.residual(
            NavState(T_i.R, T_i.t, v_i), b_i, NavState(T_j.R, T_j.t, v_j), self.gravity
        )

    def jacobians(self, values) -> list:
        T_i, v_i, b_i, T_j, v_j = self._unpack(values)
        pre = self.pre
        dt = pre.dt
        g = self.gravity
        Rt = T_i.R.T

        db = b_i.vector() - pre.bias_ref.vector()
        dbg = db[3:]
        c = pre.J_R_bg @ dbg
        dR_corr = so3.project(pre.dR @ so3.exp(c))

        r = self.residual(values)
        r_R = r[:3]
        # Jl^-1(x) == Jr^-1(-x); used wherever a perturbation enters on the left.
        Jl_inv = so3.right_jacobian_inv(-r_R)
        Jr_inv = so3.right_jacobian_inv(r_R)

        dv_term = Rt @ (v_j - v_i - g * dt)
        dp_term = Rt @ (T_j.t - T_i.t - v_i * dt - 0.5 * g * dt * dt)

        J_pose_i = np.zeros((9, 6))
        J_pose_i[0:3, 0:3] = -Jl_inv @ dR_corr.T
        J_pose_i[3:6, 0:3] = so3.hat(dv_term)
        J_pose_i[6:9, 0:3] = so3.hat(dp_term)
        J_pose_i[6:9, 3:6] = -Rt

        J_vel_i = np.zeros((9, 3))
        J_vel_i[3:6, :] = -Rt
        J_vel_i[6:9, :] = -Rt * dt

        J_bias_i = np.zeros((9, 6))
        J_bias_i[0:3, 3:6] = -Jl_inv @ so3.right_jacobian(c) @ pre.J_R_bg
        J_bias_i[3:6, 0:3] = -pre.J_v_ba
        J_bias_i[3:6, 3:6] = -pre.J_v_bg
        J_bias_i[6:9, 0:3] = -pre.J_p_ba
        J_bias_i[6:9, 3:6] = -pre.J_p_bg

        J_pose_j = np.zeros((9, 6))
        J_pose_j[0:3, 0:3] = Jr_inv
        J_pose_j[6:9, 3:6] = Rt

        J_vel_j = np.zeros((9, 3))
        J_vel_j[3:6, :] = Rt

        return [J_pose_i, J_vel_i, J_bias_i, J_pose_j, J_vel_j]


#: Largest information a bias random-walk factor is allowed to assert, per
#: axis.  A datasheet random walk of exactly zero means "this bias never
#: changes", which is a perfectly reasonable thing to simulate and an infinite
#: number to put in a matrix.  Clamping keeps the constraint extremely stiff
#: without putting an inf into the Hessian, where it becomes a NaN one matrix
#: multiply later and silently poisons the whole solve.
MAX_BIAS_WALK_INFORMATION = 1e8


class BiasRandomWalkFactor(Factor):
    """Ties consecutive biases together with the datasheet random walk."""

    def __init__(self, bias_i, bias_j, dt: float, noise, loss=None):
        _, walk = noise.discrete(max(float(dt), 1e-6))
        variance = np.maximum(np.diag(walk), 1.0 / MAX_BIAS_WALK_INFORMATION ** 2)
        sqrt_info = np.diag(1.0 / np.sqrt(variance))
        super().__init__([bias_i, bias_j], 6, sqrt_info, loss)

    def residual(self, values) -> np.ndarray:
        return values[self.keys[1]].value - values[self.keys[0]].value

    def jacobians(self, values) -> list:
        return [-np.eye(6), np.eye(6)]


# ---------------------------------------------------------------------------
# Pose graph
# ---------------------------------------------------------------------------


class RelativePoseFactor(Factor):
    """A measured relative transform, used for loop closure and pose graphs.

    Residual is ``[Log(dR_meas^T R_i^T R_j), R_i^T (p_j - p_i) - dp_meas]``.
    The translation part is expressed in frame ``i``, which is where a loop
    closure's uncertainty actually lives, a relative measurement is uncertain
    along the direction the camera was pointing, not along the world axes.
    """

    def __init__(self, key_i, key_j, measured: Pose, sqrt_info=None, loss=None):
        super().__init__([key_i, key_j], 6, sqrt_info, loss)
        self.measured = measured.copy()

    def residual(self, values) -> np.ndarray:
        T_i = values[self.keys[0]].value
        T_j = values[self.keys[1]].value
        r_R = so3.log(self.measured.R.T @ T_i.R.T @ T_j.R)
        r_t = T_i.R.T @ (T_j.t - T_i.t) - self.measured.t
        return np.concatenate([r_R, r_t])

    def jacobians(self, values) -> list:
        T_i = values[self.keys[0]].value
        T_j = values[self.keys[1]].value
        r = self.residual(values)
        Jl_inv = so3.right_jacobian_inv(-r[:3])
        Jr_inv = so3.right_jacobian_inv(r[:3])
        Rt = T_i.R.T
        dR = self.measured.R

        J_i = np.zeros((6, 6))
        J_i[:3, :3] = -Jl_inv @ dR.T
        J_i[3:, :3] = so3.hat(Rt @ (T_j.t - T_i.t))
        J_i[3:, 3:] = -Rt

        J_j = np.zeros((6, 6))
        J_j[:3, :3] = Jr_inv
        J_j[3:, 3:] = Rt
        return [J_i, J_j]


class GravityDirectionFactor(Factor):
    """Pins a pose's roll and pitch while leaving its yaw free.

    A visual-inertial system observes gravity, so roll and pitch are
    absolutely referenced and do not drift; only position and yaw do.  A
    six-degree-of-freedom pose graph does not know that, and given a loop to
    close it will cheerfully tilt the entire map, trading a quantity it
    measures well for one it does not.

    The residual is the body-frame gravity direction against the one the
    estimator reported::

        r = R^T g_hat - R_ref^T g_hat

    which is rank two: rotating about gravity leaves it unchanged, so yaw is
    untouched.  That gives the four-degree-of-freedom behaviour VINS-Mono gets
    from a purpose-built parameterisation, without needing a new variable type
    or a second solver path.
    """

    def __init__(self, key, reference_rotation, sqrt_info=None, loss=None):
        super().__init__([key], 3, sqrt_info, loss)
        self.g_hat = GRAVITY / np.linalg.norm(GRAVITY)
        self.target = np.asarray(reference_rotation, dtype=float).T @ self.g_hat

    def residual(self, values) -> np.ndarray:
        R = values[self.keys[0]].value.R
        return R.T @ self.g_hat - self.target

    def jacobians(self, values) -> list:
        R = values[self.keys[0]].value.R
        J = np.zeros((3, 6))
        # R <- R Exp(d) gives Exp(-d) R^T g = R^T g + hat(R^T g) d
        J[:, :3] = so3.hat(R.T @ self.g_hat)
        return [J]


class LinearPriorFactor(Factor):
    """A dense Gaussian prior over several variables at once.

    This is what marginalisation produces: eliminating a state from the window
    leaves behind a linear constraint coupling everything that state touched.
    It is stored in square-root form and evaluated about the linearisation
    point it was built at, which is why ``linearisation_point`` is kept rather
    than re-read from the current values.
    """

    def __init__(self, keys, linearisation_point, sqrt_info, b, dims, loss=None):
        super().__init__(keys, int(np.asarray(b).size), np.eye(int(np.asarray(b).size)), loss)
        self.linearisation_point = linearisation_point
        self.A = np.asarray(sqrt_info, dtype=float)
        self.b = np.asarray(b, dtype=float).reshape(-1)
        self.dims = list(dims)
        self._offsets = np.cumsum([0] + self.dims)

    def _delta(self, values) -> np.ndarray:
        parts = []
        for key in self.keys:
            ref = self.linearisation_point[key]
            parts.append(ref.local(values[key]))
        return np.concatenate(parts) if parts else np.zeros(0)

    def residual(self, values) -> np.ndarray:
        return self.A @ self._delta(values) + self.b

    def jacobians(self, values) -> list:
        # A column block of A, composed with the derivative of the local
        # coordinates.  That second term is the identity for Euclidean
        # variables and Jr^-1 for the rotation part of a pose; leaving it out
        # is the common shortcut and it is measurably wrong once the estimate
        # has moved away from the point the prior was built at.
        out = []
        for i, key in enumerate(self.keys):
            block = self.A[:, self._offsets[i]: self._offsets[i + 1]]
            out.append(block @ self.linearisation_point[key].local_jacobian(values[key]))
        return out
