"""Visual-inertial initialisation: gyro bias, gravity, metric scale, velocities.

This is the step where from-scratch VIO usually dies, so it is worth being
explicit about what is and is not observable.

A monocular camera gives rotation *metrically*, rotations have no scale
ambiguity and translation only up to one global scale factor.  An IMU gives
metric acceleration but cannot separate a constant acceleration from gravity
without motion.  Put together over a window with genuine translational
excitation, the pair fixes four things that neither fixes alone: the gyroscope
bias, the direction of gravity, the metric scale and the velocity at every
keyframe.  Yaw about gravity stays unobservable forever, which is why the
result is rotated into a gravity-aligned frame with an arbitrary heading rather
than pretending to know north.

The procedure is the linear one from Qin, Li and Shen's VINS-Mono, rewritten
here from the relations rather than ported:

1. **Gyroscope bias** from rotation alone.  Minimise the disagreement between
   the preintegrated rotation and the visual one over every consecutive pair.
   This is a small Gauss-Newton on three unknowns and it converges in two
   iterations because the residual is nearly linear in the bias.

2. **Gravity, scale and velocities** from a single linear least-squares
   problem.  Substituting the up-to-scale visual positions into the
   preintegration relations for ``dp`` and ``dv`` leaves a system that is
   linear in ``[v_0 ... v_n, g, s]``.

3. **Gravity refinement** on the tangent plane, because step 2 does not know
   that gravity has a known magnitude.  Reparameterising the correction into
   the two directions perpendicular to the current estimate and re-solving two
   or three times enforces ``|g| = 9.81`` without a constrained solver.


4. **Accelerometer bias**, in the same linear system.  This was originally
   left out, on the standard-sounding argument that over a short window a bias
   and a tilt of gravity look almost the same, so solving for both would make
   the system badly conditioned rather than more informative.  Measuring it
   said otherwise, emphatically so.  With a 0.55 m/s^2 bias, which is what
   EuRoC's own ground truth reports for V1_01:

   ==============  ===============  =============  =============
   keyframes       solve for bias   scale error    gravity error
   ==============  ===============  =============  =============
   12              no                     61.0%          1.67 deg
   12              yes                     0.8%          0.19 deg
   20              no                      5.9%          3.22 deg
   20              yes                     1.2%          0.08 deg
   30              no                     47.6%          4.08 deg
   30              yes                     1.4%          0.15 deg
   ==============  ===============  =============  =============

   Those are with sensor noise and a drifting bias; noise-free the "yes" rows
   go to zero.  The condition number does rise, from about 200 to about 665 at
   twelve keyframes, so the worry was not baseless.  It was simply outweighed: a bias
   that is not solved for does not go away, it is absorbed by the scale and by
   the gravity direction and those are the two things the whole procedure
   exists to produce.  The estimate is discarded if it exceeds ``max_acc_bias``,
   on the grounds that no real accelerometer has a 1.5 m/s^2 bias and something
   else has been absorbed instead.

   ``scripts/ablate.py`` regenerates the table on the simulator, where the true
   bias is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose, so3
from ..types import GRAVITY, ImuBias

GRAVITY_MAGNITUDE = float(np.linalg.norm(GRAVITY))


@dataclass
class InitialisationResult:
    ok: bool
    reason: str = ""
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: Gravity expressed in the visual odometry's own frame.
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    #: World-frame velocity per keyframe, in the visual odometry's frame and
    #: already at metric scale.
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Accelerometer bias, zero unless it was solved for.
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: Rotation taking the visual odometry frame into a gravity-aligned one.
    R_align: np.ndarray = field(default_factory=lambda: np.eye(3))
    condition_number: float = float("inf")
    #: Average translational excitation over the window, m/s^2.  Reported
    #: because a window with none will produce a confident and wrong scale.
    excitation: float = 0.0


def estimate_gyro_bias(rotations, preintegrated, iterations: int = 4) -> np.ndarray:
    """Gyro bias from the mismatch between visual and inertial rotation.

    ``rotations`` are body-to-world rotation matrices, one per keyframe.
    ``preintegrated`` are the blocks joining consecutive keyframes, so there is
    one fewer of them.
    """
    bg = np.zeros(3)
    for _ in range(iterations):
        H = np.zeros((3, 3))
        b = np.zeros(3)
        for k, pre in enumerate(preintegrated):
            c = pre.J_R_bg @ (bg - pre.bias_ref.gyro)
            dR = pre.dR @ so3.exp(c)
            r = so3.log(dR.T @ rotations[k].T @ rotations[k + 1])
            # d r / d bg, matching ImuFactor's derivation.
            J = -so3.right_jacobian_inv(-r) @ so3.right_jacobian(c) @ pre.J_R_bg
            H += J.T @ J
            b += J.T @ r
        try:
            step = np.linalg.solve(H + np.eye(3) * 1e-12, -b)
        except np.linalg.LinAlgError:
            break
        bg = bg + step
        if np.linalg.norm(step) < 1e-10:
            break
    return bg


def translational_excitation(R_wb, preintegrated) -> float:
    """How much the platform accelerated, beyond falling, over a window.

    Scale is only observable if something other than gravity acted on the
    accelerometer.  With no translational excitation the scale column of the
    alignment system is nearly collinear with the gravity columns and the solve
    returns a confident, meaningless number, which is the failure that makes
    a VIO start with a trajectory ten times too big and never recover.

    The measure is the spread of the *world-frame* specific force across the
    window.  Rotating each block's ``dv / dt`` into the world frame turns pure
    gravity into a constant vector regardless of how the body was turning, so
    the standard deviation that remains is genuine acceleration.  Doing the
    same thing in the body frame, which is the obvious first attempt, mostly
    measures how much the platform rotated.

    Returns metres per second squared.
    """
    accels = []
    for R, pre in zip(R_wb, preintegrated):
        if pre.dt > 1e-6:
            accels.append(R @ (pre.dv / pre.dt))
    if len(accels) < 2:
        return 0.0
    A = np.stack(accels)
    return float(np.mean(np.linalg.norm(A - A.mean(axis=0), axis=1)))


def _tangent_basis(v: np.ndarray) -> np.ndarray:
    """Two unit vectors spanning the plane perpendicular to ``v``."""
    a = v / np.linalg.norm(v)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(a @ tmp)) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    b1 = tmp - a * float(a @ tmp)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(a, b1)
    return np.stack([b1, b2], axis=1)


def _linear_alignment(poses_wc, T_body_cam: Pose, preintegrated,
                      refine_gravity: bool, acc_bias: bool = False):
    """Solve for ``[v_0..v_n, g, s]`` and optionally the accelerometer bias.

    Adding the accelerometer bias costs three more columns and keeps the system
    linear, because the preintegrated deltas depend on it to first order
    through the Jacobians the block already carries::

        dp_ij(b_a) = dp_ij + J_p_ba db_a
        dv_ij(b_a) = dv_ij + J_v_ba db_a

    Whether it is worth solving for is an empirical question, not an obvious
    one: over a short window an accelerometer bias and a tilt of gravity look
    almost the same, so the extra columns can make the system worse conditioned
    rather than more informative.  ``scripts/ablate.py`` measures it and
    ``docs/results.md`` reports the answer.

    Returns ``(velocities, gravity, scale, bias, condition_number)`` or
    ``None``.
    """
    n = len(poses_wc)
    # T_wc = T_wb T_bc, so R_wb = R_wc R_bc^T and p_wb = p_wc - R_wb t_bc.
    # Getting this transpose the wrong way round produces a *negative* scale
    # whose magnitude drifts with window length, which is what it did.
    t_bc = T_body_cam.t
    R_wb = [T.R @ T_body_cam.R.T for T in poses_wc]

    gravity = None
    scale = 1.0
    velocities = np.zeros((n, 3))
    cond = float("inf")

    for stage in range(4 if refine_gravity else 1):
        # After the first pass the gravity magnitude is pinned and only a
        # two-dimensional correction in the tangent plane is solved for.
        constrained = stage > 0 and gravity is not None
        n_g = 2 if constrained else 3
        n_b = 3 if acc_bias else 0
        n_x = 3 * n + n_g + 1 + n_b
        A = np.zeros((6 * (n - 1), n_x))
        b = np.zeros(6 * (n - 1))
        if constrained:
            g_dir = gravity / np.linalg.norm(gravity)
            basis = _tangent_basis(g_dir)
            g0 = g_dir * GRAVITY_MAGNITUDE

        for k, pre in enumerate(preintegrated):
            dt = pre.dt
            row = 6 * k
            dp_world = R_wb[k] @ pre.dp
            dv_world = R_wb[k] @ pre.dv
            dp_cam = poses_wc[k + 1].t - poses_wc[k].t
            dt_extrinsic = (R_wb[k + 1] - R_wb[k]) @ t_bc

            # s * dp_cam - v_i dt - g dt^2/2 = R_wb_i dp_ij + (R_wb_j - R_wb_i) t_bc
            A[row: row + 3, 3 * k: 3 * k + 3] = -np.eye(3) * dt
            A[row: row + 3, 3 * n + n_g] = dp_cam
            rhs_p = dp_world + dt_extrinsic

            # R_wb_i dv_ij = v_j - v_i - g dt
            A[row + 3: row + 6, 3 * k: 3 * k + 3] = -np.eye(3)
            A[row + 3: row + 6, 3 * (k + 1): 3 * (k + 1) + 3] = np.eye(3)
            rhs_v = dv_world

            if constrained:
                A[row: row + 3, 3 * n: 3 * n + 2] = -0.5 * basis * dt * dt
                A[row + 3: row + 6, 3 * n: 3 * n + 2] = -basis * dt
                rhs_p = rhs_p + 0.5 * g0 * dt * dt
                rhs_v = rhs_v + g0 * dt
            else:
                A[row: row + 3, 3 * n: 3 * n + 3] = -0.5 * np.eye(3) * dt * dt
                A[row + 3: row + 6, 3 * n: 3 * n + 3] = -np.eye(3) * dt

            if acc_bias:
                # Moving the bias moves the deltas, which sit on the
                # right-hand side, so these columns enter with a plus sign.
                start = 3 * n + n_g + 1
                A[row: row + 3, start: start + 3] = -R_wb[k] @ pre.J_p_ba
                A[row + 3: row + 6, start: start + 3] = -R_wb[k] @ pre.J_v_ba

            b[row: row + 3] = rhs_p
            b[row + 3: row + 6] = rhs_v

        try:
            x, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if rank < n_x:
            return None
        cond = float(sv[0] / max(sv[-1], 1e-300))

        velocities = x[: 3 * n].reshape(n, 3)
        if constrained:
            gravity = g0 + basis @ x[3 * n: 3 * n + 2]
            gravity = gravity / np.linalg.norm(gravity) * GRAVITY_MAGNITUDE
        else:
            gravity = x[3 * n: 3 * n + 3]
        scale = float(x[3 * n + n_g])
        bias = x[3 * n + n_g + 1: 3 * n + n_g + 4] if acc_bias else np.zeros(3)

    return velocities, gravity, scale, bias, cond


def initialise(
    poses_wc,
    preintegrated,
    T_body_cam: Pose,
    min_scale: float = 1e-3,
    max_gravity_error: float = 1.0,
    min_excitation: float = 0.10,
    estimate_acc_bias: bool = False,
    max_acc_bias: float = 1.5,
) -> InitialisationResult:
    """Recover bias, gravity, scale and velocities from a bootstrap window.

    ``poses_wc`` are the up-to-scale camera poses from visual odometry, in an
    arbitrary frame.  ``preintegrated`` are the IMU blocks joining consecutive
    entries, integrated at *zero* bias, the routine repropagates them itself
    once it knows the gyro bias.
    """
    n = len(poses_wc)
    if n < 4 or len(preintegrated) != n - 1:
        return InitialisationResult(False, "need at least four keyframes")

    R_wb = [T.R @ T_body_cam.R.T for T in poses_wc]
    bg = estimate_gyro_bias(R_wb, preintegrated)
    blocks = [p.repropagate(ImuBias(np.zeros(3), bg)) for p in preintegrated]

    excitation = translational_excitation(R_wb, blocks)

    solved = _linear_alignment(poses_wc, T_body_cam, blocks, refine_gravity=True,
                               acc_bias=estimate_acc_bias)
    if solved is None:
        return InitialisationResult(False, "alignment system is rank deficient",
                                    gyro_bias=bg, excitation=excitation)
    velocities, gravity, scale, acc_bias, cond = solved
    # A bias larger than any real accelerometer's means the extra columns have
    # absorbed something else, most likely a gravity tilt, so it is discarded
    # rather than propagated into the estimator.
    if np.linalg.norm(acc_bias) > max_acc_bias:
        acc_bias = np.zeros(3)

    if scale < min_scale:
        return InitialisationResult(False, f"non-positive scale {scale:.4g}",
                                    gyro_bias=bg, scale=scale, excitation=excitation)
    g_err = abs(float(np.linalg.norm(gravity)) - GRAVITY_MAGNITUDE)
    if g_err > max_gravity_error:
        return InitialisationResult(
            False, f"gravity magnitude off by {g_err:.2f} m/s^2",
            gyro_bias=bg, gravity=gravity, scale=scale, excitation=excitation)
    if excitation < min_excitation:
        return InitialisationResult(
            False, f"insufficient excitation ({excitation:.3f} m/s^2)",
            gyro_bias=bg, gravity=gravity, scale=scale, excitation=excitation)

    R_align = align_to_gravity(gravity)
    return InitialisationResult(
        ok=True,
        gyro_bias=bg,
        gravity=gravity,
        scale=scale,
        velocities=velocities,
        acc_bias=acc_bias,
        R_align=R_align,
        condition_number=cond,
        excitation=excitation,
    )


def align_to_gravity(gravity: np.ndarray) -> np.ndarray:
    """Rotation taking a frame in which gravity is ``gravity`` into a ``z``-up one.

    Yaw is unobservable, so the rotation chosen is the minimal one, the
    geodesic that takes the measured gravity direction onto ``-z`` and moves
    nothing else.  Any other choice would silently invent a heading.
    """
    g = np.asarray(gravity, dtype=float)
    a = g / np.linalg.norm(g)
    target = GRAVITY / GRAVITY_MAGNITUDE
    v = np.cross(a, target)
    s = float(np.linalg.norm(v))
    c = float(a @ target)
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # Antipodal: a half turn about any axis *perpendicular* to a.  Using an
        # axis parallel to a instead, which is the natural-looking mistake --
        # gives a reflection composed with a rotation and leaves the vector
        # pointing the wrong way.
        n = _perpendicular(a)
        return 2.0 * np.outer(n, n) - np.eye(3)
    return so3.exp(v / s * np.arctan2(s, c))


def _perpendicular(a: np.ndarray) -> np.ndarray:
    """Any unit vector orthogonal to ``a``."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(a @ seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    n = seed - a * float(a @ seed)
    return n / np.linalg.norm(n)
