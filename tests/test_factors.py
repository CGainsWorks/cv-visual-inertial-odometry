"""Every analytic Jacobian against a central finite difference.

This is the test that makes the back end believable.  A Jacobian that is
almost right produces an optimiser that almost converges, and the symptom --
slow convergence, or a solution that is subtly biased, looks exactly like a
tuning problem.  Checking each one against a finite difference of its own
residual, at randomised linearisation points, is cheap and conclusive.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_jacobians_match
from vio.backend import factors as F
from vio.backend.variables import (
    BiasVariable,
    PointVariable,
    PoseVariable,
    Values,
    VectorVariable,
)
from vio.geometry import Pose, se3, so3
from vio.geometry.camera import PinholeCamera
from vio.imu.preintegration import preintegrate
from vio.types import GRAVITY, ImuBias, ImuNoise

CAMERA = PinholeCamera(
    752, 480, 458.654, 457.296, 367.215, 248.375,
    [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], "radtan",
)
T_BODY_CAM = Pose(
    se3.exp(np.array([0.1, -0.2, 1.5, 0.0, 0.0, 0.0])).R,
    np.array([0.02, -0.06, 0.01]),
)


def _values(rng):
    v = Values()
    v["x0"] = PoseVariable(se3.exp(rng.normal(0, 0.6, 6)))
    v["x1"] = PoseVariable(se3.exp(rng.normal(0, 0.6, 6)))
    v["v0"] = VectorVariable(rng.normal(0, 0.5, 3))
    v["v1"] = VectorVariable(rng.normal(0, 0.5, 3))
    v["b0"] = BiasVariable(rng.normal(0, 0.02, 6))
    v["b1"] = BiasVariable(rng.normal(0, 0.02, 6))
    v["ext"] = PoseVariable(T_BODY_CAM)
    # A landmark that is actually in front of the camera at x0.
    T_wc = v["x0"].value * T_BODY_CAM
    v["l0"] = PointVariable(T_wc.act(np.array([0.3, -0.2, 5.0])))
    return v


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_pose_prior_jacobian(seed):
    rng = np.random.default_rng(seed)
    v = _values(rng)
    f = F.PosePriorFactor("x0", se3.exp(rng.normal(0, 0.5, 6)))
    assert_jacobians_match(f, v)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vector_prior_jacobian(seed):
    rng = np.random.default_rng(seed)
    v = _values(rng)
    assert_jacobians_match(F.VectorPriorFactor("v0", rng.normal(0, 1, 3)), v)
    assert_jacobians_match(F.VectorPriorFactor("b0", rng.normal(0, 0.05, 6)), v)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_reprojection_jacobian(seed):
    rng = np.random.default_rng(seed)
    v = _values(rng)
    f = F.ReprojectionFactor(
        "x0", "l0", rng.normal(0, 0.1, 2), CAMERA, T_BODY_CAM, 1.0
    )
    assert_jacobians_match(f, v)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_extrinsic_reprojection_jacobian(seed):
    rng = np.random.default_rng(seed)
    v = _values(rng)
    f = F.ExtrinsicReprojectionFactor(
        "x0", "l0", "ext", rng.normal(0, 0.1, 2), CAMERA, 1.0
    )
    assert_jacobians_match(f, v)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_relative_pose_jacobian(seed):
    rng = np.random.default_rng(seed)
    v = _values(rng)
    f = F.RelativePoseFactor("x0", "x1", se3.exp(rng.normal(0, 0.5, 6)))
    assert_jacobians_match(f, v)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_imu_factor_jacobian(seed, biased_sequence):
    rng = np.random.default_rng(seed)
    v = _values(rng)
    pre = preintegrate(
        biased_sequence.imu, 1.0, 1.15,
        bias=ImuBias([0.01, 0.02, -0.01], [0.001, -0.002, 0.0015]),
        noise=ImuNoise(),
    )
    f = F.ImuFactor("x0", "v0", "b0", "x1", "v1", pre)
    assert_jacobians_match(f, v)


@pytest.mark.parametrize("seed", [0, 1])
def test_bias_random_walk_jacobian(seed):
    rng = np.random.default_rng(seed)
    v = _values(rng)
    assert_jacobians_match(F.BiasRandomWalkFactor("b0", "b1", 0.15, ImuNoise()), v)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gravity_direction_jacobian(seed):
    rng = np.random.default_rng(seed)
    v = _values(rng)
    f = F.GravityDirectionFactor("x0", so3.exp(rng.normal(0, 0.4, 3)))
    assert_jacobians_match(f, v)


def test_gravity_factor_is_blind_to_yaw():
    """Rotating about gravity must leave the residual unchanged."""
    rng = np.random.default_rng(9)
    R = so3.exp(rng.normal(0, 0.3, 3))
    v = Values({"x0": PoseVariable(Pose(R, np.zeros(3)))})
    f = F.GravityDirectionFactor("x0", R)
    assert np.abs(f.residual(v)).max() < 1e-14
    g_hat = GRAVITY / np.linalg.norm(GRAVITY)
    for angle in (0.1, 0.7, 2.0):
        yawed = so3.exp(g_hat * angle) @ R
        v2 = Values({"x0": PoseVariable(Pose(yawed, np.zeros(3)))})
        assert np.abs(f.residual(v2)).max() < 1e-12
    # ...but a tilt must be seen.
    tilt = R @ so3.exp(np.array([0.05, 0.0, 0.0]))
    v3 = Values({"x0": PoseVariable(Pose(tilt, np.zeros(3)))})
    assert np.linalg.norm(f.residual(v3)) > 0.04


def test_reprojection_behind_camera_is_capped_not_infinite():
    rng = np.random.default_rng(3)
    v = _values(rng)
    T_wc = v["x0"].value * T_BODY_CAM
    v["l0"] = PointVariable(T_wc.act(np.array([0.0, 0.0, -3.0])))
    f = F.ReprojectionFactor("x0", "l0", np.zeros(2), CAMERA, T_BODY_CAM)
    r = f.residual(v)
    assert np.all(np.isfinite(r))
    assert np.allclose(r, 1e3)
    assert all(np.allclose(J, 0.0) for J in f.jacobians(v))


def test_huber_weight_shape():
    loss = F.HuberLoss(2.0)
    assert loss.weight(1.0) == 1.0
    assert loss.weight(4.0) == pytest.approx(1.0)
    assert loss.weight(100.0) == pytest.approx(2.0 / 10.0)
    # The robust cost must be monotone and below the quadratic one.
    for sq in (0.5, 4.0, 16.0, 400.0):
        assert loss.weight(sq) * sq <= sq + 1e-12


def test_cauchy_redescends():
    loss = F.CauchyLoss(1.0)
    assert loss.weight(0.0) == 1.0
    assert loss.weight(1e6) < 1e-5


def test_imu_residual_vanishes_on_truth(clean_sequence):
    seq = clean_sequence
    pre = preintegrate(seq.imu, 1.0, 1.2, noise=ImuNoise())
    s0, s1 = seq.state_at(1.0), seq.state_at(1.2)
    v = Values({
        "x0": PoseVariable(s0.pose), "v0": VectorVariable(s0.v),
        "b0": BiasVariable(np.zeros(6)),
        "x1": PoseVariable(s1.pose), "v1": VectorVariable(s1.v),
    })
    f = F.ImuFactor("x0", "v0", "b0", "x1", "v1", pre)
    assert np.abs(f.residual(v)).max() < 1e-6


def test_linear_prior_reproduces_its_quadratic():
    """A LinearPriorFactor is a quadratic about its linearisation point."""
    rng = np.random.default_rng(5)
    base = Values({
        "x0": PoseVariable(se3.exp(rng.normal(0, 0.3, 6))),
        "v0": VectorVariable(rng.normal(0, 0.4, 3)),
    })
    A = rng.normal(0, 1, (9, 9))
    b = rng.normal(0, 1, 9)
    f = F.LinearPriorFactor(
        ["x0", "v0"], {k: v.copy() for k, v in base.items()}, A, b, [6, 3]
    )
    assert np.allclose(f.residual(base), b)
    delta = np.concatenate([rng.normal(0, 1e-3, 6), rng.normal(0, 1e-3, 3)])
    moved = base.retract({"x0": delta[:6], "v0": delta[6:]})
    assert np.allclose(f.residual(moved), A @ delta + b, atol=1e-6)
    assert_jacobians_match(f, moved, tolerance=1e-4)
