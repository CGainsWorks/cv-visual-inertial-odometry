"""SO(3), SE(3) and the camera model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vio.geometry import se3, so3
from vio.geometry.camera import PinholeCamera, projection_jacobian

EUROC_CAM0 = PinholeCamera(
    752, 480, 458.654, 457.296, 367.215, 248.375,
    [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], "radtan",
)
TUMVI_CAM0 = PinholeCamera(
    512, 512, 190.978, 190.973, 254.932, 256.897,
    [0.00348, 0.000715, -0.00205, 0.000949], "equidistant",
)


# ---------------------------------------------------------------- SO(3)


def test_exp_agrees_with_scipy():
    rng = np.random.default_rng(0)
    for _ in range(500):
        v = rng.normal(0, 1, 3) * rng.choice([1e-9, 1e-4, 1.0, 3.0])
        assert np.allclose(so3.exp(v), Rotation.from_rotvec(v).as_matrix(), atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 1e-12, 1e-6, 1e-3, 1.0, 3.0,
                                   np.pi - 1e-9, np.pi])
def test_log_inverts_exp_across_the_whole_range(theta):
    """Includes the half turn, where the trace-based formula loses precision."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        R = so3.exp(axis * theta)
        back = so3.log(R)
        assert np.linalg.norm(so3.exp(back) - R) < 1e-13
        if theta < np.pi - 1e-6:
            assert np.allclose(back, axis * theta, atol=1e-12)


def test_hat_is_the_cross_product():
    rng = np.random.default_rng(1)
    for _ in range(100):
        a, b = rng.normal(0, 1, 3), rng.normal(0, 1, 3)
        assert np.allclose(so3.hat(a) @ b, np.cross(a, b))
        assert np.allclose(so3.vee(so3.hat(a)), a)


def test_right_jacobian_matches_its_definition():
    """``Exp(p + dp) == Exp(p) Exp(Jr(p) dp)`` to first order."""
    rng = np.random.default_rng(2)
    h = 1e-7
    for _ in range(200):
        phi = rng.normal(0, 1, 3) * rng.choice([1e-6, 0.1, 1.0, 2.5])
        Jr = so3.right_jacobian(phi)
        for i in range(3):
            d = np.zeros(3)
            d[i] = h
            lhs = so3.exp(phi + d)
            rhs = so3.exp(phi) @ so3.exp(Jr @ d)
            assert np.linalg.norm(lhs - rhs) < 1e-11


def test_right_jacobian_inverse():
    rng = np.random.default_rng(3)
    for _ in range(200):
        phi = rng.normal(0, 1, 3) * rng.choice([1e-8, 0.3, 2.0])
        assert np.allclose(
            so3.right_jacobian(phi) @ so3.right_jacobian_inv(phi), np.eye(3), atol=1e-9
        )


def test_left_jacobian_identity_used_by_the_factors():
    """``Jl^-1(x) == Jr^-1(-x)``, relied on by ImuFactor and RelativePoseFactor."""
    rng = np.random.default_rng(4)
    for _ in range(100):
        x = rng.normal(0, 1, 3)
        Jl = so3.right_jacobian(-x)
        assert np.allclose(np.linalg.inv(Jl), so3.right_jacobian_inv(-x), atol=1e-9)


def test_project_returns_the_nearest_rotation():
    rng = np.random.default_rng(5)
    for _ in range(100):
        R = so3.exp(rng.normal(0, 1, 3))
        drifted = R + rng.normal(0, 1e-6, (3, 3))
        fixed = so3.project(drifted)
        assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(fixed) > 0
        assert so3.angle_between(R, fixed) < 1e-5


# ---------------------------------------------------------------- SE(3)


def test_se3_exp_log_roundtrip():
    """``log(exp(xi)) == xi`` for every rotation magnitude short of a half turn."""
    rng = np.random.default_rng(6)
    worst = 0.0
    for _ in range(2000):
        scale = rng.choice([0.0, 1e-9, 1e-4, 1.0, 3.0, np.pi - 1e-7])
        axis = rng.normal(0, 1, 3)
        axis = axis / max(np.linalg.norm(axis), 1e-12) * scale
        xi = np.concatenate([axis, rng.normal(0, 2, 3)])
        worst = max(worst, float(np.abs(se3.log(se3.exp(xi)) - xi).max()))
    assert worst < 1e-12


def test_se3_log_is_a_group_inverse_at_a_half_turn():
    """At exactly pi the logarithm is two-valued, so only the group law holds.

    ``+pi a`` and ``-pi a`` name the same rotation, and they give *different*
    translation components through ``V(phi)``.  So ``log(exp(xi)) == xi`` is
    not a theorem there and asserting it would be asserting a convention, not
    a correctness property.  What must hold is that the pose is recovered.
    """
    rng = np.random.default_rng(16)
    for _ in range(200):
        axis = rng.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis) * np.pi
        xi = np.concatenate([axis, rng.normal(0, 2, 3)])
        T = se3.exp(xi)
        assert np.linalg.norm(se3.exp(se3.log(T)).matrix() - T.matrix()) < 1e-11


def test_pose_composition_and_inverse():
    rng = np.random.default_rng(8)
    for _ in range(200):
        a = se3.exp(rng.normal(0, 1, 6))
        b = se3.exp(rng.normal(0, 1, 6))
        p = rng.normal(0, 3, 3)
        assert np.allclose((a * b).act(p), a.act(b.act(p)))
        assert np.allclose((a * a.inverse()).matrix(), np.eye(4), atol=1e-12)
        assert np.allclose(a.matrix() @ b.matrix(), (a * b).matrix())


def test_pose_act_handles_arrays():
    rng = np.random.default_rng(9)
    T = se3.exp(rng.normal(0, 1, 6))
    pts = rng.normal(0, 2, (50, 3))
    single = np.stack([T.act(p) for p in pts])
    assert np.allclose(T.act(pts), single)


def test_adjoint_definition():
    """``T Exp(xi) T^-1 == Exp(Adj(T) xi)``."""
    rng = np.random.default_rng(10)
    for _ in range(100):
        T = se3.exp(rng.normal(0, 0.8, 6))
        xi = rng.normal(0, 1e-4, 6)
        lhs = T * se3.exp(xi) * T.inverse()
        rhs = se3.exp(T.adjoint() @ xi)
        assert np.linalg.norm(lhs.matrix() - rhs.matrix()) < 1e-9


def test_retract_and_local_are_inverses():
    rng = np.random.default_rng(11)
    for _ in range(200):
        T = se3.exp(rng.normal(0, 1, 6))
        xi = rng.normal(0, 0.3, 6)
        assert np.allclose(T.local(T.retract(xi)), xi, atol=1e-10)


# ---------------------------------------------------------------- camera


@pytest.mark.parametrize("cam,extent", [(EUROC_CAM0, 0.65), (TUMVI_CAM0, 1.2)])
def test_undistort_inverts_project_to_machine_precision(cam, extent):
    """The Newton inverse must be exact, not merely close.

    OpenCV's default five fixed-point iterations leave about a tenth of a pixel
    at the corners, which is larger than the tracking noise the estimator
    models, hence the hand-written inverse this checks.
    """
    rng = np.random.default_rng(12)
    xn = rng.uniform(-extent, extent, (2000, 2))
    uv = cam.project(xn)
    assert np.abs(cam.undistort(uv) - xn).max() < 1e-12


def test_undistort_handles_empty_input():
    assert EUROC_CAM0.undistort(np.zeros((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("cam", [EUROC_CAM0, TUMVI_CAM0])
def test_distortion_jacobian_matches_finite_differences(cam):
    rng = np.random.default_rng(13)
    xn = rng.uniform(-0.5, 0.5, (200, 2))
    J = cam.distort_jacobian(xn)
    h = 1e-7
    for i in range(2):
        d = np.zeros(2)
        d[i] = h
        num = (cam.distort(xn + d) - cam.distort(xn - d)) / (2 * h)
        assert np.abs(J[:, :, i] - num).max() < 1e-6


def test_projection_jacobian_matches_finite_differences():
    rng = np.random.default_rng(14)
    h = 1e-7
    for _ in range(200):
        p = np.array([rng.normal(0, 1), rng.normal(0, 1), rng.uniform(0.5, 20.0)])
        J = projection_jacobian(p)
        for i in range(3):
            d = np.zeros(3)
            d[i] = h
            f = lambda q: q[:2] / q[2]  # noqa: E731
            num = (f(p + d) - f(p - d)) / (2 * h)
            assert np.abs(J[:, i] - num).max() < 1e-5


def test_project_points_rejects_behind_and_out_of_frame():
    cam = EUROC_CAM0
    pts = np.array([
        [0.0, 0.0, 5.0],      # centre, valid
        [0.0, 0.0, -5.0],     # behind
        [0.0, 0.0, 0.0],      # on the plane
        [50.0, 0.0, 1.0],     # far out of frame
    ])
    uv, ok = cam.project_points(pts)
    assert list(ok) == [True, False, False, False]
    assert np.allclose(uv[0], [cam.cx, cam.cy], atol=1e-9)


def test_bearings_are_unit_vectors():
    rng = np.random.default_rng(15)
    uv = np.stack([rng.uniform(0, 752, 200), rng.uniform(0, 480, 200)], axis=1)
    b = EUROC_CAM0.bearings(uv)
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0)
    assert np.all(b[:, 2] > 0)


def test_camera_serialises_round_trip():
    d = EUROC_CAM0.to_dict()
    back = PinholeCamera.from_dict(d)
    assert back.to_dict() == d


def test_unknown_distortion_model_is_rejected():
    with pytest.raises(ValueError):
        PinholeCamera(10, 10, 1, 1, 5, 5, np.zeros(4), "brown-conrady")
