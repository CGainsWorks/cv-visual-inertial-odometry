"""Visual-inertial initialisation: scale, gravity, gyro bias, velocity."""

from __future__ import annotations

import numpy as np
import pytest

from vio.geometry import Pose
from vio.imu.initialisation import (
    GRAVITY_MAGNITUDE,
    align_to_gravity,
    estimate_gyro_bias,
    initialise,
    translational_excitation,
)
from vio.imu.preintegration import preintegrate
from vio.sim.simulator import SimConfig, simulate
from vio.types import GRAVITY, ImuBias, ImuNoise


def _window(seq, n_keyframes: int, stride: int = 3, scale: float = 1.0):
    """Up-to-scale camera poses and the IMU blocks between them.

    Rotations are exact, a monocular system does observe those, and
    translations are divided by ``scale``, which is what an arbitrary
    monocular reconstruction looks like.
    """
    T_bc = seq.config.T_body_cam
    idx = list(range(0, n_keyframes * stride, stride))
    origin = seq.states[idx[0]].pose * T_bc
    poses = []
    for i in idx:
        T = seq.states[i].pose * T_bc
        poses.append(Pose(T.R, origin.t + (T.t - origin.t) / scale))
    blocks = [
        preintegrate(seq.imu, seq.frames[a].true_stamp, seq.frames[b].true_stamp,
                     noise=seq.config.imu_noise)
        for a, b in zip(idx[:-1], idx[1:])
    ]
    return idx, poses, blocks


@pytest.fixture(scope="module")
def init_sequence():
    return simulate(SimConfig(
        duration=8.0,
        imu_noise=ImuNoise(0.0, 0.0, 0.0, 0.0),
        initial_bias=ImuBias([0.05, -0.03, 0.02], [0.004, -0.006, 0.003]),
        bias_walk=False,
        seed=13,
    ))


def test_gyro_bias_is_recovered_from_rotation_alone(init_sequence):
    seq = init_sequence
    _, poses, blocks = _window(seq, 20)
    R_wb = [T.R @ seq.config.T_body_cam.R.T for T in poses]
    bg = estimate_gyro_bias(R_wb, blocks)
    truth = seq.config.initial_bias.gyro
    assert np.linalg.norm(bg - truth) < 1e-5


@pytest.mark.parametrize("scale", [0.05, 1.0, 25.0])
def test_recovery_is_scale_equivariant(init_sequence, scale):
    """A monocular reconstruction has an arbitrary scale, so the *error* must
    not depend on which one it happened to be built at."""
    seq = init_sequence
    _, poses, blocks = _window(seq, 20, scale=scale)
    result = initialise(poses, blocks, seq.config.T_body_cam)
    assert result.ok, result.reason
    assert abs(result.scale - scale) / scale < 0.06

    angle = np.degrees(np.arccos(np.clip(
        result.gravity @ GRAVITY
        / np.linalg.norm(result.gravity) / np.linalg.norm(GRAVITY), -1, 1
    )))
    assert angle < 1.5
    assert abs(np.linalg.norm(result.gravity) - GRAVITY_MAGNITUDE) < 1e-6


def test_scale_error_is_the_same_at_every_scale(init_sequence):
    seq = init_sequence
    relative = []
    for scale in (0.05, 1.0, 25.0):
        _, poses, blocks = _window(seq, 20, scale=scale)
        r = initialise(poses, blocks, seq.config.T_body_cam)
        relative.append(abs(r.scale - scale) / scale)
    assert max(relative) - min(relative) < 1e-6


def test_velocities_are_recovered(init_sequence):
    seq = init_sequence
    idx, poses, blocks = _window(seq, 20)
    result = initialise(poses, blocks, seq.config.T_body_cam)
    assert result.ok
    truth = np.stack([seq.states[i].v for i in idx])
    assert np.mean(np.linalg.norm(result.velocities - truth, axis=1)) < 0.06


def test_too_few_keyframes_is_refused(init_sequence):
    seq = init_sequence
    _, poses, blocks = _window(seq, 3)
    assert not initialise(poses, blocks, seq.config.T_body_cam).ok


def test_a_stationary_window_is_refused():
    """No translational excitation means no observable scale, and saying so is
    the whole point of the excitation gate."""
    from vio.sim.trajectory import Sinusoid, Trajectory

    still = Trajectory(
        position=Sinusoid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0, 0, 0], [0, 0, 1.4]),
        rotation=Sinusoid([0.1, 0.08, 0.1], [1.2, 0.9, 0.7], [0, 1, 2], [0, 0, 0]),
    )
    seq = simulate(SimConfig(
        duration=6.0, trajectory=still, imu_noise=ImuNoise(0, 0, 0, 0),
        initial_bias=ImuBias(), bias_walk=False, seed=4,
    ))
    _, poses, blocks = _window(seq, 20)
    result = initialise(poses, blocks, seq.config.T_body_cam)
    assert not result.ok
    assert "excitation" in result.reason


def test_excitation_ignores_rotation():
    """Rotating on the spot must not read as translational excitation.

    Measuring the spread of the body-frame specific force, the obvious first
    attempt, mostly measures how much the platform turned, because gravity
    swings around the body frame as it does.
    """
    from vio.sim.trajectory import Sinusoid, Trajectory

    spinning = Trajectory(
        position=Sinusoid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0, 0, 0], [0, 0, 1.4]),
        rotation=Sinusoid([0.5, 0.4, 0.6], [1.5, 1.1, 0.9], [0, 1, 2], [0, 0, 0]),
    )
    seq = simulate(SimConfig(
        duration=6.0, trajectory=spinning, imu_noise=ImuNoise(0, 0, 0, 0),
        initial_bias=ImuBias(), bias_walk=False, seed=5,
    ))
    _, poses, blocks = _window(seq, 20)
    R_wb = [T.R @ seq.config.T_body_cam.R.T for T in poses]
    assert translational_excitation(R_wb, blocks) < 0.02


def test_align_to_gravity_maps_gravity_down():
    rng = np.random.default_rng(6)
    for _ in range(200):
        v = rng.normal(0, 1, 3)
        v = v / np.linalg.norm(v) * GRAVITY_MAGNITUDE
        R = align_to_gravity(v)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) > 0
        assert np.allclose(R @ v, GRAVITY, atol=1e-9)


def test_align_to_gravity_handles_the_aligned_and_inverted_cases():
    assert np.allclose(align_to_gravity(GRAVITY) @ GRAVITY, GRAVITY, atol=1e-9)
    R = align_to_gravity(-GRAVITY)
    assert np.allclose(R @ (-GRAVITY), GRAVITY, atol=1e-9)


def test_noise_degrades_but_does_not_break_initialisation():
    seq = simulate(SimConfig(
        duration=8.0,
        initial_bias=ImuBias([0.05, -0.03, 0.02], [0.004, -0.006, 0.003]),
        seed=17,
    ))
    _, poses, blocks = _window(seq, 24)
    result = initialise(poses, blocks, seq.config.T_body_cam)
    assert result.ok, result.reason
    assert 0.85 < result.scale < 1.15
    assert np.linalg.norm(result.gyro_bias - seq.config.initial_bias.gyro) < 2e-3
