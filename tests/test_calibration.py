"""Camera-IMU extrinsic and temporal calibration.

Every test here injects a *known* error and checks that it comes back.  A
calibration routine that returns its own initial value is the easiest thing in
the world to write and the hardest to notice, so nothing here starts from the
right answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from vio.calibration import extrinsic, temporal
from vio.geometry import Pose, so3
from vio.sim.simulator import SimConfig, simulate
from vio.types import ImuBias, ImuNoise


@pytest.fixture(scope="module")
def calib_sequence():
    return simulate(SimConfig(
        duration=20.0,
        imu_noise=ImuNoise(0.0, 0.0, 0.0, 0.0),
        initial_bias=ImuBias(),
        bias_walk=False,
        pixel_noise=0.3,
        n_landmarks=900,
        seed=41,
    ))


@pytest.fixture(scope="module")
def offset_sequence():
    """The camera clock runs 14.3 ms behind the IMU clock."""
    return simulate(SimConfig(
        duration=20.0,
        imu_noise=ImuNoise(0.0, 0.0, 0.0, 0.0),
        initial_bias=ImuBias(),
        bias_walk=False,
        time_offset=0.0143,
        seed=42,
    ))


# ---------------------------------------------------------------- rotation


def test_extrinsic_rotation_is_recovered_from_rotation_alone(calib_sequence):
    seq = calib_sequence
    T_bc = seq.config.T_body_cam
    cam_rots = [(s.pose * T_bc).R for s in seq.states]
    body_rots = [s.R for s in seq.states]
    R, residual, n = extrinsic.rotation_from_streams(cam_rots, body_rots, stride=4)
    assert R is not None and n > 50
    assert np.degrees(so3.angle_between(R, T_bc.R)) < 0.05
    assert residual < 1e-6


def test_extrinsic_rotation_survives_gyro_noise():
    seq = simulate(SimConfig(duration=20.0, seed=43))
    T_bc = seq.config.T_body_cam
    cam_rots = [(s.pose * T_bc).R for s in seq.states]
    body_rots = [s.R for s in seq.states]
    rng = np.random.default_rng(0)
    body_rots = [R @ so3.exp(rng.normal(0, 2e-3, 3)) for R in body_rots]
    R, _, _ = extrinsic.rotation_from_streams(cam_rots, body_rots, stride=6)
    assert np.degrees(so3.angle_between(R, T_bc.R)) < 1.5


def test_tiny_rotations_are_excluded_rather_than_averaged_in():
    """Below the threshold the logarithm is noise, and Procrustes weights every
    pair by its magnitude squared, so a noise-only pair contributes pure noise
    at full weight."""
    rng = np.random.default_rng(1)
    R_true = so3.exp(np.array([0.3, -1.1, 0.7]))
    good_cam = [so3.exp(rng.normal(0, 0.3, 3)) for _ in range(40)]
    good_body = [R_true @ R @ R_true.T for R in good_cam]
    junk_cam = [so3.exp(rng.normal(0, 1e-4, 3)) for _ in range(200)]
    junk_body = [so3.exp(rng.normal(0, 1e-4, 3)) for _ in range(200)]

    R, _, n = extrinsic.rotation_from_increments(
        good_cam + junk_cam, good_body + junk_body, min_angle=0.02
    )
    assert n == 40
    assert np.degrees(so3.angle_between(R, R_true)) < 0.01


def test_rotation_estimation_needs_enough_pairs():
    R, residual, n = extrinsic.rotation_from_increments([], [])
    assert R is None and n == 0


def test_excitation_reports_per_axis_rotation(calib_sequence):
    seq = calib_sequence
    total = extrinsic.excitation([s.R for s in seq.states])
    assert total.shape == (3,)
    assert np.all(total > 0.2)


# ---------------------------------------------------------------- full extrinsic


def test_joint_refinement_recovers_a_disturbed_extrinsic(calib_sequence):
    """Start 3 degrees and 4 cm away from the truth and converge back."""
    seq = calib_sequence
    T_bc = seq.config.T_body_cam
    disturbed = Pose(
        so3.project(T_bc.R @ so3.exp(np.array([0.03, -0.02, 0.025]))),
        T_bc.t + np.array([0.03, -0.02, 0.015]),
    )
    assert np.degrees(so3.angle_between(disturbed.R, T_bc.R)) > 2.0

    step = 4
    indices = list(range(0, 120, step))
    poses = [seq.states[i].pose for i in indices]
    observations = []
    seen = {}
    for k, i in enumerate(indices):
        frame = seq.frames[i]
        xn = seq.config.camera.undistort(frame.uv)
        for lid, x in zip(frame.ids, xn):
            seen.setdefault(int(lid), []).append((k, x))
    landmarks = {}
    rng = np.random.default_rng(2)
    for lid, obs in seen.items():
        if len(obs) < 4:
            continue
        landmarks[lid] = seq.landmarks[lid] + rng.normal(0, 0.02, 3)
        observations.extend((k, lid, x) for k, x in obs)

    result = extrinsic.refine(
        poses, landmarks, observations, seq.config.camera, disturbed,
        pixel_sigma=0.3, fix_poses=True, iterations=40,
    )
    assert result.ok, result.reason
    assert np.degrees(so3.angle_between(result.T_body_cam.R, T_bc.R)) < 0.3
    assert np.linalg.norm(result.T_body_cam.t - T_bc.t) < 0.01
    assert result.rotation_change_deg > 1.0
    assert result.covariance is not None


def test_refinement_reports_failure_with_too_few_observations(calib_sequence):
    seq = calib_sequence
    result = extrinsic.refine(
        [seq.states[0].pose], {}, [], seq.config.camera, seq.config.T_body_cam
    )
    assert not result.ok


# ---------------------------------------------------------------- temporal


def test_time_offset_is_recovered_by_correlation(offset_sequence):
    seq = offset_sequence
    T_bc = seq.config.T_body_cam
    # The rotations a visual estimator would produce, on the camera's clock.
    cam_times = seq.times
    cam_rots = [(s.pose * T_bc).R for s in seq.states]
    result = temporal.estimate_by_correlation(
        cam_times, cam_rots, seq.imu.t, seq.imu.gyro
    )
    assert result.ok, result.reason
    assert result.correlation > 0.9
    assert abs(result.offset - seq.config.time_offset) < 0.004


def test_zero_offset_is_recovered_as_zero(calib_sequence):
    seq = calib_sequence
    T_bc = seq.config.T_body_cam
    result = temporal.estimate_by_correlation(
        seq.times, [(s.pose * T_bc).R for s in seq.states], seq.imu.t, seq.imu.gyro
    )
    assert result.ok
    assert abs(result.offset) < 0.004


@pytest.mark.parametrize("offset", [-0.030, -0.008, 0.008, 0.030])
def test_offsets_of_both_signs_are_recovered(offset):
    seq = simulate(SimConfig(
        duration=20.0, imu_noise=ImuNoise(0, 0, 0, 0), initial_bias=ImuBias(),
        bias_walk=False, time_offset=offset, seed=44,
    ))
    T_bc = seq.config.T_body_cam
    result = temporal.estimate_by_correlation(
        seq.times, [(s.pose * T_bc).R for s in seq.states], seq.imu.t, seq.imu.gyro
    )
    assert result.ok
    assert abs(result.offset - offset) < 0.005, result.offset


def test_correlation_beats_the_grid_resolution(offset_sequence):
    """The parabolic fit through the peak has to be doing something."""
    seq = offset_sequence
    T_bc = seq.config.T_body_cam
    result = temporal.estimate_by_correlation(
        seq.times, [(s.pose * T_bc).R for s in seq.states],
        seq.imu.t, seq.imu.gyro, steps=61,
    )
    assert result.ok
    assert result.resolution > 0.004
    assert abs(result.offset - seq.config.time_offset) < result.resolution


def test_correlation_refuses_a_sequence_with_no_rotation():
    from vio.sim.trajectory import Sinusoid, Trajectory

    straight = Trajectory(
        position=Sinusoid([3, 2, 0.4], [0.3, 0.4, 0.9], [0, 1, 2], [0, 0, 1.4]),
        rotation=Sinusoid([0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]),
    )
    seq = simulate(SimConfig(
        duration=10.0, trajectory=straight, imu_noise=ImuNoise(0, 0, 0, 0),
        initial_bias=ImuBias(), bias_walk=False, seed=45,
    ))
    T_bc = seq.config.T_body_cam
    result = temporal.estimate_by_correlation(
        seq.times, [(s.pose * T_bc).R for s in seq.states], seq.imu.t, seq.imu.gyro
    )
    assert not result.ok


def test_feature_velocity_refinement_recovers_a_known_offset():
    """A synthetic problem with an exact answer: features move at a known rate
    and the frames are labelled 9 ms early."""
    rng = np.random.default_rng(3)
    true_td = 0.009
    n_tracks = 60
    frames, predictions = [], {}
    positions = rng.uniform(-0.4, 0.4, (n_tracks, 2))
    velocities = rng.normal(0, 0.5, (n_tracks, 2))
    dt = 0.05
    for i in range(30):
        t = i * dt
        observed = {j: positions[j] + velocities[j] * t for j in range(n_tracks)}
        frames.append((t, observed))
        if i > 0:
            predictions[i] = {
                j: observed[j] + velocities[j] * true_td for j in range(n_tracks)
            }
    result = temporal.refine_by_feature_velocity(frames, predictions, initial=0.0)
    assert result.ok and result.refined
    assert abs(result.offset - true_td) < 1e-6


def test_feature_velocity_refinement_reports_when_it_cannot_run():
    result = temporal.refine_by_feature_velocity([(0.0, {})], {})
    assert not result.ok


def test_angular_rate_from_rotations_matches_the_gyroscope(calib_sequence):
    """The two signals the correlation aligns must be the same signal.

    ``calib_sequence`` has no time offset and no gyro noise, so the visual
    angular rate and the measured one should agree to the discretisation of the
    camera's own frame interval.
    """
    seq = calib_sequence
    mid, visual_rate = temporal.angular_rate_from_rotations(
        seq.times, [s.R for s in seq.states]
    )
    measured = np.stack([
        np.interp(mid, seq.imu.t, seq.imu.gyro[:, k]) for k in range(3)
    ], axis=1)
    measured_rate = np.linalg.norm(measured, axis=1)
    assert np.abs(visual_rate - measured_rate).max() < 0.02
    assert np.corrcoef(visual_rate, measured_rate)[0, 1] > 0.999
