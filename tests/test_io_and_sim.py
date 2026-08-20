"""The ASL loader, the simulator and the renderer.

The loader is tested by writing a *simulated* sequence out in ASL format and
reading it back: every field then has a value the test already knows, which a
real dataset cannot give you.  A column read out of order shows up as a
mismatch rather than as a mysteriously worse trajectory three modules later.
"""

from __future__ import annotations

import numpy as np
import pytest

from vio.geometry import so3
from vio.io import euroc
from vio.sim.render import SceneRenderer, write_asl
from vio.sim.simulator import SimConfig, simulate
from vio.sim.trajectory import Trajectory
from vio.types import GRAVITY, ImuBias, ImuNoise


# ---------------------------------------------------------------- trajectory


def test_trajectory_derivatives_are_exact():
    """Velocity, acceleration and body angular rate against finite differences.

    The angular-rate identity ``omega = Jr(r) r_dot`` is the only step in the
    trajectory that is not obvious by inspection, so it is checked directly
    against a difference of the rotation matrices it claims to describe.
    """
    tr = Trajectory(yaw_rate=0.35)
    h = 1e-6
    for t in np.linspace(0.5, 40.0, 200):
        t = float(t)
        w_fd = so3.log(tr.R(t).T @ tr.R(t + h)) / h
        assert np.abs(w_fd - tr.omega_body(t)).max() < 1e-6
        assert np.abs((tr.p(t + h) - tr.p(t)) / h - tr.v(t)).max() < 1e-5
        assert np.abs((tr.v(t + h) - tr.v(t)) / h - tr.a(t)).max() < 1e-5


def test_specific_force_reads_g_upwards_at_rest():
    """An accelerometer at rest reads +9.81 up, not zero: it measures specific
    force, not acceleration."""
    from vio.sim.trajectory import Sinusoid

    still = Trajectory(
        position=Sinusoid([0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 1.0]),
        rotation=Sinusoid([0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]),
    )
    f = still.specific_force(0.0, GRAVITY)
    assert np.allclose(f, np.array([0.0, 0.0, 9.81]), atol=1e-9)


# ---------------------------------------------------------------- simulator


def test_simulation_is_deterministic():
    a = simulate(SimConfig(duration=3.0, seed=5))
    b = simulate(SimConfig(duration=3.0, seed=5))
    assert np.allclose(a.imu.acc, b.imu.acc)
    assert np.allclose(a.landmarks, b.landmarks)
    assert np.allclose(a.frames[10].uv, b.frames[10].uv)


def test_simulation_changes_with_the_seed():
    a = simulate(SimConfig(duration=3.0, seed=5))
    b = simulate(SimConfig(duration=3.0, seed=6))
    assert not np.allclose(a.imu.acc, b.imu.acc)


def test_simulated_observations_are_plentiful_and_in_frame():
    seq = simulate(SimConfig(duration=6.0, seed=7))
    cam = seq.config.camera
    counts = np.array([len(f.ids) for f in seq.frames])
    assert counts.min() > 50
    for f in seq.frames[::20]:
        assert np.all(f.uv_clean[:, 0] >= 0) and np.all(f.uv_clean[:, 0] < cam.width)
        assert np.all(f.uv_clean[:, 1] >= 0) and np.all(f.uv_clean[:, 1] < cam.height)


def test_noise_free_configuration_really_is_noise_free():
    cfg = SimConfig(duration=2.0, imu_noise=ImuNoise(0, 0, 0, 0),
                    initial_bias=ImuBias(), bias_walk=False, pixel_noise=0.0, seed=8)
    seq = simulate(cfg)
    tr = cfg.trajectory
    for i in range(0, len(seq.imu), 37):
        t = float(seq.imu.t[i])
        assert np.allclose(seq.imu.acc[i], tr.specific_force(t, cfg.gravity), atol=1e-12)
        assert np.allclose(seq.imu.gyro[i], tr.omega_body(t), atol=1e-12)
    assert np.allclose(seq.frames[5].uv, seq.frames[5].uv_clean)


def test_scene_is_not_degenerate_for_the_fundamental_matrix():
    """Six exactly flat walls make each view planar, and a plane admits a
    one-parameter family of fundamental matrices.

    RANSAC does not fail loudly on that, it returns a confident, arbitrary
    model and discards most of the correspondences.  With an earlier version of
    the scene the front end was throwing away 34 tracks per frame on synthetic
    images against 0.2 per frame on EuRoC.  So the property to assert is not
    "there are boxes", it is "the two-view model keeps its input".
    """
    import cv2

    seq = simulate(SimConfig(duration=3.0, n_landmarks=2500, pixel_noise=0.3, seed=9))
    ratios = []
    for k in range(0, 40, 4):
        a, b = seq.frames[k], seq.frames[k + 2]
        common = np.intersect1d(a.ids, b.ids)
        if len(common) < 60:
            continue
        ia = {int(i): u for i, u in zip(a.ids, a.uv)}
        ib = {int(i): u for i, u in zip(b.ids, b.uv)}
        pa = np.stack([ia[int(c)] for c in common])
        pb = np.stack([ib[int(c)] for c in common])
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.999)
        assert mask is not None
        ratios.append(float(mask.reshape(-1).mean()))
    assert len(ratios) >= 5
    assert float(np.median(ratios)) > 0.85, ratios


def test_inter_frame_pixel_motion_is_trackable():
    """The default motion must sit in the range KLT can resolve.

    Seventeen pixels per frame against a 25-pixel feature spacing is where an
    earlier version of these defaults sat, and the tracker mismatched
    neighbouring features on nearly every track while reporting success.
    """
    seq = simulate(SimConfig(duration=8.0, seed=10))
    motions = []
    for a, b in zip(seq.frames[:-1], seq.frames[1:]):
        common = np.intersect1d(a.ids, b.ids)
        if len(common) < 20:
            continue
        ia = {int(i): u for i, u in zip(a.ids, a.uv_clean)}
        ib = {int(i): u for i, u in zip(b.ids, b.uv_clean)}
        motions.append(np.median([
            np.linalg.norm(ib[int(c)] - ia[int(c)]) for c in common
        ]))
    median = float(np.median(motions))
    assert 2.0 < median < 12.0, median


def test_imu_data_between_includes_the_bracketing_samples():
    seq = simulate(SimConfig(duration=2.0, seed=11))
    block = seq.imu.between(0.5013, 0.7017)
    assert block.t[0] <= 0.5013
    assert block.t[-1] >= 0.7017


# ---------------------------------------------------------------- renderer


def test_rendered_images_have_realistic_contrast():
    """Flat images with isolated dots are what the first renderer produced, and
    KLT cannot track them."""
    seq = simulate(SimConfig(duration=1.0, seed=12))
    image = SceneRenderer(seq, noise_sigma=2.0).render(0)
    assert image.dtype == np.uint8
    assert image.shape == (seq.config.camera.height, seq.config.camera.width)
    assert image.std() > 25.0


def test_rendering_is_deterministic():
    seq = simulate(SimConfig(duration=1.0, seed=13))
    a = SceneRenderer(seq, noise_sigma=2.0).render(3)
    b = SceneRenderer(seq, noise_sigma=2.0).render(3)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- ASL loader


@pytest.fixture(scope="module")
def asl_sequence(tmp_path_factory):
    root = tmp_path_factory.mktemp("asl")
    seq = simulate(SimConfig(duration=2.0, n_landmarks=400, seed=14))
    write_asl(seq, root, write_images=True, texture_points=1500)
    return seq, root


def test_loader_recovers_what_the_writer_wrote(asl_sequence):
    seq, root = asl_sequence
    loaded = euroc.load(root, stereo=False)

    assert len(loaded.cam0) == len(seq.frames)
    assert loaded.cam0.camera.width == seq.config.camera.width
    assert loaded.cam0.camera.fx == pytest.approx(seq.config.camera.fx)
    assert loaded.cam0.camera.model == seq.config.camera.model
    assert np.allclose(
        loaded.cam0.camera.distortion, seq.config.camera.distortion, atol=1e-12
    )


def test_loader_reads_the_extrinsic_the_right_way_round(asl_sequence):
    """``T_BS`` maps sensor coordinates into the body frame, not the reverse."""
    seq, root = asl_sequence
    loaded = euroc.load(root, stereo=False)
    assert np.allclose(
        loaded.cam0.T_body_cam.matrix(), seq.config.T_body_cam.matrix(), atol=1e-9
    )


def test_loader_reads_imu_columns_in_the_right_order(asl_sequence):
    """The ASL layout is gyro first, then accelerometer.  Swapping them gives
    an estimator that fails in a way that looks like a tuning problem."""
    seq, root = asl_sequence
    loaded = euroc.load(root, stereo=False)
    n = min(len(loaded.imu), len(seq.imu))
    positive = seq.imu.t >= 0
    assert np.allclose(
        loaded.imu.acc[:50], seq.imu.acc[positive][:50], atol=1e-6
    )
    assert np.allclose(
        loaded.imu.gyro[:50], seq.imu.gyro[positive][:50], atol=1e-6
    )
    assert n > 100


def test_loader_reads_the_noise_densities(asl_sequence):
    seq, root = asl_sequence
    loaded = euroc.load(root, stereo=False)
    assert loaded.imu_noise.acc_noise == pytest.approx(seq.config.imu_noise.acc_noise)
    assert loaded.imu_noise.gyro_walk == pytest.approx(seq.config.imu_noise.gyro_walk)


def test_loader_recovers_ground_truth_poses(asl_sequence):
    seq, root = asl_sequence
    loaded = euroc.load(root, stereo=False)
    gt = loaded.groundtruth
    assert gt is not None and len(gt) > 100
    for t in (0.4, 1.0, 1.6):
        pose = gt.pose_at(t)
        truth = seq.state_at(t)
        assert np.linalg.norm(pose.t - truth.p) < 1e-3
        assert so3.angle_between(pose.R, truth.R) < 1e-4


def test_groundtruth_outside_its_range_returns_none(asl_sequence):
    _, root = asl_sequence
    gt = euroc.load(root, stereo=False).groundtruth
    assert gt.pose_at(-5.0) is None
    assert gt.pose_at(1e6) is None


def test_loader_rebases_timestamps_to_the_first_frame(asl_sequence):
    _, root = asl_sequence
    loaded = euroc.load(root, stereo=False)
    assert abs(float(loaded.cam0.times[0])) < 1e-9
    assert loaded.origin_ns == int(loaded.cam0.stamps_ns[0])


def test_loader_reads_images(asl_sequence):
    _, root = asl_sequence
    loaded = euroc.load(root, stereo=False)
    img = loaded.cam0.image(0)
    assert img.dtype == np.uint8
    assert img.shape == (loaded.cam0.camera.height, loaded.cam0.camera.width)


def test_available_lists_sequences(asl_sequence, tmp_path):
    _, root = asl_sequence
    parent = root.parent
    assert root.name in euroc.available(parent)
    assert euroc.available(tmp_path / "nowhere") == []


def test_loading_a_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        euroc.load(tmp_path / "not-a-sequence")
