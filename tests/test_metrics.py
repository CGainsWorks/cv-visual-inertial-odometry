"""Trajectory metrics, checked against closed-form answers and against evo.

The point of writing ATE and RPE out rather than shelling out to ``evo`` was
to be able to defend the numbers.  The point of *also* running ``evo`` is that
a metric implementation which agrees only with itself is not evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from vio.evaluation.metrics import (
    Statistics,
    associate,
    ate,
    drift_rate,
    path_length,
    rpe,
    umeyama,
)
from vio.geometry import Pose, se3, so3
from vio.io.trajectory import read_tum, write_tum


def _circle(n=200, radius=2.0, turns=1.0):
    t = np.linspace(0.0, 10.0, n)
    angle = np.linspace(0.0, 2 * np.pi * turns, n)
    poses = [
        Pose(so3.exp(np.array([0.0, 0.0, a])),
             np.array([radius * np.cos(a), radius * np.sin(a), 0.3 * np.sin(3 * a)]))
        for a in angle
    ]
    return t, poses


# ---------------------------------------------------------------- Umeyama


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(0, 2, (60, 3))
    R_true = so3.exp(rng.normal(0, 1, 3))
    t_true = rng.normal(0, 3, 3)
    s_true = 2.37
    dst = s_true * (src @ R_true.T) + t_true

    R, t, s = umeyama(src, dst, with_scale=True)
    assert s == pytest.approx(s_true, rel=1e-9)
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-8)


def test_umeyama_without_scale_ignores_it():
    rng = np.random.default_rng(1)
    src = rng.normal(0, 2, (60, 3))
    R_true = so3.exp(rng.normal(0, 1, 3))
    dst = 3.0 * (src @ R_true.T)
    _, _, s = umeyama(src, dst, with_scale=False)
    assert s == 1.0


def test_umeyama_never_returns_a_reflection():
    """A nearly planar set can fit better with a reflection, which is not a motion."""
    rng = np.random.default_rng(2)
    src = rng.normal(0, 2, (80, 3))
    src[:, 2] *= 1e-4
    dst = src.copy()
    dst[:, 2] *= -1.0
    R, _, _ = umeyama(src, dst, with_scale=True)
    assert np.linalg.det(R) > 0


# ---------------------------------------------------------------- association


def test_association_is_one_to_one():
    """Many estimates against sparse reference must not reuse a reference sample."""
    est = np.linspace(0.0, 1.0, 101)
    ref = np.linspace(0.0, 1.0, 11)
    ei, ri = associate(est, ref, max_difference=1.0)
    assert len(set(ri.tolist())) == len(ri)
    assert len(ri) == 11


def test_association_drops_pairs_that_are_too_far_apart():
    est = np.array([0.0, 5.0, 10.0])
    ref = np.array([0.01, 9.99])
    ei, ri = associate(est, ref, max_difference=0.05)
    assert list(ei) == [0, 2]
    assert list(ri) == [0, 1]


def test_association_handles_empty_input():
    ei, ri = associate([], [1.0, 2.0])
    assert len(ei) == 0 and len(ri) == 0


# ---------------------------------------------------------------- ATE / RPE


def test_ate_is_zero_on_a_transformed_copy_of_the_truth():
    """A perfect estimate in a different frame must score zero after alignment."""
    t, poses = _circle()
    T = se3.exp(np.array([0.4, -0.2, 1.1, 3.0, -2.0, 0.5]))
    moved = [T * p for p in poses]
    result = ate(t, moved, t, poses, mode="se3")
    assert result.translation.rmse < 1e-9
    assert result.rotation_deg.rmse < 1e-7
    assert result.n_matched == len(t)


def test_sim3_alignment_absorbs_a_scale_error_and_se3_does_not():
    """The distinction the whole visual-versus-inertial comparison rests on."""
    t, poses = _circle()
    scaled = [Pose(p.R, p.t * 0.6) for p in poses]
    with_scale = ate(t, scaled, t, poses, mode="sim3")
    without = ate(t, scaled, t, poses, mode="se3")
    assert with_scale.translation.rmse < 1e-9
    assert with_scale.scale == pytest.approx(1.0 / 0.6, rel=1e-6)
    assert without.translation.rmse > 0.5


def test_ate_grows_with_a_known_offset():
    t, poses = _circle()
    offset = np.array([0.1, 0.0, 0.0])
    # A constant offset is absorbed by alignment, so make it grow along the path.
    drifted = [
        Pose(p.R, p.t + offset * i / len(poses)) for i, p in enumerate(poses)
    ]
    result = ate(t, drifted, t, poses, mode="se3")
    assert 0.0 < result.translation.rmse < 0.1


def test_rpe_is_zero_for_a_globally_transformed_trajectory():
    t, poses = _circle()
    T = se3.exp(np.array([0.4, -0.2, 1.1, 3.0, -2.0, 0.5]))
    moved = [T * p for p in poses]
    result = rpe(t, moved, t, poses, delta=1.0, unit="m")
    assert result.n_pairs > 5
    assert result.translation.rmse < 1e-9


def test_rpe_by_distance_and_by_time_differ_when_the_platform_stops():
    """A hovering platform accumulates time but not distance and a time-based
    RPE rewards it for that."""
    t = np.linspace(0, 20, 401)
    poses = []
    for tt in t:
        x = tt if tt < 10 else 10.0     # moves, then stops dead
        poses.append(Pose(np.eye(3), np.array([x, 0.0, 0.0])))
    est = [Pose(p.R, p.t * 1.02) for p in poses]
    by_distance = rpe(t, est, t, poses, delta=2.0, unit="m")
    by_time = rpe(t, est, t, poses, delta=2.0, unit="s")
    assert by_distance.n_pairs > 0 and by_time.n_pairs > 0
    assert by_time.translation.mean < by_distance.translation.mean


def test_path_length_is_the_sum_of_steps():
    """A flat circle of unit radius, whose length is known exactly."""
    angle = np.linspace(0.0, 2 * np.pi, 2000)
    poses = [
        Pose(np.eye(3), np.array([np.cos(a), np.sin(a), 0.0])) for a in angle
    ]
    lengths = path_length(poses)
    assert lengths[0] == 0.0
    assert np.all(np.diff(lengths) >= 0.0)
    assert lengths[-1] == pytest.approx(2 * np.pi, rel=1e-5)


def test_drift_rate_reports_both_final_and_mean():
    t, poses = _circle(n=300)
    drifted = [
        Pose(p.R, p.t + np.array([0.2, 0.0, 0.0]) * i / len(poses))
        for i, p in enumerate(poses)
    ]
    d = drift_rate(t, drifted, t, poses, mode="se3")
    assert d["distance_m"] > 1.0
    assert np.isfinite(d["final_per_metre"])
    assert np.isfinite(d["mean_per_metre"])
    assert d["mean_per_metre"] <= d["final_per_metre"] * 2.0


def test_statistics_of_an_empty_series_is_all_nan():
    s = Statistics.of([])
    assert s.n == 0 and np.isnan(s.rmse)


def test_ate_of_too_short_a_trajectory_does_not_raise():
    result = ate([0.0], [Pose.identity()], [0.0], [Pose.identity()])
    assert result.n_matched <= 1


# ---------------------------------------------------------------- TUM I/O


def test_tum_round_trip(tmp_path):
    t, poses = _circle(n=50)
    path = write_tum(tmp_path / "traj.tum", t, poses, comment="a test")
    back_t, back_p = read_tum(path)
    assert np.allclose(back_t, t, atol=1e-9)
    for a, b in zip(poses, back_p):
        assert np.linalg.norm(a.t - b.t) < 1e-9
        assert so3.angle_between(a.R, b.R) < 1e-8


def test_tum_file_has_a_header_and_skips_comments(tmp_path):
    t, poses = _circle(n=5)
    path = write_tum(tmp_path / "t.tum", t, poses, offset_ns=1403715273262142976)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert "timestamp_offset_ns" in text
    assert len(read_tum(path)[0]) == 5


# ---------------------------------------------------------------- evo


def test_agrees_with_evo():
    """An independent referee for the ATE implementation.

    ``evo`` is a widely used trajectory evaluation package with its own
    association, its own Umeyama and its own statistics.  Agreement to 1e-9 on
    the same inputs is the strongest available evidence that neither is wrong
    in a way the other would share.
    """
    evo_main = pytest.importorskip("evo.main_ape")
    from evo.core import metrics as evo_metrics
    from evo.core import sync
    from evo.core.trajectory import PoseTrajectory3D

    rng = np.random.default_rng(3)
    t, poses = _circle(n=300)
    noisy = [p.copy() for p in poses]
    for i, p in enumerate(noisy):
        p.t = p.t + rng.normal(0, 0.02, 3) + np.array([0.15, 0.0, 0.0]) * i / len(noisy)
    T = se3.exp(np.array([0.3, -0.5, 1.4, 2.0, -1.0, 0.7]))
    noisy = [T * p for p in noisy]

    def to_evo(times, ps):
        return PoseTrajectory3D(
            poses_se3=[p.matrix() for p in ps], timestamps=np.asarray(times)
        )

    ref = to_evo(t, poses)
    est = to_evo(t, noisy)
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=0.01)

    for mode, correct_scale in (("se3", False), ("sim3", True)):
        result = evo_main.ape(
            ref_s, est_s, pose_relation=evo_metrics.PoseRelation.translation_part,
            align=True, correct_scale=correct_scale,
        )
        mine = ate(t, noisy, t, poses, mode=mode)
        assert mine.translation.rmse == pytest.approx(
            result.stats["rmse"], rel=1e-7, abs=1e-9
        ), mode
        assert mine.translation.mean == pytest.approx(
            result.stats["mean"], rel=1e-7, abs=1e-9
        ), mode
