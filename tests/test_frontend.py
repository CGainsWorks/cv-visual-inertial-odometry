"""Feature tracking, triangulation, two-view geometry and PnP."""

from __future__ import annotations

import numpy as np
import pytest

from vio.frontend import geometry as G
from vio.frontend.tracker import Tracker, TrackerConfig
from vio.geometry import Pose, so3


# ---------------------------------------------------------------- tracker


@pytest.mark.slow
def test_tracker_produces_long_consistent_tracks(rendered_sequence):
    seq, images = rendered_sequence
    tracker = Tracker(seq.config.camera, TrackerConfig())
    frames = [
        tracker.process(img, f.stamp, f.index)
        for img, f in zip(images, seq.frames)
    ]
    tracked = np.array([f.n_tracked for f in frames[1:]])
    assert tracked.mean() > 100
    assert all(f.ok for f in frames)

    lengths = np.array([t.length for t in tracker.tracks.values()])
    assert np.median(lengths) >= 2
    assert lengths.max() > 20


def test_tracked_features_stay_inside_the_image(rendered_sequence):
    seq, images = rendered_sequence
    cam = seq.config.camera
    tracker = Tracker(cam, TrackerConfig())
    for img, f in zip(images[:40], seq.frames[:40]):
        out = tracker.process(img, f.stamp, f.index)
        assert np.all(out.uv[:, 0] >= 0) and np.all(out.uv[:, 0] < cam.width)
        assert np.all(out.uv[:, 1] >= 0) and np.all(out.uv[:, 1] < cam.height)
        assert len(out.ids) == len(set(out.ids.tolist()))


def test_detection_respects_the_spacing_mask(rendered_sequence):
    seq, images = rendered_sequence
    cfg = TrackerConfig(min_distance=30.0, max_features=120)
    tracker = Tracker(seq.config.camera, cfg)
    out = tracker.process(images[0], seq.frames[0].stamp, 0)
    d = np.linalg.norm(out.uv[:, None, :] - out.uv[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    # goodFeaturesToTrack enforces this exactly; the mask enforces it against
    # existing tracks.  A few pixels of slack for rounding in the mask circles.
    assert d.min() > cfg.min_distance - 3.0


def test_tracker_prunes_dead_tracks(rendered_sequence):
    seq, images = rendered_sequence
    tracker = Tracker(seq.config.camera, TrackerConfig())
    for img, f in zip(images[:60], seq.frames[:60]):
        tracker.process(img, f.stamp, f.index)
    before = len(tracker.tracks)
    removed = tracker.prune(before_frame=50)
    assert removed > 0
    assert len(tracker.tracks) == before - removed
    assert all(t.last_frame >= 50 for t in tracker.tracks.values())


@pytest.mark.slow
def test_gyro_prediction_does_not_break_tracking(rendered_sequence):
    """It is off by default because it buys nothing at 20 Hz.  It must still work."""
    from vio.imu.preintegration import preintegrate
    from vio.types import ImuBias

    seq, images = rendered_sequence
    R_bc = seq.config.T_body_cam.R
    tracker = Tracker(seq.config.camera, TrackerConfig(use_gyro_prediction=True))
    prev_t = None
    counts = []
    for img, f in zip(images[:40], seq.frames[:40]):
        R = None
        if prev_t is not None:
            pre = preintegrate(seq.imu, prev_t, f.stamp, bias=ImuBias(),
                               noise=seq.config.imu_noise)
            R = R_bc.T @ pre.dR @ R_bc
        out = tracker.process(img, f.stamp, f.index, gyro_rotation=R)
        counts.append(out.n_tracked)
        prev_t = f.stamp
    assert np.mean(counts[1:]) > 90


def test_track_parallax_grows(rendered_sequence):
    seq, images = rendered_sequence
    tracker = Tracker(seq.config.camera, TrackerConfig())
    for img, f in zip(images[:30], seq.frames[:30]):
        tracker.process(img, f.stamp, f.index)
    long_tracks = [t for t in tracker.tracks.values() if t.length > 10]
    assert long_tracks
    assert max(t.parallax() for t in long_tracks) > 0.02


# ---------------------------------------------------------------- geometry


@pytest.fixture(scope="module")
def two_views():
    from vio.sim.simulator import SimConfig, simulate

    cfg = SimConfig(duration=4.0, n_landmarks=900, pixel_noise=0.4, seed=31)
    seq = simulate(cfg)
    poses = [seq.states[k].pose * cfg.T_body_cam for k in range(len(seq.frames))]
    return seq, poses


def test_triangulation_recovers_landmarks(two_views):
    seq, poses = two_views
    cam = seq.config.camera
    a, b = seq.frames[0], seq.frames[24]
    common = np.intersect1d(a.ids, b.ids)
    ia = {int(i): u for i, u in zip(a.ids, a.uv)}
    ib = {int(i): u for i, u in zip(b.ids, b.uv)}
    errors = []
    for lid in common:
        x0 = cam.undistort(ia[int(lid)][None])[0]
        x1 = cam.undistort(ib[int(lid)][None])[0]
        p, ok = G.triangulate_two_view(poses[0], poses[24], x0, x1)
        if ok:
            errors.append(np.linalg.norm(p - seq.landmarks[lid]))
    assert len(errors) > 30
    assert np.median(errors) < 0.4


def test_triangulation_rejects_insufficient_parallax(two_views):
    seq, poses = two_views
    cam = seq.config.camera
    a, b = seq.frames[0], seq.frames[1]
    common = np.intersect1d(a.ids, b.ids)
    ia = {int(i): u for i, u in zip(a.ids, a.uv)}
    ib = {int(i): u for i, u in zip(b.ids, b.uv)}
    rejected = 0
    for lid in common[:80]:
        x0 = cam.undistort(ia[int(lid)][None])[0]
        x1 = cam.undistort(ib[int(lid)][None])[0]
        _, ok = G.triangulate_two_view(poses[0], poses[1], x0, x1, min_angle_deg=2.0)
        rejected += not ok
    assert rejected > 60


def test_triangulation_rejects_points_behind_the_camera(two_views):
    """Two identical bearings triangulate to a point at infinity or behind."""
    _, poses = two_views
    centre = np.array([0.0, 0.0])
    _, ok = G.triangulate_two_view(
        poses[0], poses[10], centre, centre, min_angle_deg=0.0
    )
    assert not ok


def test_triangulation_needs_two_views():
    _, ok = G.triangulate_dlt([Pose.identity()], np.zeros((1, 2)))
    assert not ok


def test_refinement_does_not_make_things_worse(two_views):
    seq, poses = two_views
    cam = seq.config.camera
    a, b, c = seq.frames[0], seq.frames[16], seq.frames[32]
    common = np.intersect1d(np.intersect1d(a.ids, b.ids), c.ids)
    maps = [{int(i): u for i, u in zip(f.ids, f.uv)} for f in (a, b, c)]
    view_poses = [poses[0], poses[16], poses[32]]
    before, after = [], []
    for lid in common[:60]:
        xs = np.stack([cam.undistort(m[int(lid)][None])[0] for m in maps])
        p, ok = G.triangulate_dlt(view_poses, xs)
        if not ok:
            continue
        before.append(np.linalg.norm(p - seq.landmarks[lid]))
        after.append(np.linalg.norm(
            G.refine_point(view_poses, xs, p) - seq.landmarks[lid]
        ))
    assert len(before) > 10
    assert np.median(after) <= np.median(before) * 1.05


def test_relative_pose_recovers_rotation_and_direction(two_views):
    seq, poses = two_views
    cam = seq.config.camera
    a, b = seq.frames[0], seq.frames[30]
    common = np.intersect1d(a.ids, b.ids)
    ia = {int(i): u for i, u in zip(a.ids, a.uv)}
    ib = {int(i): u for i, u in zip(b.ids, b.uv)}
    x0 = cam.undistort(np.stack([ia[int(c)] for c in common]))
    x1 = cam.undistort(np.stack([ib[int(c)] for c in common]))

    result = G.relative_pose(x0, x1)
    assert result.ok
    assert result.inliers.sum() > 0.7 * len(common)
    truth = poses[0].inverse() * poses[30]
    assert np.degrees(so3.angle_between(result.T_0_1.R, truth.R)) < 3.0

    unit_est = result.T_0_1.t / np.linalg.norm(result.T_0_1.t)
    unit_true = truth.t / np.linalg.norm(truth.t)
    assert np.degrees(np.arccos(np.clip(unit_est @ unit_true, -1, 1))) < 12.0


def test_relative_pose_reports_failure_rather_than_guessing():
    result = G.relative_pose(np.zeros((4, 2)), np.zeros((4, 2)))
    assert not result.ok
    assert result.reason


def test_pnp_recovers_the_camera_pose(two_views):
    seq, poses = two_views
    cam = seq.config.camera
    f = seq.frames[40]
    result = G.solve_pnp(seq.landmarks[f.ids], cam.undistort(f.uv))
    assert result.ok
    assert result.inliers.sum() > 0.8 * len(f.ids)
    assert np.linalg.norm(result.T_world_cam.t - poses[40].t) < 0.02
    assert np.degrees(so3.angle_between(result.T_world_cam.R, poses[40].R)) < 0.3


def test_pnp_rejects_outliers(two_views):
    seq, poses = two_views
    cam = seq.config.camera
    f = seq.frames[40]
    pts = seq.landmarks[f.ids].copy()
    rng = np.random.default_rng(0)
    corrupt = rng.choice(len(pts), size=len(pts) // 4, replace=False)
    pts[corrupt] += rng.normal(0, 2.0, (len(corrupt), 3))
    result = G.solve_pnp(pts, cam.undistort(f.uv))
    assert result.ok
    assert not result.inliers[corrupt].mean() > 0.5
    assert np.linalg.norm(result.T_world_cam.t - poses[40].t) < 0.05


def test_pnp_with_too_few_points_fails_cleanly():
    result = G.solve_pnp(np.zeros((2, 3)), np.zeros((2, 2)))
    assert not result.ok and result.reason


def test_reprojection_errors_flag_points_behind_the_camera():
    T = Pose.identity()
    pts = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
    errs = G.reprojection_errors(T, pts, np.zeros((2, 2)), focal=400.0)
    assert errs[0] == pytest.approx(0.0)
    assert not np.isfinite(errs[1])
