"""IMU preintegration against an analytic trajectory.

The reference here is not another integrator, it is the closed-form
trajectory in :mod:`vio.sim.trajectory`, whose derivatives are exact.  So an
integration error of 4e-8 metres is genuinely the estimator's error and not
the difference between two approximations.
"""

from __future__ import annotations

import numpy as np
import pytest

from vio.geometry import so3
from vio.imu.preintegration import Preintegrated, preintegrate
from vio.types import GRAVITY, ImuBias, ImuData, ImuNoise


def test_midpoint_beats_euler_by_orders_of_magnitude(clean_sequence):
    """The whole reason the default is not the scheme in the paper.

    Rotating both endpoint accelerations by the attitude at their own end of
    the interval, rather than both by the start-of-interval attitude, is what
    makes this true.  The naive midpoint, average the raw measurements, keep
    the old rotation, is *worse* than Euler on velocity, which is how the
    first version of this was caught.
    """
    seq = clean_sequence
    results = {}
    for scheme in ("euler", "midpoint"):
        errs = []
        for t0 in np.arange(0.5, 4.5, 0.25):
            pre = preintegrate(seq.imu, t0, t0 + 0.2, scheme=scheme)
            pred = pre.predict(seq.state_at(t0))
            truth = seq.state_at(t0 + 0.2)
            errs.append((
                np.linalg.norm(pred.p - truth.p),
                np.linalg.norm(pred.v - truth.v),
                so3.angle_between(pred.R, truth.R),
            ))
        results[scheme] = np.max(errs, axis=0)

    assert results["midpoint"][0] < results["euler"][0] / 100.0
    assert results["midpoint"][1] < results["euler"][1] / 100.0
    assert results["midpoint"][2] < results["euler"][2] / 100.0


@pytest.mark.parametrize("span", [0.05, 0.2, 1.0])
def test_prediction_matches_the_analytic_trajectory(clean_sequence, span):
    seq = clean_sequence
    for t0 in np.arange(0.5, 4.0, 0.5):
        pre = preintegrate(seq.imu, t0, t0 + span)
        pred = pre.predict(seq.state_at(t0))
        truth = seq.state_at(t0 + span)
        assert np.linalg.norm(pred.p - truth.p) < 1e-5 * max(span, 0.1) / 0.1
        assert so3.angle_between(pred.R, truth.R) < 1e-6


def test_partial_intervals_are_interpolated(clean_sequence):
    """A block over an interval that misses the sample grid still integrates exactly.

    Image timestamps do not land on IMU samples.  Rounding to the nearest
    sample loses up to 5 ms of motion per keyframe at 200 Hz and it loses it
    in the same direction every time.
    """
    seq = clean_sequence
    t0, t1 = 1.0023, 1.2017
    pre = preintegrate(seq.imu, t0, t1)
    assert abs(pre.dt - (t1 - t0)) < 1e-12
    pred = pre.predict(seq.state_at(t0))
    truth = seq.state_at(t1)
    assert np.linalg.norm(pred.p - truth.p) < 1e-4


def test_bias_jacobians_match_a_full_reintegration(biased_sequence):
    """The first-order correction against the real thing, by finite difference."""
    seq = biased_sequence
    b0 = ImuBias([0.02, -0.01, 0.03], [0.001, 0.002, -0.0015])
    base = preintegrate(seq.imu, 1.0, 1.25, bias=b0)
    h = 1e-6
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        perturbed = ImuBias.from_vector(b0.vector() + d)
        exact = preintegrate(seq.imu, 1.0, 1.25, bias=perturbed)
        lin_R, lin_v, lin_p = base.corrected(perturbed)
        assert so3.angle_between(exact.dR, lin_R) / h < 1e-6
        assert np.abs(exact.dv - lin_v).max() / h < 1e-6
        assert np.abs(exact.dp - lin_p).max() / h < 1e-6


def test_repropagation_equals_a_fresh_integration(biased_sequence):
    seq = biased_sequence
    base = preintegrate(seq.imu, 1.0, 1.4, bias=ImuBias())
    new_bias = ImuBias([0.2, -0.1, 0.15], [0.02, -0.01, 0.03])
    assert base.needs_repropagation(new_bias)
    repropagated = base.repropagate(new_bias)
    fresh = preintegrate(seq.imu, 1.0, 1.4, bias=new_bias)
    assert so3.angle_between(repropagated.dR, fresh.dR) < 1e-14
    assert np.abs(repropagated.dv - fresh.dv).max() < 1e-14
    assert np.abs(repropagated.dp - fresh.dp).max() < 1e-14
    assert abs(repropagated.dt - fresh.dt) < 1e-15


def test_small_bias_change_does_not_trigger_repropagation():
    pre = Preintegrated(bias_ref=ImuBias([0.01] * 3, [0.001] * 3))
    assert not pre.needs_repropagation(ImuBias([0.02] * 3, [0.002] * 3))
    assert pre.needs_repropagation(ImuBias([0.5] * 3, [0.001] * 3))


@pytest.mark.slow
def test_covariance_matches_a_monte_carlo(clean_sequence):
    """The 9x9 covariance is checked against 1500 noisy realisations.

    Within about ten per cent is the bar: the propagation ignores both the
    variance reduction from the midpoint average and the correlation between
    adjacent intervals that share an endpoint sample, which together make it
    mildly conservative by design.
    """
    seq = clean_sequence
    noise = ImuNoise()
    t0, t1 = 1.0, 1.5
    truth = preintegrate(seq.imu, t0, t1, noise=noise)
    dt = 1.0 / seq.config.imu_rate

    rng = np.random.default_rng(0)
    sub = seq.imu.between(t0 - 0.02, t1 + 0.02)
    samples = np.empty((1500, 9))
    for i in range(len(samples)):
        acc = sub.acc + rng.normal(0, noise.acc_noise / np.sqrt(dt), sub.acc.shape)
        gyro = sub.gyro + rng.normal(0, noise.gyro_noise / np.sqrt(dt), sub.gyro.shape)
        p = preintegrate(ImuData(sub.t, acc, gyro), t0, t1, noise=noise)
        samples[i, 0:3] = so3.log(truth.dR.T @ p.dR)
        samples[i, 3:6] = p.dv - truth.dv
        samples[i, 6:9] = p.dp - truth.dp

    predicted = np.sqrt(np.diag(truth.cov))
    empirical = np.sqrt(np.diag(np.cov(samples.T)))
    ratio = predicted / empirical
    assert np.all(ratio > 0.85) and np.all(ratio < 1.15), ratio

    chi2 = float(np.mean(np.einsum(
        "ni,ij,nj->n", samples, np.linalg.inv(truth.cov), samples
    )))
    assert 7.5 < chi2 < 10.5, chi2


def test_covariance_grows_with_the_interval(clean_sequence):
    seq = clean_sequence
    sizes = [
        np.trace(preintegrate(seq.imu, 1.0, 1.0 + span, noise=ImuNoise()).cov)
        for span in (0.1, 0.3, 0.9)
    ]
    assert sizes[0] < sizes[1] < sizes[2]


def test_discrete_noise_scales_the_right_way():
    """White noise as ``sigma^2 / dt``, random walk as ``sigma^2 * dt``.

    Swapping these produces an estimator that behaves on short sequences and
    diverges on long ones, which is an expensive way to find out.
    """
    noise = ImuNoise(acc_noise=0.01, gyro_noise=0.02, acc_walk=0.03, gyro_walk=0.04)
    meas_a, walk_a = noise.discrete(0.01)
    meas_b, walk_b = noise.discrete(0.02)
    assert np.allclose(np.diag(meas_a)[:3], 0.01 ** 2 / 0.01)
    assert np.allclose(np.diag(walk_a)[:3], 0.03 ** 2 * 0.01)
    assert np.diag(meas_b)[0] < np.diag(meas_a)[0]
    assert np.diag(walk_b)[0] > np.diag(walk_a)[0]


def test_residual_is_zero_on_ground_truth(clean_sequence):
    """The residual the back end minimises, evaluated at the true states."""
    seq = clean_sequence
    for t0 in (1.0, 2.0, 3.0):
        pre = preintegrate(seq.imu, t0, t0 + 0.15)
        r = pre.residual(
            seq.state_at(t0), ImuBias(), seq.state_at(t0 + 0.15), GRAVITY
        )
        assert np.abs(r).max() < 1e-6


def test_empty_and_degenerate_blocks_do_not_raise(clean_sequence):
    seq = clean_sequence
    assert preintegrate(seq.imu, 2.0, 2.0).dt == 0.0
    assert preintegrate(seq.imu, 2.0, 1.0).dt == 0.0
    empty = ImuData(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)))
    assert preintegrate(empty, 0.0, 1.0).dt == 0.0


def test_information_is_symmetric_positive_definite(clean_sequence):
    pre = preintegrate(clean_sequence.imu, 1.0, 1.2, noise=ImuNoise())
    info = pre.information()
    assert np.allclose(info, info.T, rtol=1e-6)
    assert np.all(np.linalg.eigvalsh(info) > 0)
    L = pre.sqrt_information()
    assert np.allclose(L.T @ L, info, rtol=1e-6)
