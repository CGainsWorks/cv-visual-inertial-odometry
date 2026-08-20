"""The solver, the vectorised groups and marginalisation."""

from __future__ import annotations

import numpy as np
import pytest

from vio.backend import factors as F
from vio.backend.graph import FactorGraph, linearise
from vio.backend.groups import ReprojectionGroup
from vio.backend.marginalisation import marginalise, square_root
from vio.backend.solver import SolverOptions, covariance, optimise
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
from vio.types import ImuBias

CAMERA = PinholeCamera(
    752, 480, 458.654, 457.296, 367.215, 248.375,
    [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], "radtan",
)
T_BODY_CAM = Pose(
    se3.exp(np.array([0.1, -0.2, 1.5, 0.0, 0.0, 0.0])).R,
    np.array([0.02, -0.06, 0.01]),
)


def _bundle(rng, n_poses=8, n_points=90, loss=None):
    values = Values()
    for i in range(n_poses):
        values[f"x{i}"] = PoseVariable(se3.exp(rng.normal(0, 0.15, 6)))
    group = ReprojectionGroup(CAMERA, T_BODY_CAM, 1.0, loss)
    for j in range(n_points):
        values[f"l{j}"] = PointVariable(
            rng.normal(0, 1.5, 3) + np.array([0.0, 0.0, 6.0])
        )
        for i in range(n_poses):
            T = values[f"x{i}"].value * T_BODY_CAM
            pc = T.R.T @ (values[f"l{j}"].value - T.t)
            if pc[2] < 0.5:
                continue
            group.add(f"x{i}", f"l{j}", pc[:2] / pc[2] + rng.normal(0, 0.002, 2))
    return values, group


# ---------------------------------------------------------------- groups


def test_group_matches_scalar_factors_exactly():
    """The fast path and the readable path must not diverge."""
    rng = np.random.default_rng(4)
    values, group = _bundle(rng, loss=F.HuberLoss(2.0))
    assert len(group) > 500

    a = linearise(FactorGraph([group]), values)
    b = linearise(FactorGraph(group.to_factors()), values)
    assert a.layout == b.layout
    assert abs(a.cost - b.cost) < 1e-9
    assert np.abs(a.r - b.r).max() < 1e-10
    assert np.abs(a.J.toarray() - b.J.toarray()).max() < 1e-9


def test_group_and_scalar_agree_without_a_robust_kernel():
    rng = np.random.default_rng(5)
    values, group = _bundle(rng, loss=None)
    a = linearise(FactorGraph([group]), values)
    b = linearise(FactorGraph(group.to_factors()), values)
    assert abs(a.cost - b.cost) < 1e-9
    assert np.abs(a.J.toarray() - b.J.toarray()).max() < 1e-9


def test_group_error_matches_graph_error():
    rng = np.random.default_rng(6)
    values, group = _bundle(rng, loss=F.HuberLoss(2.0))
    assert group.error(values) == pytest.approx(
        sum(f.error(values) for f in group.to_factors()), rel=1e-9
    )


def test_empty_group_is_harmless():
    group = ReprojectionGroup(CAMERA, T_BODY_CAM)
    assert group.rows() == 0
    assert group.error(Values()) == 0.0
    rows, cols, data, r, cost = group.linearise_into(Values(), {}, 0)
    assert not rows and r.size == 0 and cost == 0.0


# ---------------------------------------------------------------- solver


def test_bundle_adjustment_recovers_a_perturbed_structure():
    """A bundle adjustment from a disturbed start returns to the truth."""
    rng = np.random.default_rng(7)
    truth, group = _bundle(rng, n_poses=8, n_points=90)
    start = truth.copy()
    for k in start:
        if k.startswith("x") and k not in ("x0", "x1"):
            start[k] = start[k].retract(rng.normal(0, 0.02, 6))
        elif k.startswith("l"):
            start[k] = start[k].retract(rng.normal(0, 0.08, 3))

    g = FactorGraph([group])
    g.add(F.PosePriorFactor("x0", truth.pose("x0"), np.eye(6) * 1e4))
    g.add(F.PosePriorFactor("x1", truth.pose("x1"), np.eye(6) * 1e4))

    res = optimise(g, start, options=SolverOptions(max_iterations=30))
    assert res.final_cost < res.initial_cost / 20.0

    errs = [
        np.linalg.norm(res.values.pose(f"x{i}").t - truth.pose(f"x{i}").t)
        for i in range(8)
    ]
    assert max(errs) < 0.01
    rots = [
        so3.angle_between(res.values.pose(f"x{i}").R, truth.pose(f"x{i}").R)
        for i in range(8)
    ]
    assert max(rots) < np.radians(0.2)


def test_schur_complement_agrees_with_the_direct_solve():
    """Eliminating the landmarks must not change the answer, only the cost of it."""
    rng = np.random.default_rng(8)
    truth, group = _bundle(rng, n_poses=5, n_points=40)
    start = truth.copy()
    for k in start:
        if k.startswith("l"):
            start[k] = start[k].retract(rng.normal(0, 0.05, 3))
        elif k != "x0":
            start[k] = start[k].retract(rng.normal(0, 0.01, 6))
    g = FactorGraph([group])
    g.add(F.PosePriorFactor("x0", truth.pose("x0"), np.eye(6) * 1e4))
    g.add(F.PosePriorFactor("x1", truth.pose("x1"), np.eye(6) * 1e4))

    with_schur = optimise(g, start, options=SolverOptions(max_iterations=20))

    # Force the direct path by hiding the landmark type from the solver.
    plain = Values({
        k: (VectorVariable(v.value) if k.startswith("l") else v.copy())
        for k, v in start.items()
    })
    direct = optimise(g, plain, options=SolverOptions(max_iterations=20))
    assert with_schur.final_cost == pytest.approx(direct.final_cost, rel=2e-3)
    for i in range(5):
        assert np.linalg.norm(
            with_schur.values.pose(f"x{i}").t - direct.values.pose(f"x{i}").t
        ) < 2e-3


def test_solver_reports_a_reason_and_leaves_values_valid():
    rng = np.random.default_rng(9)
    values, group = _bundle(rng, n_poses=3, n_points=20)
    g = FactorGraph([group])
    g.add(F.PosePriorFactor("x0", values.pose("x0"), np.eye(6) * 1e4))
    res = optimise(g, values, options=SolverOptions(max_iterations=5))
    assert res.reason
    assert res.summary()
    for k, v in res.values.items():
        assert np.all(np.isfinite(v.value if not isinstance(v, PoseVariable)
                                  else v.value.matrix()))


def test_fixed_variables_do_not_move():
    rng = np.random.default_rng(10)
    values, group = _bundle(rng, n_poses=4, n_points=30)
    before = values.pose("x2").copy()
    g = FactorGraph([group])
    g.add(F.PosePriorFactor("x0", values.pose("x0"), np.eye(6) * 1e4))
    res = optimise(g, values, fixed={"x2"},
                   options=SolverOptions(max_iterations=10))
    assert np.allclose(res.values.pose("x2").matrix(), before.matrix())


def test_covariance_is_positive_definite_where_observable():
    rng = np.random.default_rng(11)
    values, group = _bundle(rng, n_poses=4, n_points=40)
    g = FactorGraph([group])
    g.add(F.PosePriorFactor("x0", values.pose("x0"), np.eye(6) * 1e3))
    g.add(F.PosePriorFactor("x1", values.pose("x1"), np.eye(6) * 1e3))
    cov = covariance(g, values, ["x2"])
    assert cov is not None
    assert np.all(np.linalg.eigvalsh(cov["x2"]) > 0)


# -------------------------------------------------------- marginalisation


def test_square_root_reproduces_the_quadratic():
    rng = np.random.default_rng(12)
    M = rng.normal(0, 1, (7, 7))
    H = M @ M.T
    b = rng.normal(0, 1, 7)
    A, b_res = square_root(H, b)
    assert np.allclose(A.T @ A, H, atol=1e-8)
    assert np.allclose(A.T @ b_res, b, atol=1e-8)


def test_square_root_discards_the_null_space():
    """A rank-deficient Hessian must not produce enormous numbers."""
    rng = np.random.default_rng(13)
    M = rng.normal(0, 1, (6, 4))
    H = M @ M.T           # rank 4 in a 6-dimensional space
    b = H @ rng.normal(0, 1, 6)
    A, b_res = square_root(H, b)
    assert A.shape[0] == 4
    assert np.all(np.isfinite(b_res))
    assert np.allclose(A.T @ A, H, atol=1e-8)


def test_marginalising_a_state_preserves_the_solution(clean_sequence):
    """Solving with a marginal prior must match solving the full problem.

    Three inertial states in a chain: solve all three, then eliminate the first
    and solve the remaining two against the prior it left behind.  The
    surviving states must land in the same place.
    """
    seq = clean_sequence
    stamps = [1.0, 1.2, 1.4]
    values = Values()
    for i, t in enumerate(stamps):
        st = seq.state_at(t)
        values[f"x{i}"] = PoseVariable(st.pose)
        values[f"v{i}"] = VectorVariable(st.v)
        values[f"b{i}"] = BiasVariable(np.zeros(6))

    full = FactorGraph()
    blocks = []
    for i in range(2):
        pre = preintegrate(seq.imu, stamps[i], stamps[i + 1], bias=ImuBias())
        blocks.append(pre)
        full.add(F.ImuFactor(f"x{i}", f"v{i}", f"b{i}",
                             f"x{i+1}", f"v{i+1}", pre))
        full.add(F.BiasRandomWalkFactor(f"b{i}", f"b{i+1}", pre.dt, seq.config.imu_noise))
    full.add(F.PosePriorFactor("x0", values.pose("x0"), np.eye(6) * 1e5))
    full.add(F.VectorPriorFactor("v0", values.vector("v0"), np.eye(3) * 1e4))
    full.add(F.VectorPriorFactor("b0", np.zeros(6), np.eye(6) * 1e3))

    rng = np.random.default_rng(14)
    start = values.copy()
    for i in (1, 2):
        start[f"x{i}"] = start[f"x{i}"].retract(rng.normal(0, 0.01, 6))
        start[f"v{i}"] = start[f"v{i}"].retract(rng.normal(0, 0.02, 3))

    reference = optimise(full, start, options=SolverOptions(max_iterations=25))

    first = FactorGraph()
    first.add(F.ImuFactor("x0", "v0", "b0", "x1", "v1", blocks[0]))
    first.add(F.BiasRandomWalkFactor("b0", "b1", blocks[0].dt, seq.config.imu_noise))
    first.add(F.PosePriorFactor("x0", values.pose("x0"), np.eye(6) * 1e5))
    first.add(F.VectorPriorFactor("v0", values.vector("v0"), np.eye(3) * 1e4))
    first.add(F.VectorPriorFactor("b0", np.zeros(6), np.eye(6) * 1e3))
    prior, absorbed, keep = marginalise(first, reference.values, ["x0", "v0", "b0"])
    assert prior is not None
    assert set(keep) == {"x1", "v1", "b1"}
    assert len(absorbed) == len(first.factors)

    reduced = FactorGraph([prior])
    reduced.add(F.ImuFactor("x1", "v1", "b1", "x2", "v2", blocks[1]))
    reduced.add(F.BiasRandomWalkFactor("b1", "b2", blocks[1].dt, seq.config.imu_noise))
    trimmed = Values({k: v.copy() for k, v in start.items()
                      if not k.endswith("0")})
    after = optimise(reduced, trimmed, options=SolverOptions(max_iterations=25))

    for key in ("x1", "x2"):
        assert np.linalg.norm(
            after.values.pose(key).t - reference.values.pose(key).t
        ) < 5e-3


def test_marginalising_nothing_returns_nothing():
    g = FactorGraph()
    values = Values({"x0": PoseVariable(Pose.identity())})
    prior, absorbed, keep = marginalise(g, values, ["x0"])
    assert prior is None and absorbed == [] and keep == []
