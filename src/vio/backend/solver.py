"""Levenberg-Marquardt over a factor graph, with a Schur complement for landmarks.

The structure that makes bundle adjustment tractable is that landmarks do not
see each other.  A landmark appears in a handful of reprojection factors and in
nothing else, so the landmark-landmark block of the Hessian is block diagonal
with 3x3 blocks.  Eliminating it costs one 3x3 inverse per landmark and leaves
a much smaller system over the poses::

    [ A   B ] [ dx_p ]     [ -g_a ]              (A - B C^-1 B^T) dx_p = -(g_a - B C^-1 g_c)
    [ B^T C ] [ dx_l ]  =  [ -g_c ]      ==>     dx_l = -C^-1 (g_c + B^T dx_p)

For a window of ten keyframes and two thousand landmarks that turns a 6060-
dimensional solve into a 150-dimensional one.  ``scipy`` does the factorisation;
building the normal equations and arranging the complement is what is written
out here, because that is where the structure lives.

The damping is Marquardt's, proportional to the diagonal of the Hessian
rather than to the identity, so the step is invariant to how the variables
happen to be scaled.  Position in metres and rotation in radians differ by
roughly an order of magnitude in a typical Hessian here and identity damping
punishes whichever happens to be smaller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .graph import FactorGraph, linearise
from .variables import Values


@dataclass
class SolverOptions:
    max_iterations: int = 30
    #: Stop when the cost stops falling by this fraction.
    function_tolerance: float = 1e-8
    #: Stop when the step is this small in the tangent space.
    step_tolerance: float = 1e-10
    initial_lambda: float = 1e-4
    min_lambda: float = 1e-12
    max_lambda: float = 1e12
    #: Sparse factorisation above this many reduced columns; dense below.  A
    #: dense LAPACK solve beats scipy's sparse LU up to a few hundred columns.
    dense_threshold: int = 400
    verbose: bool = False
    #: Ridge added to the landmark blocks before inversion.  A landmark seen
    #: from two nearly identical viewpoints has a near-singular 3x3 block, and
    #: without this the complement produces a step of order 1e9.
    point_ridge: float = 1e-9


@dataclass
class SolverResult:
    values: Values
    initial_cost: float
    final_cost: float
    iterations: int
    converged: bool
    reason: str = ""
    seconds: float = 0.0
    history: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.iterations} it, cost {self.initial_cost:.4g} -> "
            f"{self.final_cost:.4g}, {self.seconds * 1e3:.0f} ms, {self.reason}"
        )


def optimise(
    graph: FactorGraph,
    values: Values,
    fixed=frozenset(),
    options: SolverOptions | None = None,
) -> SolverResult:
    """Minimise the graph's robustified cost, starting from ``values``."""
    opt = options or SolverOptions()
    t_start = time.perf_counter()
    current = values.copy()

    system = linearise(graph, current, fixed)
    cost = system.cost
    initial_cost = cost
    lam = opt.initial_lambda
    history = [cost]
    reason = "max iterations"
    converged = False
    it = 0

    for it in range(1, opt.max_iterations + 1):
        if system.J.shape[1] == 0:
            reason, converged = "nothing to optimise", True
            break

        step, ok = _solve_step(system, lam, opt)
        if not ok:
            lam = min(lam * 10.0, opt.max_lambda)
            if lam >= opt.max_lambda:
                reason = "linear solve failed"
                break
            continue

        step_norm = float(np.linalg.norm(step))
        deltas = {
            key: step[off: off + dim] for key, (off, dim) in system.layout.items()
        }
        candidate = current.retract(deltas)
        cand_system = linearise(graph, candidate, fixed)

        if cand_system.cost < cost:
            rel = (cost - cand_system.cost) / max(cost, 1e-300)
            current, system, cost = candidate, cand_system, cand_system.cost
            history.append(cost)
            lam = max(lam * 0.3, opt.min_lambda)
            if opt.verbose:
                print(f"  it {it:2d}  cost {cost:.6g}  |dx| {step_norm:.3g}  lam {lam:.1e}")
            if rel < opt.function_tolerance:
                reason, converged = "function tolerance", True
                break
            if step_norm < opt.step_tolerance:
                reason, converged = "step tolerance", True
                break
        else:
            lam = min(lam * 10.0, opt.max_lambda)
            if opt.verbose:
                print(f"  it {it:2d}  rejected, lam -> {lam:.1e}")
            if lam >= opt.max_lambda:
                reason, converged = "lambda saturated", True
                break

    return SolverResult(
        values=current,
        initial_cost=initial_cost,
        final_cost=cost,
        iterations=it,
        converged=converged,
        reason=reason,
        seconds=time.perf_counter() - t_start,
        history=history,
    )


def _solve_step(system, lam: float, opt: SolverOptions):
    """One damped Gauss-Newton step, eliminating landmarks if there are any."""
    J, r = system.J, system.r
    H = (J.T @ J).tocsr()
    g = J.T @ r

    diag = H.diagonal().copy()
    # A variable with no observations this iteration gets a floor, so the
    # damped block stays invertible instead of producing a NaN that silently
    # spreads through the whole step.
    diag = np.maximum(diag, 1e-12)
    damping = lam * diag

    n_kept = system.n_kept
    n_total = H.shape[0]
    n_points = n_total - n_kept

    if n_points == 0:
        return _dense_or_sparse_solve(H, g, damping, n_total, opt)

    A = H[:n_kept, :n_kept]
    B = H[:n_kept, n_kept:]
    C = H[n_kept:, n_kept:]
    g_a, g_c = g[:n_kept], g[n_kept:]

    # C is block diagonal with 3x3 blocks, one per landmark.  Pull the blocks
    # out and invert them individually rather than asking a general sparse
    # routine to rediscover the structure.
    n_blocks = n_points // 3
    C_dense = np.asarray(C.todense()).reshape(n_points, n_points)
    blocks = np.empty((n_blocks, 3, 3))
    d_c = damping[n_kept:]
    for i in range(n_blocks):
        s = 3 * i
        blocks[i] = C_dense[s: s + 3, s: s + 3] + np.diag(
            d_c[s: s + 3] + opt.point_ridge
        )
    try:
        inv_blocks = np.linalg.inv(blocks)
    except np.linalg.LinAlgError:
        return None, False

    C_inv = sp.block_diag([sp.coo_array(b) for b in inv_blocks], format="csr")
    BCinv = (B @ C_inv).tocsr()
    S = A - BCinv @ B.T
    rhs = g_a - BCinv @ g_c

    step_a, ok = _dense_or_sparse_solve(S, rhs, damping[:n_kept], n_kept, opt)
    if not ok:
        return None, False

    step_c = -(C_inv @ (g_c + B.T @ step_a))
    return np.concatenate([step_a, step_c]), True


def _dense_or_sparse_solve(H, g, damping, n, opt: SolverOptions):
    """Solve ``(H + diag(damping)) x = -g`` and report whether it worked."""
    if n == 0:
        return np.zeros(0), True
    if n <= opt.dense_threshold:
        M = np.asarray(H.todense()) if sp.issparse(H) else np.asarray(H)
        M = M + np.diag(damping)
        try:
            return np.linalg.solve(M, -g), True
        except np.linalg.LinAlgError:
            return None, False
    M = (H + sp.diags(damping)).tocsc()
    try:
        return spla.spsolve(M, -g), True
    except (RuntimeError, ValueError):
        return None, False


def covariance(graph: FactorGraph, values: Values, keys, fixed=frozenset()):
    """Marginal covariance of ``keys``, by inverting the reduced Hessian.

    Dense, so this is for reporting and for the calibration confidence
    intervals, not for anything in the estimator's inner loop.  Returns
    ``None`` if the system is rank deficient, which is the honest answer for an
    unobservable direction rather than a large number that looks like a result.
    """
    system = linearise(graph, values, fixed)
    H = np.asarray((system.J.T @ system.J).todense())
    try:
        cov = np.linalg.inv(H + np.eye(H.shape[0]) * 1e-12)
    except np.linalg.LinAlgError:
        return None
    out = {}
    for key in keys:
        if key not in system.layout:
            continue
        off, dim = system.layout[key]
        out[key] = cov[off: off + dim, off: off + dim]
    return out
