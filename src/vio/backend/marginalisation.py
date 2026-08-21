"""Marginalisation: turning states that leave the window into a linear prior.

A fixed-lag smoother that simply *deletes* the oldest keyframe throws away
every constraint that keyframe carried, and the window quietly becomes
under-determined: nothing anchors the remaining states, so the whole thing
drifts as a rigid body and the optimiser is happy about it.  Marginalisation is
the alternative.  Eliminating a state from the linearised system leaves behind
a Gaussian over everything that state touched, and that Gaussian becomes a
factor.

Concretely: order the variables as ``[keep, drop]``, form the Hessian, and take
the Schur complement::

    H' = H_kk - H_kd H_dd^-1 H_dk
    b' = b_k  - H_kd H_dd^-1 b_d

then express ``(H', b')`` in square-root form as a residual so it can go back
into the same solver as everything else.  Two details matter and both are dealt
with below.

**The prior is anchored to a linearisation point.**  It is a quadratic
approximation built at one set of values, so it has to be evaluated as a
function of the *displacement* from those values, not of the values themselves.
That is why :class:`vio.backend.factors.LinearPriorFactor` keeps a copy of the
states it was built at.  Letting it float is the classic bug that makes a
sliding-window estimator slowly rotate its own gravity direction.

**Unobservable directions must be dropped.**  A monocular visual-inertial
system has four gauge freedoms: three of position and yaw about gravity.  The
marginal Hessian is therefore rank deficient by four, and inverting it or
taking its square root naively produces enormous numbers along the null space.
The eigendecomposition below discards eigenvalues below a relative threshold,
which keeps the prior informative in the directions it actually knows about and
silent in the ones it does not.
"""

from __future__ import annotations

import numpy as np

from .factors import LinearPriorFactor
from .graph import _keys_of, linearise
from .variables import Values

#: Eigenvalues below this fraction of the largest are treated as null space.
RANK_TOLERANCE = 1e-8


def marginalise(graph, values: Values, drop_keys, tolerance: float = RANK_TOLERANCE):
    """Eliminate ``drop_keys`` and return the prior they leave behind.

    Returns ``(prior_factor, absorbed_factors, keep_keys)``.  ``absorbed``
    are the factors that touched a dropped key and must be removed from the
    graph, because their information now lives inside the prior and counting it
    twice makes the estimator overconfident by exactly the amount that later
    looks like divergence.

    ``prior_factor`` is ``None`` when the dropped states were connected to
    nothing that survives, which happens for a keyframe whose landmarks all
    died with it.
    """
    # Order matters and is preserved deliberately.  ``drop_keys`` fixes the
    # column ordering of the system below, that ordering fixes the
    # eigendecomposition, and the eigendecomposition fixes the prior down to
    # the last bit.  Iterating a *set* of these keys instead makes the ordering
    # depend on Python's per-process string hash randomisation, which is how
    # two runs of the same commit with the same seed came out 0.257 and 0.362
    # on EuRoC V1_01.  Reproducibility is not something to leave to a hash.
    ordered_drop = list(dict.fromkeys(drop_keys))
    drop = set(ordered_drop)
    absorbed = [f for f in graph.factors if drop.intersection(_keys_of(f))]
    if not absorbed:
        return None, [], []

    keep = []
    for f in absorbed:
        for k in _keys_of(f):
            if k not in drop and k not in keep:
                keep.append(k)
    if not keep:
        return None, absorbed, []

    # Build the system over exactly the absorbed factors, ordering keep first.
    sub = type(graph)(list(absorbed))
    ordered = keep + [
        k for k in ordered_drop if any(k in _keys_of(f) for f in absorbed)
    ]
    system = _ordered_system(sub, values, ordered)
    if system is None:
        return None, absorbed, []
    H, b, layout = system

    n_keep = sum(values[k].dim for k in keep)
    H_kk = H[:n_keep, :n_keep]
    H_kd = H[:n_keep, n_keep:]
    H_dd = H[n_keep:, n_keep:]
    b_k, b_d = b[:n_keep], b[n_keep:]

    if H_dd.shape[0] == 0:
        H_marg, b_marg = H_kk, b_k
    else:
        # A pseudo-inverse rather than a solve: the dropped block can itself be
        # singular, for instance a landmark seen from a single keyframe, and
        # the right answer there is to contribute nothing rather than to raise.
        H_dd_inv = np.linalg.pinv(H_dd + np.eye(H_dd.shape[0]) * 1e-12)
        H_marg = H_kk - H_kd @ H_dd_inv @ H_kd.T
        b_marg = b_k - H_kd @ H_dd_inv @ b_d

    A, b_res = square_root(H_marg, b_marg, tolerance)
    if A is None:
        return None, absorbed, keep

    prior = LinearPriorFactor(
        keys=keep,
        linearisation_point={k: values[k].copy() for k in keep},
        sqrt_info=A,
        b=b_res,
        dims=[values[k].dim for k in keep],
    )
    return prior, absorbed, keep


def square_root(H: np.ndarray, b: np.ndarray, tolerance: float = RANK_TOLERANCE):
    """Factor ``(H, b)`` into a residual ``A x + b_res`` with ``A^T A = H``.

    Uses a symmetric eigendecomposition rather than a Cholesky, because the
    matrix is expected to be singular and Cholesky would simply fail.  Returns
    ``(None, None)`` if nothing survives the rank test.
    """
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    largest = float(w.max()) if w.size else 0.0
    if largest <= 0.0:
        return None, None
    keep = w > largest * tolerance
    if not np.any(keep):
        return None, None
    w_k = w[keep]
    V_k = V[:, keep]
    sqrt_w = np.sqrt(w_k)
    A = (sqrt_w[:, None] * V_k.T)
    b_res = (V_k.T @ b) / sqrt_w
    return A, b_res


def _ordered_system(graph, values: Values, ordered_keys):
    """Hessian and gradient of ``graph`` with variables in the given order."""
    present = [k for k in ordered_keys if k in values]
    if not present:
        return None
    # linearise() imposes its own ordering, so build a permutation into ours.
    system = linearise(graph, values, fixed=set())
    layout = system.layout
    perm = []
    for k in present:
        if k not in layout:
            return None
        off, dim = layout[k]
        perm.extend(range(off, off + dim))
    perm = np.asarray(perm, dtype=int)

    J = system.J[:, perm]
    H = np.asarray((J.T @ J).todense())
    b = J.T @ system.r
    out_layout, off = {}, 0
    for k in present:
        d = values[k].dim
        out_layout[k] = (off, d)
        off += d
    return H, np.asarray(b).reshape(-1), out_layout
