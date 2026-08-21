"""A factor graph and the sparse linear system it linearises to."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .variables import PointVariable, Values


@dataclass
class FactorGraph:
    """A bag of factors.  Ordering is irrelevant; the solver imposes its own."""

    factors: list = field(default_factory=list)

    def add(self, factor):
        self.factors.append(factor)
        return factor

    def extend(self, factors):
        self.factors.extend(factors)

    def __len__(self) -> int:
        return len(self.factors)

    def keys(self) -> set:
        out = set()
        for f in self.factors:
            out.update(_keys_of(f))
        return out

    def error(self, values: Values) -> float:
        return float(sum(f.error(values) for f in self.factors))

    def errors_by_type(self, values: Values) -> dict:
        """Cost split by factor class, which is how a bad weighting gets found."""
        out = {}
        for f in self.factors:
            name = type(f).__name__
            out[name] = out.get(name, 0.0) + f.error(values)
        return out

    def factors_touching(self, key) -> list:
        return [f for f in self.factors if key in _keys_of(f)]

    def remove(self, factors) -> None:
        drop = set(id(f) for f in factors)
        self.factors = [f for f in self.factors if id(f) not in drop]

    def dimension(self) -> int:
        return sum(
            f.rows() if hasattr(f, "linearise_into") else f.dim for f in self.factors
        )


@dataclass
class LinearSystem:
    """The result of linearising a graph: ``J``, ``r`` and where things live."""

    J: sp.csr_matrix
    r: np.ndarray
    cost: float
    #: key -> (offset, dim) in the tangent vector, for the free variables only.
    layout: dict
    #: Keys eliminated by Schur complement, in layout order.
    point_keys: list
    #: Keys kept in the reduced system, in layout order.
    kept_keys: list
    n_kept: int


def linearise(graph: FactorGraph, values: Values, fixed=frozenset()) -> LinearSystem:
    """Build the whitened, robustly reweighted Jacobian and residual.

    Free variables are ordered with everything that is *not* a landmark first,
    so the Schur complement is a contiguous trailing block rather than a
    scatter-gather.  On a 2000-landmark bundle that ordering alone is worth
    about a factor of three in the linear solve, because scipy's sparse
    routines keep the reduced system in one dense corner.
    """
    fixed = set(fixed)
    touched = graph.keys() - fixed

    kept_keys, point_keys = [], []
    for key in sorted(touched, key=_sort_key):
        (point_keys if isinstance(values[key], PointVariable) else kept_keys).append(key)

    layout, offset = {}, 0
    for key in kept_keys + point_keys:
        dim = values[key].dim
        layout[key] = (offset, dim)
        offset += dim
    n_cols = offset
    n_kept = sum(values[k].dim for k in kept_keys)

    free_layout = {k: v for k, v in layout.items() if k not in fixed}
    rows, cols, data = [], [], []
    residuals = []
    cost = 0.0
    row = 0
    for f in graph.factors:
        if hasattr(f, "linearise_into"):
            rr, cc, dd, r, c = f.linearise_into(values, free_layout, row)
            rows.extend(rr)
            cols.extend(cc)
            data.extend(dd)
            residuals.append(r)
            cost += c
            row += f.rows()
            continue
        r, Js, c = f.linearise(values)
        cost += c
        residuals.append(r)
        for key, J in zip(f.keys, Js):
            if key in fixed:
                continue
            off, dim = layout[key]
            rr, cc = np.nonzero(J)
            if len(rr) == 0:
                continue
            rows.append(rr + row)
            cols.append(cc + off)
            data.append(J[rr, cc])
        row += f.dim

    if residuals:
        r_all = np.concatenate(residuals)
    else:
        r_all = np.zeros(0)
    if data:
        J = sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(row, n_cols),
        )
    else:
        J = sp.csr_matrix((row, n_cols))

    return LinearSystem(J, r_all, cost, layout, point_keys, kept_keys, n_kept)


def _keys_of(factor):
    """Keys a factor or a vectorised group touches."""
    keys = factor.keys
    return keys() if callable(keys) else keys


def _sort_key(key):
    """Stable ordering for mixed key types, so a solve is reproducible."""
    return (0, key) if isinstance(key, str) else (1, str(key))
