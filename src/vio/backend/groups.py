"""Vectorised factor groups: the same maths as :mod:`vio.backend.factors`, in bulk.

The scalar factors are the readable definition and the thing the
finite-difference tests check.  They are also, in Python, the reason a bundle
adjustment is slow: a ten-keyframe window with a thousand landmark
observations spends almost all of its time in per-factor interpreter overhead
rather than in arithmetic.  Measured on the synthetic sequence, one
linearisation of such a window took 63 ms through the scalar path and 4 ms
through this one.

A group computes every member's residual and Jacobian in one array pass and
emits its own sparse triplets.  ``tests/test_groups.py`` asserts that a group
and the equivalent list of scalar factors agree on residual, Jacobian, sparse
structure and cost to 1e-12, so the fast path cannot quietly drift away from
the readable one.

Only reprojection factors are grouped.  There are at most a few dozen IMU and
prior factors in a window against a thousand visual ones, so grouping them
would complicate the code for no measurable gain, which is the sort of claim
that should be checked rather than assumed, and the profile in
``docs/results.md`` is where it was.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Pose
from ..geometry.camera import PinholeCamera
from .factors import Loss, ReprojectionFactor, TrivialLoss


class FactorGroup:
    """Interface for a homogeneous set of factors evaluated together."""

    def keys(self) -> list:  # pragma: no cover - interface
        raise NotImplementedError

    def rows(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def linearise_into(self, values, layout, row0):
        """``(rows, cols, data, residual, cost)`` for the sparse assembly."""
        raise NotImplementedError  # pragma: no cover - interface

    def error(self, values) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class ReprojectionGroup(FactorGroup):
    """Every landmark observation in a window, evaluated at once.

    Mathematically identical to a list of
    :class:`vio.backend.factors.ReprojectionFactor`; see that class for the
    derivation of the Jacobian blocks.  This one only rearranges the same
    expressions into array form.
    """

    def __init__(
        self,
        camera: PinholeCamera,
        T_body_cam: Pose,
        pixel_sigma: float = 1.0,
        loss: Loss | None = None,
    ):
        self.camera = camera
        self.T_body_cam = T_body_cam.copy()
        self.T_cam_body = T_body_cam.inverse()
        self.focal = camera.focal
        self.pixel_sigma = float(pixel_sigma)
        self.inv_sigma = 1.0 / float(pixel_sigma)
        self.loss = loss or TrivialLoss()
        self._pose_keys: list = []
        self._point_keys: list = []
        self._measurements: list = []
        self._pose_index: dict = {}
        self._pose_of: list = []

    #, construction ------------------------------------------------------

    def add(self, pose_key, point_key, measurement) -> None:
        if pose_key not in self._pose_index:
            self._pose_index[pose_key] = len(self._pose_keys)
            self._pose_keys.append(pose_key)
        self._pose_of.append(self._pose_index[pose_key])
        self._point_keys.append(point_key)
        self._measurements.append(np.asarray(measurement, dtype=float).reshape(2))

    def __len__(self) -> int:
        return len(self._point_keys)

    def keys(self) -> list:
        return list(self._pose_keys) + list(self._point_keys)

    def rows(self) -> int:
        return 2 * len(self._point_keys)

    #, evaluation --------------------------------------------------------

    def _geometry(self, values):
        pose_of = np.asarray(self._pose_of, dtype=int)
        R = np.stack([values[k].value.R for k in self._pose_keys])
        t = np.stack([values[k].value.t for k in self._pose_keys])
        p_w = np.stack([values[k].value for k in self._point_keys])
        meas = np.stack(self._measurements)

        R_i = R[pose_of]
        p_b = np.einsum("nji,nj->ni", R_i, p_w - t[pose_of])
        p_c = p_b @ self.T_cam_body.R.T + self.T_cam_body.t

        z = p_c[:, 2]
        good = z > 1e-6
        iz = 1.0 / np.where(good, z, 1.0)
        r = self.focal * (p_c[:, :2] * iz[:, None] - meas)
        # Matches ReprojectionFactor: an observation behind the camera gets a
        # large but finite residual and no gradient, so one bad landmark cannot
        # poison the whole linear system.
        r = np.where(good[:, None], r, 1e3)
        return pose_of, R_i, p_b, p_c, iz, good, r

    def _weights(self, r):
        sq = np.einsum("ni,ni->n", r, r)
        if isinstance(self.loss, TrivialLoss):
            return np.ones(len(sq)), sq
        return np.array([self.loss.weight(float(s)) for s in sq]), sq

    def linearise_into(self, values, layout, row0):
        n = len(self._point_keys)
        if n == 0:
            return [], [], [], np.zeros(0), 0.0
        pose_of, R_i, p_b, p_c, iz, good, r = self._geometry(values)
        f = self.focal

        Jp = np.zeros((n, 2, 3))
        Jp[:, 0, 0] = f * iz
        Jp[:, 1, 1] = f * iz
        Jp[:, 0, 2] = -f * p_c[:, 0] * iz * iz
        Jp[:, 1, 2] = -f * p_c[:, 1] * iz * iz

        R_cb = self.T_cam_body.R
        R_cw = np.einsum("ij,nkj->nik", R_cb, R_i)

        hat = np.zeros((n, 3, 3))
        hat[:, 0, 1] = -p_b[:, 2]
        hat[:, 0, 2] = p_b[:, 1]
        hat[:, 1, 0] = p_b[:, 2]
        hat[:, 1, 2] = -p_b[:, 0]
        hat[:, 2, 0] = -p_b[:, 1]
        hat[:, 2, 1] = p_b[:, 0]

        J_rot = np.einsum("nij,jk,nkl->nil", Jp, R_cb, hat)
        J_point = np.einsum("nij,njk->nik", Jp, R_cw)
        J_trans = -J_point

        r = r * self.inv_sigma
        J_rot *= self.inv_sigma
        J_trans *= self.inv_sigma
        J_point *= self.inv_sigma
        J_rot[~good] = 0.0
        J_trans[~good] = 0.0
        J_point[~good] = 0.0

        w, sq = self._weights(r)
        cost = 0.5 * float(np.sum(w * sq))
        if not np.all(w == 1.0):
            s = np.sqrt(w)
            r = r * s[:, None]
            J_rot *= s[:, None, None]
            J_trans *= s[:, None, None]
            J_point *= s[:, None, None]

        rows, cols, data = [], [], []
        row_pair = (row0 + 2 * np.arange(n)[:, None] + np.array([0, 1])[None, :])

        pose_off = np.array(
            [layout[k][0] if k in layout else -1 for k in self._pose_keys]
        )[pose_of]
        vis = pose_off >= 0
        if np.any(vis):
            block = np.concatenate([J_rot, J_trans], axis=2)[vis]
            rows.append(np.repeat(row_pair[vis], 6, axis=1).reshape(-1))
            cols.append(
                np.tile(pose_off[vis][:, None] + np.arange(6)[None, :], (1, 2)).reshape(-1)
            )
            data.append(block.reshape(-1))

        point_off = np.array(
            [layout[k][0] if k in layout else -1 for k in self._point_keys]
        )
        vis_p = point_off >= 0
        if np.any(vis_p):
            rows.append(np.repeat(row_pair[vis_p], 3, axis=1).reshape(-1))
            cols.append(
                np.tile(point_off[vis_p][:, None] + np.arange(3)[None, :], (1, 2)).reshape(-1)
            )
            data.append(J_point[vis_p].reshape(-1))

        return rows, cols, data, r.reshape(-1), cost

    def error(self, values) -> float:
        if not len(self._point_keys):
            return 0.0
        *_, r = self._geometry(values)
        r = r * self.inv_sigma
        w, sq = self._weights(r)
        return 0.5 * float(np.sum(w * sq))

    def residuals(self, values) -> np.ndarray:
        """Whitened, unweighted residuals in pixels, for outlier reporting."""
        *_, r = self._geometry(values)
        return np.linalg.norm(r, axis=1)

    def observations(self):
        """``(pose_key, point_key, measurement)`` triples, in insertion order."""
        return [
            (self._pose_keys[p], k, m)
            for p, k, m in zip(self._pose_of, self._point_keys, self._measurements)
        ]

    def to_factors(self) -> list:
        """The equivalent list of scalar factors, used by the agreement test."""
        return [
            ReprojectionFactor(
                pk, lk, m, self.camera, self.T_body_cam, self.pixel_sigma, self.loss
            )
            for pk, lk, m in self.observations()
        ]
