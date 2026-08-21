"""Pose-graph optimisation for loop correction.

Once a loop is detected the correction has to be distributed over the whole
trajectory rather than applied as a jump at the closing keyframe.  That is a
pose graph: one node per keyframe, one edge per consecutive pair carrying the
relative motion the estimator reported and one edge per verified loop.  The
graph is the same machinery as the sliding-window back end, the same
``RelativePoseFactor``, the same Levenberg-Marquardt, over a different set of
variables.

Two decisions worth stating.

**Four degrees of freedom, not six.**  A visual-inertial system observes
gravity, so roll and pitch are absolutely referenced and do not drift; only
position and yaw do.  A six-degree-of-freedom pose graph does not know that and
will happily tilt the whole map to close a loop, trading a well-observed
quantity for a poorly observed one.  VINS-Mono handles this with a purpose-built
four-degree-of-freedom parameterisation.  Here the same effect comes from a
:class:`vio.backend.factors.GravityDirectionFactor` on every node, which pins
the body-frame gravity direction and leaves yaw free, rank two per node, with
the null space exactly along the gravity axis.  It needs no new variable type,
and ``scripts/run_loop_closure.py --no-gravity-lock`` measures what
happens with it off.

**The robustness comes from verification, not from the kernel.**  The first
version of this used a Cauchy kernel on loop edges, on the reasonable-sounding
grounds that a loop retrieved by appearance might be wrong.  It does not work,
and the reason is worth stating because it is easy to get backwards: the
residual of a loop edge at the initial linearisation point is large *by
construction*.  That residual is the accumulated drift, which is the whole
thing the edge exists to remove.  Measured on a 60-node loop with 0.37 m of
drift, the Cauchy weight came out at 0.055 and the correction was four
hundredths of a millimetre, the kernel had classified the answer as an
outlier.

A kernel cannot distinguish "large because the trajectory drifted" from "large
because the loop is wrong"; only the geometric verification in
:mod:`vio.loop_closure.detector` can and that is where the defence belongs.
What the kernel is for here is bounding the damage from an edge that survived
verification and is still catastrophically wrong, so it is a Huber with a
generous threshold rather than anything redescending.  ``CauchyLoss`` is still
selectable for a caller who wants it and knows why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend import factors as F
from ..backend.graph import FactorGraph
from ..backend.solver import SolverOptions, optimise
from ..backend.variables import PoseVariable, Values


@dataclass
class PoseGraphConfig:
    #: Standard deviations for a sequential edge: rotation (rad), position (m).
    #: Scaled by the length of the edge, so a long gap is trusted less.
    odometry_sigma: tuple = (0.004, 0.02)
    #: ...and for a loop edge, which is a single-shot PnP rather than a bundle.
    loop_sigma: tuple = (0.03, 0.08)
    #: Huber threshold on loop edges, in whitened units.  Generous on purpose:
    #: see the module docstring.  Five sigma still bounds an edge that is
    #: wildly wrong while leaving a genuine multi-sigma correction alone.
    loop_kernel: float = 5.0
    #: ``huber`` or ``cauchy``.  Cauchy is retained but is not the default,
    #: because it suppresses genuine corrections; the docstring has the number.
    loop_loss: str = "huber"
    #: Standard deviation of the gravity-direction lock, dimensionless.
    gravity_sigma: float = 0.02
    lock_gravity: bool = True
    max_iterations: int = 40


@dataclass
class PoseGraphResult:
    poses: dict
    initial_cost: float
    final_cost: float
    iterations: int
    seconds: float
    #: How far each node moved, metres.
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def max_displacement(self) -> float:
        return float(self.displacement.max()) if self.displacement.size else 0.0


def build(keyframe_ids, poses, closures, config: PoseGraphConfig | None = None):
    """Assemble the graph and its initial values.

    ``poses`` maps keyframe id to the body pose the estimator produced.
    ``closures`` are :class:`vio.loop_closure.detector.LoopClosure` objects.
    """
    cfg = config or PoseGraphConfig()
    g = FactorGraph()
    values = Values()
    ids = list(keyframe_ids)
    for k in ids:
        values[f"p{k}"] = PoseVariable(poses[k])

    sr, st = cfg.odometry_sigma
    for a, b in zip(ids[:-1], ids[1:]):
        rel = poses[a].inverse() * poses[b]
        span = max(float(np.linalg.norm(rel.t)), 0.02)
        info = np.diag(
            np.concatenate([
                np.full(3, 1.0 / (sr * np.sqrt(span))),
                np.full(3, 1.0 / (st * np.sqrt(span))),
            ])
        )
        g.add(F.RelativePoseFactor(f"p{a}", f"p{b}", rel, info))

    lr, lt = cfg.loop_sigma
    loop_info = np.diag(
        np.concatenate([np.full(3, 1.0 / lr), np.full(3, 1.0 / lt)])
    )
    kernel = (
        F.CauchyLoss(cfg.loop_kernel) if cfg.loop_loss == "cauchy"
        else F.HuberLoss(cfg.loop_kernel)
    )
    for c in closures:
        if c.match_id not in poses or c.query_id not in poses:
            continue
        g.add(F.RelativePoseFactor(
            f"p{c.match_id}", f"p{c.query_id}", c.relative_pose, loop_info, kernel
        ))

    if cfg.lock_gravity:
        info = np.eye(3) / cfg.gravity_sigma
        for k in ids:
            g.add(F.GravityDirectionFactor(f"p{k}", poses[k].R, info))

    # The first node is held: a pose graph is otherwise free to translate and
    # rotate as a whole and the solver would wander through that null space.
    g.add(F.PosePriorFactor(f"p{ids[0]}", poses[ids[0]], np.eye(6) * 1e4))
    return g, values


def optimise_graph(keyframe_ids, poses, closures,
                   config: PoseGraphConfig | None = None) -> PoseGraphResult:
    """Build and solve the pose graph."""
    cfg = config or PoseGraphConfig()
    ids = list(keyframe_ids)
    if len(ids) < 3 or not closures:
        return PoseGraphResult(dict(poses), 0.0, 0.0, 0, 0.0)

    g, values = build(ids, poses, closures, cfg)
    res = optimise(g, values, options=SolverOptions(max_iterations=cfg.max_iterations))
    out = {k: res.values.pose(f"p{k}") for k in ids}
    disp = np.array([float(np.linalg.norm(out[k].t - poses[k].t)) for k in ids])
    return PoseGraphResult(
        poses=out,
        initial_cost=res.initial_cost,
        final_cost=res.final_cost,
        iterations=res.iterations,
        seconds=res.seconds,
        displacement=disp,
    )


def apply_correction(frame_times, frame_poses, keyframe_times, keyframe_ids,
                     original, corrected):
    """Push a keyframe-level correction out to every frame.

    Each frame keeps its pose *relative to the nearest preceding keyframe*, so
    the fine-grained motion the estimator produced is preserved and only the
    accumulated drift is redistributed::

        T_new = T_corrected[anchor] * (T_original[anchor]^-1 * T_frame)

    Frames before the first keyframe are anchored to it instead.  The
    correction is piecewise constant between keyframes rather than
    interpolated: interpolating would smooth the joins but would also alter the
    relative motion inside each interval, which is the one part of the estimate
    that was not drifting.
    """
    kf_times = np.asarray(keyframe_times, dtype=float)
    ids = list(keyframe_ids)
    if not ids:
        return list(frame_poses)

    out = []
    for t, T in zip(frame_times, frame_poses):
        if T is None:
            out.append(None)
            continue
        i = int(np.searchsorted(kf_times, float(t), side="right")) - 1
        i = min(max(i, 0), len(ids) - 1)
        anchor = ids[i]
        out.append(corrected[anchor] * (original[anchor].inverse() * T))
    return out
