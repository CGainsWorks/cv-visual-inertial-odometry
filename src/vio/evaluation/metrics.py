"""Trajectory error metrics: association, alignment, ATE, RPE, drift rate.

Written out rather than shelled out to ``evo``, for one reason: a number you
cannot derive yourself is a number you cannot defend.  ``evo`` is then used as
an *independent referee*, ``evaluation/evo_check.py`` runs both on the same
files and reports the disagreement, which in testing is at the 1e-9 level.  If
the two ever diverge, that is a finding rather than an inconvenience.

Three definitions, because they are where evaluations quietly differ:

**Association.**  Estimated and reference trajectories are sampled on different
clocks.  Each estimate is matched to the nearest reference sample, and pairs
further apart than ``max_difference`` are dropped rather than interpolated.
EuRoC's ground truth runs at 200 Hz, so the worst unmatched error is 2.5 ms of
motion.

**Alignment.**  A monocular system has no scale and no absolute frame, so its
trajectory has to be aligned before the error means anything.  ``sim3``
(Umeyama with scale) is the only honest choice for visual-only output;
``se3`` (Umeyama without scale) is the right one for visual-inertial output,
because scale is something a visual-inertial system is claiming to know and
letting the alignment absorb a scale error would hide exactly the failure the
IMU is there to prevent.  Which mode produced a number is recorded next to it
everywhere in this repository.

**Relative pose error.**  Computed over fixed *distance* intervals by default
rather than fixed time intervals.  A drone that hovers for ten seconds
accumulates almost no error in that time, and time-based RPE rewards it for
that; distance-based RPE does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose, so3


@dataclass
class Statistics:
    """The usual summary of an error series."""

    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    n: int

    @staticmethod
    def of(errors) -> "Statistics":
        e = np.asarray(errors, dtype=float).reshape(-1)
        if e.size == 0:
            nan = float("nan")
            return Statistics(nan, nan, nan, nan, nan, nan, 0)
        return Statistics(
            rmse=float(np.sqrt(np.mean(e ** 2))),
            mean=float(np.mean(e)),
            median=float(np.median(e)),
            std=float(np.std(e)),
            min=float(np.min(e)),
            max=float(np.max(e)),
            n=int(e.size),
        )

    def as_dict(self) -> dict:
        return {
            "rmse": self.rmse, "mean": self.mean, "median": self.median,
            "std": self.std, "min": self.min, "max": self.max, "n": self.n,
        }


@dataclass
class AteResult:
    translation: Statistics
    rotation_deg: Statistics
    scale: float
    alignment: str
    n_matched: int
    #: Per-sample translation error, for plotting.
    errors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def as_dict(self) -> dict:
        return {
            "ate_translation_m": self.translation.as_dict(),
            "ate_rotation_deg": self.rotation_deg.as_dict(),
            "alignment": self.alignment,
            "alignment_scale": self.scale,
            "n_matched": self.n_matched,
        }


@dataclass
class RpeResult:
    translation: Statistics
    rotation_deg: Statistics
    #: Translation error as a fraction of the interval length, per cent.
    drift_percent: Statistics
    delta: float
    unit: str
    n_pairs: int

    def as_dict(self) -> dict:
        return {
            "rpe_translation_m": self.translation.as_dict(),
            "rpe_rotation_deg": self.rotation_deg.as_dict(),
            "rpe_drift_percent": self.drift_percent.as_dict(),
            "delta": self.delta,
            "delta_unit": self.unit,
            "n_pairs": self.n_pairs,
        }


#: Relative slack on the association tolerance.
#:
#: A sample sitting exactly on the boundary is a knife edge, and floating-point
#: noise decides which side it falls.  That is not hypothetical: writing a
#: trajectory to a TUM file and reading it back perturbs the timestamps by
#: about 3e-14 s, which was enough to flip one borderline pair on EuRoC MH_01
#: and move the reported ATE by 1.5 mm.  The tolerance is a modelling choice
#: rather than an exact quantity, so it is applied with a slack far above that
#: noise and far below anything that could change which samples are genuinely
#: close.
ASSOCIATION_SLACK = 1e-9


def associate(est_times, ref_times, max_difference: float = 0.02):
    """Nearest-neighbour association between two time series.

    Returns ``(est_indices, ref_indices)``.  Each estimate is used at most
    once, and each reference sample at most once: without the second condition
    a fast estimate against slow ground truth maps many estimates onto one
    reference sample and the error statistics silently reweight themselves.
    """
    est_times = np.asarray(est_times, dtype=float)
    ref_times = np.asarray(ref_times, dtype=float)
    if est_times.size == 0 or ref_times.size == 0:
        return np.zeros(0, int), np.zeros(0, int)

    idx = np.searchsorted(ref_times, est_times)
    idx = np.clip(idx, 1, len(ref_times) - 1)
    left, right = ref_times[idx - 1], ref_times[idx]
    nearest = np.where(np.abs(est_times - left) <= np.abs(right - est_times),
                       idx - 1, idx)
    diff = np.abs(ref_times[nearest] - est_times)
    ok = diff <= max_difference * (1.0 + ASSOCIATION_SLACK)

    est_idx, ref_idx, used = [], [], set()
    for i in np.nonzero(ok)[0]:
        j = int(nearest[i])
        if j in used:
            continue
        used.add(j)
        est_idx.append(int(i))
        ref_idx.append(j)
    return np.asarray(est_idx, int), np.asarray(ref_idx, int)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform taking ``src`` onto ``dst``.

    Umeyama 1991.  Returns ``(R, t, s)`` with ``dst ~= s R src + t``.  The
    determinant correction matters: without it a noisy or nearly planar point
    set can produce a reflection, which fits better and is not a rigid motion.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    if n < 3:
        return np.eye(3), np.zeros(3), 1.0

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = float(np.mean(np.sum(sc ** 2, axis=1)))
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def align_trajectory(est_poses, ref_poses, mode: str = "se3"):
    """Align an estimated trajectory to a reference.

    ``mode`` is ``"sim3"``, ``"se3"`` or ``"none"``.
    """
    if mode == "none":
        return [p.copy() for p in est_poses], np.eye(3), np.zeros(3), 1.0
    src = np.stack([p.t for p in est_poses])
    dst = np.stack([p.t for p in ref_poses])
    R, t, s = umeyama(src, dst, with_scale=(mode == "sim3"))
    out = [Pose(so3.project(R @ p.R), s * (R @ p.t) + t) for p in est_poses]
    return out, R, t, s


def ate(est_times, est_poses, ref_times, ref_poses,
        mode: str = "se3", max_difference: float = 0.02) -> AteResult:
    """Absolute trajectory error after a single global alignment."""
    ei, ri = associate(est_times, ref_times, max_difference)
    if len(ei) < 3:
        empty = Statistics.of([])
        return AteResult(empty, empty, 1.0, mode, len(ei))

    e_poses = [est_poses[i] for i in ei]
    r_poses = [ref_poses[i] for i in ri]
    aligned, _, _, s = align_trajectory(e_poses, r_poses, mode)

    trans = np.linalg.norm(
        np.stack([a.t for a in aligned]) - np.stack([r.t for r in r_poses]), axis=1
    )
    rot = np.array([
        np.degrees(so3.angle_between(a.R, r.R)) for a, r in zip(aligned, r_poses)
    ])
    return AteResult(
        translation=Statistics.of(trans),
        rotation_deg=Statistics.of(rot),
        scale=float(s),
        alignment=mode,
        n_matched=len(ei),
        errors=trans,
        times=np.asarray(est_times, dtype=float)[ei],
    )


def path_length(poses) -> np.ndarray:
    """Cumulative distance travelled, one entry per pose."""
    p = np.stack([q.t for q in poses])
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def rpe(est_times, est_poses, ref_times, ref_poses,
        delta: float = 1.0, unit: str = "m",
        max_difference: float = 0.02, scale: float | None = None) -> RpeResult:
    """Relative pose error over intervals of ``delta`` metres or seconds.

    ``scale`` applies a known similarity scale to the estimate before
    comparing, which is what a visual-only trajectory needs.  Passing ``None``
    for a visual-only run silently reports its arbitrary scale as drift.
    """
    ei, ri = associate(est_times, ref_times, max_difference)
    if len(ei) < 4:
        empty = Statistics.of([])
        return RpeResult(empty, empty, empty, delta, unit, 0)

    e_poses = [est_poses[i] for i in ei]
    r_poses = [ref_poses[i] for i in ri]
    if scale is not None and scale != 1.0:
        e_poses = [Pose(p.R, scale * p.t) for p in e_poses]

    if unit == "m":
        axis = path_length(r_poses)
    else:
        axis = np.asarray(ref_times, dtype=float)[ri]
        axis = axis - axis[0]

    trans, rot, frac = [], [], []
    j = 0
    for i in range(len(e_poses)):
        target = axis[i] + delta
        if target > axis[-1]:
            break
        j = max(j, i + 1)
        while j < len(axis) - 1 and axis[j] < target:
            j += 1
        if j <= i:
            continue
        d_est = e_poses[i].inverse() * e_poses[j]
        d_ref = r_poses[i].inverse() * r_poses[j]
        err = d_ref.inverse() * d_est
        span = float(np.linalg.norm(d_ref.t))
        trans.append(float(np.linalg.norm(err.t)))
        rot.append(float(np.degrees(np.linalg.norm(so3.log(err.R)))))
        if span > 1e-6:
            frac.append(100.0 * trans[-1] / span)

    return RpeResult(
        translation=Statistics.of(trans),
        rotation_deg=Statistics.of(rot),
        drift_percent=Statistics.of(frac),
        delta=delta,
        unit=unit,
        n_pairs=len(trans),
    )


def drift_rate(est_times, est_poses, ref_times, ref_poses,
               mode: str = "se3", max_difference: float = 0.02) -> dict:
    """Final and mean absolute error per metre travelled.

    The headline number the brief asks for.  Two versions are reported because
    they answer different questions: the *final* error per metre is what a
    navigation system cares about, and the *mean* is what a mapping system
    cares about.  Quoting only the smaller of the two is the usual sleight of
    hand.
    """
    ei, ri = associate(est_times, ref_times, max_difference)
    if len(ei) < 3:
        return {"distance_m": 0.0, "final_error_m": float("nan"),
                "final_per_metre": float("nan"), "mean_per_metre": float("nan")}
    e_poses = [est_poses[i] for i in ei]
    r_poses = [ref_poses[i] for i in ri]
    aligned, _, _, _ = align_trajectory(e_poses, r_poses, mode)
    dist = path_length(r_poses)
    total = float(dist[-1])
    errs = np.linalg.norm(
        np.stack([a.t for a in aligned]) - np.stack([r.t for r in r_poses]), axis=1
    )
    return {
        "distance_m": total,
        "final_error_m": float(errs[-1]),
        "final_per_metre": float(errs[-1] / total) if total > 1e-9 else float("nan"),
        "mean_per_metre": float(errs.mean() / total) if total > 1e-9 else float("nan"),
    }
