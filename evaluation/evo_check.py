#!/usr/bin/env python
"""Cross-check our trajectory metrics against evo on the same files.

``vio.evaluation.metrics`` implements association, Umeyama alignment, ATE and
RPE from scratch, because a number you cannot derive yourself is a number you
cannot defend.  That leaves the obvious objection: an implementation which
agrees only with itself is not evidence.  ``evo`` is a widely used, independent
package with its own association, its own alignment and its own statistics, so
running both over the same ``.tum`` files and printing the difference is the
strongest available answer.

Anything above ``--tolerance`` is reported and makes the script exit non-zero.

    python evaluation/evo_check.py --results results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio.evaluation.metrics import ate  # noqa: E402
from vio.io.trajectory import read_tum  # noqa: E402


def to_evo(times, poses):
    from evo.core.trajectory import PoseTrajectory3D

    return PoseTrajectory3D(
        poses_se3=[p.matrix() for p in poses], timestamps=np.asarray(times)
    )


def compare(traj_path: Path, gt_path: Path, mode: str, max_diff: float):
    from evo.core import metrics as evo_metrics
    from evo.core import sync
    from evo.main_ape import ape

    est_t, est_p = read_tum(traj_path)
    gt_t, gt_p = read_tum(gt_path)
    ref, est = to_evo(gt_t, gt_p), to_evo(est_t, est_p)
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=max_diff)

    out = []
    for alignment, correct_scale in (("se3", False), ("sim3", True)):
        theirs = ape(
            ref_s, est_s,
            pose_relation=evo_metrics.PoseRelation.translation_part,
            align=True, correct_scale=correct_scale,
        ).stats
        mine = ate(est_t, est_p, gt_t, gt_p, mode=alignment, max_difference=max_diff)
        out.append({
            "alignment": alignment,
            "ours_rmse": mine.translation.rmse,
            "evo_rmse": float(theirs["rmse"]),
            "ours_mean": mine.translation.mean,
            "evo_mean": float(theirs["mean"]),
            "ours_matched": mine.n_matched,
            "evo_matched": int(ref_s.num_poses),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--max-diff", type=float, default=0.02)
    args = ap.parse_args()

    try:
        import evo  # noqa: F401
    except ImportError:
        print("evo is not installed; pip install evo")
        return 2

    root = Path(args.results)
    worst = 0.0
    checked = 0
    failures = []
    for traj in sorted(root.glob("*_v*.tum")):
        stem = traj.stem
        if stem.endswith("_gt"):
            continue
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("vio", "vo"):
            continue
        gt = root / f"{parts[0]}_gt.tum"
        if not gt.exists():
            continue
        for row in compare(traj, gt, parts[1], args.max_diff):
            checked += 1
            delta = abs(row["ours_rmse"] - row["evo_rmse"])
            worst = max(worst, delta)
            flag = "ok  " if delta <= args.tolerance else "FAIL"
            print(f"{flag} {stem:26s} {row['alignment']:5s} "
                  f"ours {row['ours_rmse']:.9f}  evo {row['evo_rmse']:.9f}  "
                  f"delta {delta:.2e}  matched {row['ours_matched']}/"
                  f"{row['evo_matched']}")
            if delta > args.tolerance:
                failures.append((stem, row["alignment"], delta))

    if not checked:
        print(f"no trajectories found under {root}")
        return 1
    print(f"\n{checked} comparisons, worst disagreement {worst:.2e} m")
    if failures:
        print("disagreements above tolerance:")
        for stem, alignment, delta in failures:
            print(f"  {stem} [{alignment}] {delta:.2e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
