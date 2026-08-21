#!/usr/bin/env python
"""Recompute every metric from the trajectory files on disk.

The runs in ``results/`` already contain their metrics but those were computed
by the same process that produced the trajectory.  This script starts again
from the ``.tum`` files alone, which is the form a reader can check with their
own tools.  If it disagrees with the stored numbers, something is wrong with
one of them and the difference is printed rather than reconciled.

    python evaluation/evaluate.py --results results
    python evaluation/evaluate.py --results results --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio.evaluation.metrics import ate, drift_rate, rpe  # noqa: E402
from vio.io.trajectory import read_tum  # noqa: E402


def evaluate_one(traj_path: Path, gt_path: Path, mode: str) -> dict:
    est_t, est_p = read_tum(traj_path)
    gt_t, gt_p = read_tum(gt_path)
    alignment = "se3" if mode == "vio" else "sim3"

    a = ate(est_t, est_p, gt_t, gt_p, mode=alignment)
    a_other = ate(est_t, est_p, gt_t, gt_p, mode="sim3" if alignment == "se3" else "se3")
    d = drift_rate(est_t, est_p, gt_t, gt_p, mode=alignment)
    r = rpe(est_t, est_p, gt_t, gt_p, delta=1.0, unit="m",
            scale=None if alignment == "se3" else a.scale)
    r10 = rpe(est_t, est_p, gt_t, gt_p, delta=10.0, unit="m",
              scale=None if alignment == "se3" else a.scale)

    return {
        "alignment": alignment,
        "poses": int(len(est_t)),
        "matched": a.n_matched,
        "ate_rmse_m": a.translation.rmse,
        "ate_median_m": a.translation.median,
        "ate_max_m": a.translation.max,
        "ate_rot_rmse_deg": a.rotation_deg.rmse,
        "scale": a_other.scale if alignment == "se3" else a.scale,
        "scale_error_percent": 100.0 * abs(
            (a_other.scale if alignment == "se3" else a.scale) - 1.0
        ),
        "rpe_1m_rmse_m": r.translation.rmse,
        "rpe_1m_drift_percent": r.drift_percent.median,
        "rpe_10m_rmse_m": r10.translation.rmse,
        "distance_m": d["distance_m"],
        "drift_final_percent_per_m": 100.0 * d["final_per_metre"],
        "drift_mean_percent_per_m": 100.0 * d["mean_per_metre"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--format", choices=["table", "markdown", "json"], default="table")
    ap.add_argument("--out", default="")
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="disagreement with the stored metrics worth reporting")
    args = ap.parse_args()

    root = Path(args.results)
    rows = []
    disagreements = []
    for traj in sorted(root.glob("*_v*.tum")):
        stem = traj.stem
        if stem.endswith("_gt"):
            continue
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("vio", "vo"):
            continue
        name, mode = parts
        gt = root / f"{name}_gt.tum"
        if not gt.exists():
            print(f"  skipping {stem}: no ground truth alongside it")
            continue
        row = {"sequence": name, "mode": mode}
        row.update(evaluate_one(traj, gt, mode))
        rows.append(row)

        stored_path = root / f"{stem}.json"
        if stored_path.exists():
            stored = json.loads(stored_path.read_text(encoding="utf-8"))
            key = f"ate_{row['alignment']}"
            if key in stored.get("metrics", {}):
                stored_rmse = stored["metrics"][key]["ate_translation_m"]["rmse"]
                delta = abs(stored_rmse - row["ate_rmse_m"])
                if delta > args.tolerance:
                    disagreements.append((stem, stored_rmse, row["ate_rmse_m"], delta))

    if not rows:
        print(f"no trajectories found under {root}")
        return 1

    rows.sort(key=lambda r: (r["sequence"], r["mode"]))

    if args.format == "json":
        text = json.dumps(rows, indent=2, default=float)
    elif args.format == "markdown":
        head = ("| sequence | mode | align | ATE rmse (m) | ATE median (m) | "
                "rot rmse (deg) | scale err (%) | RPE 1 m (m) | drift (%/m) |")
        sep = "| --- " * 9 + "|"
        lines = [head, sep]
        for r in rows:
            lines.append(
                f"| {r['sequence']} | {r['mode']} | {r['alignment']} | "
                f"{r['ate_rmse_m']:.4f} | {r['ate_median_m']:.4f} | "
                f"{r['ate_rot_rmse_deg']:.3f} | {r['scale_error_percent']:.2f} | "
                f"{r['rpe_1m_rmse_m']:.4f} | {r['drift_final_percent_per_m']:.3f} |"
            )
        text = "\n".join(lines)
    else:
        lines = [f"{'sequence':18s} {'mode':4s} {'align':5s} {'ATE rmse':>9s} "
                 f"{'median':>8s} {'rot deg':>8s} {'scale %':>8s} {'RPE 1m':>8s} "
                 f"{'%/m':>7s}"]
        for r in rows:
            lines.append(
                f"{r['sequence']:18s} {r['mode']:4s} {r['alignment']:5s} "
                f"{r['ate_rmse_m']:9.4f} {r['ate_median_m']:8.4f} "
                f"{r['ate_rot_rmse_deg']:8.3f} {r['scale_error_percent']:8.2f} "
                f"{r['rpe_1m_rmse_m']:8.4f} {r['drift_final_percent_per_m']:7.3f}"
            )
        for mode in ("vio", "vo"):
            subset = [r for r in rows if r["mode"] == mode]
            if subset:
                lines.append(
                    f"{'mean ' + mode:18s} {'':4s} {'':5s} "
                    f"{np.mean([r['ate_rmse_m'] for r in subset]):9.4f} "
                    f"{np.mean([r['ate_median_m'] for r in subset]):8.4f} "
                    f"{np.mean([r['ate_rot_rmse_deg'] for r in subset]):8.3f} "
                    f"{np.mean([r['scale_error_percent'] for r in subset]):8.2f} "
                    f"{np.mean([r['rpe_1m_rmse_m'] for r in subset]):8.4f} "
                    f"{np.mean([r['drift_final_percent_per_m'] for r in subset]):7.3f}"
                )
        text = "\n".join(lines)

    print(text)
    if disagreements:
        print("\nstored metrics disagree with a fresh evaluation:")
        for stem, stored, fresh, delta in disagreements:
            print(f"  {stem}: stored {stored:.6f}, recomputed {fresh:.6f} "
                  f"(delta {delta:.2e})")
        return 1
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
