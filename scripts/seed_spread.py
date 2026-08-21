#!/usr/bin/env python
"""Does the answer move when only RANSAC's random draws change?

The measured answer is no.  The reason the question needed asking is a bug
this repository used to have: two runs of the same commit at the same seed
once gave 0.257 m and 0.362 m of absolute trajectory error on EuRoC V1_01.
The obvious suspect was the random sampling inside OpenCV's RANSAC, which the
front end uses for its two-view check and for PnP, so the seed was made
configurable and this script was written to publish an error bar around every
reported number.

The suspect was innocent.  The variation came from
``vio.backend.marginalisation`` iterating a set of string keys, which Python
hashes differently in every process.  The column order of the marginal system
followed that iteration order and the prior followed the columns.  With the
ordering fixed, this script reports a spread of zero: five seeds on V1_01 and
five on V1_02 give an ATE identical to every digit printed, standard
deviation 0.0000.  A direct check explains why.  ``cv2.setRNGSeed`` does not
change what ``findFundamentalMat`` or ``findEssentialMat`` return under
OpenCV 5, even on a problem with twenty per cent outliers, because those
routines sample through USAC rather than through the global generator.

So the numbers in the documents are not samples with error bars.  They are
the answer.  A tuning change that moves ATE at all has moved it for a reason.

This script stays because that conclusion is a measurement about one version
of one library rather than a guarantee, worth re-running when either changes.

    python scripts/seed_spread.py V1_01_easy V1_02_medium --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio.evaluation.metrics import ate  # noqa: E402
from vio.geometry import Pose  # noqa: E402
from vio.io import euroc  # noqa: E402
from vio.pipeline.estimator import Estimator, EstimatorConfig  # noqa: E402

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)


#: See ``scripts/ablate.py``: these have to be in the environment before a
#: worker starts, not set from inside it, or every worker opens its own BLAS
#: thread pool and the sweep oversubscribes the machine.
SINGLE_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def one(job) -> dict:
    sequence, seed, data, frames, mode = job
    cv2.setNumThreads(1)
    loaded = euroc.load(Path(data) / sequence, stereo=False)
    cam = loaded.cam0
    n = len(cam) if frames <= 0 else min(frames, len(cam))

    cfg = EstimatorConfig(use_imu=(mode == "vio"), ransac_seed=seed)
    est = Estimator(cam.camera, cam.T_body_cam,
                    loaded.imu if cfg.use_imu else None, loaded.imu_noise, cfg)
    t0 = time.perf_counter()
    for i in range(n):
        est.process(cam.image(i), float(cam.times[i]), i)
    elapsed = time.perf_counter() - t0

    times, poses = est.poses(initialised_only=cfg.use_imu)
    gt = loaded.groundtruth
    gt_poses = [Pose(R, p) for R, p in zip(gt.rotations, gt.positions)]
    alignment = "se3" if cfg.use_imu else "sim3"
    rigid = ate(times, poses, gt.times, gt_poses, mode=alignment)
    similar = ate(times, poses, gt.times, gt_poses, mode="sim3")
    return {
        "sequence": sequence,
        "seed": seed,
        "mode": mode,
        "ate": rigid.translation.rmse,
        "rotation_deg": rigid.rotation_deg.rmse,
        "scale_error_percent": 100.0 * abs(similar.scale - 1.0),
        "poses": int(len(times)),
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequences", nargs="*", default=["V1_01_easy", "V1_02_medium"])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--mode", choices=["vio", "vo"], default="vio")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max((os.cpu_count() or 4) // 2, 1))
    args = ap.parse_args()

    jobs = [
        (s, seed, args.data, args.frames, args.mode)
        for s in args.sequences for seed in range(args.seeds)
    ]
    print(f"{len(jobs)} runs on {args.jobs} workers", flush=True)
    for name in SINGLE_THREAD_ENV:
        os.environ[name] = "1"
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(one, jobs):
            rows.append(row)
            print(f"  {row['sequence']:18s} seed {row['seed']}  "
                  f"ATE {row['ate']:.4f}  scale {row['scale_error_percent']:.2f}%  "
                  f"{row['seconds']:.0f} s", flush=True)

    summary = {}
    print(f"\n{'sequence':18s} {'seeds':>5s} {'ATE mean':>9s} {'sd':>8s} "
          f"{'min':>8s} {'max':>8s} {'spread':>8s}")
    for sequence in args.sequences:
        subset = [r["ate"] for r in rows if r["sequence"] == sequence]
        if not subset:
            continue
        mean, sd = float(np.mean(subset)), float(np.std(subset, ddof=1))
        summary[sequence] = {
            "n": len(subset), "mean": mean, "sd": sd,
            "min": float(np.min(subset)), "max": float(np.max(subset)),
            "spread_percent": 100.0 * (np.max(subset) - np.min(subset)) / mean,
            "scale_error_mean": float(np.mean(
                [r["scale_error_percent"] for r in rows if r["sequence"] == sequence]
            )),
        }
        s = summary[sequence]
        print(f"{sequence:18s} {s['n']:5d} {mean:9.4f} {sd:8.4f} "
              f"{s['min']:8.4f} {s['max']:8.4f} {s['spread_percent']:7.1f}%")

    spreads = [v["spread_percent"] for v in summary.values()]
    if spreads and max(spreads) < 1e-9:
        print()
        print("Spread is exactly zero on every sequence.  The seed changes nothing"
              " in this OpenCV build, so each ATE above is the answer")
        print("rather than one sample of it.  The module docstring says why,"
              " together with what would have to change for that to stop holding.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "seed_spread.json"
    path.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2,
                               default=float), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
