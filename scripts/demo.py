#!/usr/bin/env python
"""The smoke test: both configurations on a generated sequence, no dataset needed.

This is what ``make demo`` runs and what CI runs.  It generates a synthetic
visual-inertial sequence, renders it, runs monocular visual odometry and then
visual-inertial odometry over the same images and prints the comparison that
the whole project exists to make:

* visual-only has no scale, so it can only be scored after a similarity
  alignment that is *given* the true scale;
* visual-inertial claims a metric trajectory, so it is scored rigidly and its
  scale error is reported as a number rather than absorbed by the alignment.

Runs in about ninety seconds on one core.  The numbers it prints come from
synthetic images and are not the numbers in the README, which come from EuRoC.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio.evaluation.metrics import ate, drift_rate, rpe  # noqa: E402
from vio.pipeline.estimator import Estimator, EstimatorConfig  # noqa: E402
from vio.sim.render import SceneRenderer  # noqa: E402
from vio.sim.simulator import SimConfig, simulate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = SimConfig(duration=args.duration, n_landmarks=1200,
                    pixel_noise=0.4, seed=args.seed)
    t0 = time.perf_counter()
    seq = simulate(cfg)
    renderer = SceneRenderer(seq, noise_sigma=2.0)
    images = [renderer.render(i) for i in range(len(seq.frames))]
    print(f"generated {len(images)} frames over {seq.path_length():.1f} m "
          f"in {time.perf_counter() - t0:.1f} s")

    gt_times = seq.true_times
    gt_poses = [s.pose for s in seq.states]
    summary = {}

    for use_imu in (False, True):
        label = "visual-inertial" if use_imu else "visual only    "
        est = Estimator(cfg.camera, cfg.T_body_cam,
                        seq.imu if use_imu else None, cfg.imu_noise,
                        EstimatorConfig(use_imu=use_imu))
        t0 = time.perf_counter()
        for image, frame in zip(images, seq.frames):
            est.process(image, frame.stamp, frame.index)
        elapsed = time.perf_counter() - t0

        times, poses = est.poses(initialised_only=use_imu)
        if len(times) < 10:
            print(f"{label}  FAILED: only {len(times)} poses")
            return 1

        rigid = ate(times, poses, gt_times, gt_poses, mode="se3")
        similar = ate(times, poses, gt_times, gt_poses, mode="sim3")
        relative = rpe(times, poses, gt_times, gt_poses, delta=1.0, unit="m",
                       scale=None if use_imu else similar.scale)
        drift = drift_rate(times, poses, gt_times, gt_poses,
                           mode="se3" if use_imu else "sim3")

        print(f"\n{label}  {len(times)} poses, {est.timing.keyframes} keyframes, "
              f"{elapsed:.1f} s ({1e3 * elapsed / len(images):.0f} ms/frame)")
        if use_imu:
            print(f"  ATE, rigid alignment      {rigid.translation.rmse:8.4f} m")
            print(f"  scale error               "
                  f"{100 * abs(similar.scale - 1.0):8.2f} %"
                  "   <- what the IMU bought")
        else:
            print(f"  ATE, rigid alignment      {rigid.translation.rmse:8.4f} m"
                  "   <- meaningless, no scale")
            print(f"  ATE, similarity alignment {similar.translation.rmse:8.4f} m"
                  "   <- only after being given the scale")
            print(f"  scale it was off by       "
                  f"{100 * abs(similar.scale - 1.0):8.1f} %")
        print(f"  rotation error            {rigid.rotation_deg.rmse:8.3f} deg")
        print(f"  RPE over 1 m              {relative.translation.rmse:8.4f} m")
        print(f"  drift, final              "
              f"{100 * drift['final_per_metre']:8.3f} %/m")

        summary["vio" if use_imu else "vo"] = {
            "poses": int(len(times)),
            "keyframes": est.timing.keyframes,
            "ms_per_frame": round(1e3 * elapsed / len(images), 2),
            "ate_se3_rmse_m": rigid.translation.rmse,
            "ate_sim3_rmse_m": similar.translation.rmse,
            "scale": similar.scale,
            "rotation_rmse_deg": rigid.rotation_deg.rmse,
            "rpe_1m_rmse_m": relative.translation.rmse,
            "drift_final_per_metre": drift["final_per_metre"],
            "timing": est.timing.as_dict(),
        }

    print("\nThe visual-only trajectory is not wrong so much as unitless: its "
          "shape is good\nand its size is arbitrary.  Everything the inertial "
          "measurements add is in that\nscale figure.")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
