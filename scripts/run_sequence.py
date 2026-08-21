#!/usr/bin/env python
"""Run the estimator over one sequence and write a trajectory plus a result file.

This is the script every number in the README comes from.  It writes:

* ``<out>/<sequence>_<mode>.tum``      the estimated trajectory
* ``<out>/<sequence>_gt.tum``          ground truth on the same clock
* ``<out>/<sequence>_<mode>_map.ply``  the sparse landmark map
* ``<out>/<sequence>_<mode>.json``     metrics, timing, configuration and the
                                       exact commit the run was made from

Nothing here computes a metric that is not reproducible from those files with
``evaluation/evaluate.py`` or with ``evo`` directly.

Usage
-----

    python scripts/run_sequence.py V1_01_easy --mode vio
    python scripts/run_sequence.py --sim --duration 30 --mode vo
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio.evaluation.metrics import ate, drift_rate, rpe  # noqa: E402
from vio.frontend.tracker import TrackerConfig  # noqa: E402
from vio.io import euroc  # noqa: E402
from vio.io.pointcloud import write_ply  # noqa: E402
from vio.io.trajectory import read_tum, write_tum  # noqa: E402
from vio.pipeline.estimator import Estimator, EstimatorConfig  # noqa: E402

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_config(args) -> EstimatorConfig:
    cfg = EstimatorConfig(use_imu=(args.mode == "vio"))
    cfg.window_size = args.window
    cfg.max_landmarks = args.max_landmarks
    cfg.marginalise = not args.no_marginalise
    cfg.tracker = TrackerConfig(
        max_features=args.features,
        min_distance=args.min_distance,
        use_gyro_prediction=args.gyro_prediction,
    )
    return cfg


def load_source(args):
    """Either a real sequence or a generated one, in the same shape."""
    if args.sim:
        from vio.sim.render import SceneRenderer
        from vio.sim.simulator import SimConfig, simulate

        sim_cfg = SimConfig(duration=args.duration, seed=args.seed)
        seq = simulate(sim_cfg)
        renderer = SceneRenderer(seq, noise_sigma=2.0)
        frames = [
            (i, f.stamp, lambda i=i: renderer.render(i)) for i, f in enumerate(seq.frames)
        ]
        gt_times = seq.true_times
        gt_poses = [s.pose for s in seq.states]
        return (
            "sim", sim_cfg.camera, sim_cfg.T_body_cam, seq.imu, sim_cfg.imu_noise,
            frames, gt_times, gt_poses, 0,
        )

    root = Path(args.data) / args.sequence
    seq = euroc.load(root, stereo=False)
    cam = seq.cam0
    n = len(cam) if args.frames <= 0 else min(args.frames, len(cam))
    frames = [
        (i, float(cam.times[i]), lambda i=i: cam.image(i))
        for i in range(args.start, n, args.stride)
    ]
    gt = seq.groundtruth
    gt_poses = [
        __import__("vio.geometry", fromlist=["Pose"]).Pose(R, p)
        for R, p in zip(gt.rotations, gt.positions)
    ]
    return (
        seq.name, cam.camera, cam.T_body_cam, seq.imu, seq.imu_noise,
        frames, gt.times, gt_poses, seq.origin_ns,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="?", default="V1_01_easy")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--mode", choices=["vio", "vo"], default="vio")
    ap.add_argument("--out", default="results")
    ap.add_argument("--frames", type=int, default=0, help="0 means the whole sequence")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--features", type=int, default=200)
    ap.add_argument("--min-distance", type=float, default=25.0)
    ap.add_argument("--max-landmarks", type=int, default=150)
    ap.add_argument("--no-marginalise", action="store_true")
    ap.add_argument("--gyro-prediction", action="store_true")
    ap.add_argument("--sim", action="store_true", help="run on a generated sequence")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="how many runs shared the machine, recorded "
                         "so the timing table can ignore contended runs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    (name, camera, T_bc, imu, noise, frames, gt_times, gt_poses,
     origin_ns) = load_source(args)

    cfg = build_config(args)
    est = Estimator(camera, T_bc, imu if cfg.use_imu else None, noise, cfg)

    t0 = time.perf_counter()
    for k, (index, stamp, get_image) in enumerate(frames):
        est.process(get_image(), stamp, index)
        if not args.quiet and k % 250 == 0:
            print(f"  {k}/{len(frames)} frames, state {est.state}, "
                  f"{len(est.landmarks)} landmarks", flush=True)
    elapsed = time.perf_counter() - t0

    times, poses = est.poses(initialised_only=cfg.use_imu)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_{args.mode}"
    traj_path = out_dir / f"{tag}.tum"
    write_tum(traj_path, times, poses, origin_ns,
              comment=f"{name} {args.mode} commit {git_commit()}")
    write_tum(out_dir / f"{name}_gt.tum", gt_times, gt_poses, origin_ns,
              comment=f"{name} ground truth")

    # The sparse map the run built.  The window holds at most max_landmarks at
    # once; this is every landmark's position as it left the window.
    cloud = est.landmark_cloud(whole_map=True)
    map_path = write_ply(
        out_dir / f"{tag}_map.ply", cloud,
        note=f"{name} {args.mode}, {len(cloud)} landmarks, commit {git_commit()}",
    )

    # Metrics are computed from the files that were just written, not from the
    # arrays in memory.  They are then, by construction, the numbers anyone
    # else gets from the same files with evaluation/evaluate.py or with evo,
    # rather than numbers that happen to agree.
    times, poses = read_tum(traj_path)
    gt_times, gt_poses = read_tum(out_dir / f"{name}_gt.tum")

    # Both alignments are always reported.  A monocular visual-only trajectory
    # can only be scored under sim3 and quoting a visual-inertial number under
    # sim3 would hide the very thing the IMU is there to provide.
    metrics = {}
    for mode in ("se3", "sim3"):
        a = ate(times, poses, gt_times, gt_poses, mode=mode)
        metrics[f"ate_{mode}"] = a.as_dict()
        metrics[f"drift_{mode}"] = drift_rate(times, poses, gt_times, gt_poses, mode=mode)
        scale = a.scale if mode == "sim3" else None
        for delta in (1.0, 5.0, 10.0):
            r = rpe(times, poses, gt_times, gt_poses, delta=delta, unit="m", scale=scale)
            metrics[f"rpe_{mode}_{int(delta)}m"] = r.as_dict()

    record = {
        "sequence": name,
        "mode": args.mode,
        "commit": git_commit(),
        "frames_processed": len(frames),
        "frames_with_pose": int(len(times)),
        "initialised": bool(est.initialised),
        "wall_seconds": round(elapsed, 3),
        "workers": args.workers,
        "timing": est.timing.as_dict(),
        "events": est.events,
        "event_counts": est.event_summary(),
        "config": {
            "window_size": cfg.window_size,
            "max_landmarks": cfg.max_landmarks,
            "marginalise": cfg.marginalise,
            "max_features": cfg.tracker.max_features,
            "min_distance": cfg.tracker.min_distance,
            "use_gyro_prediction": cfg.tracker.use_gyro_prediction,
            "stride": args.stride,
        },
        "metrics": metrics,
        "trajectory_file": str(traj_path),
        "map_file": str(map_path),
        "map_points": int(len(cloud)),
    }
    result_path = out_dir / f"{tag}.json"
    result_path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    a = metrics["ate_se3"]["ate_translation_m"]
    s = metrics["ate_sim3"]
    print(f"\n{name} [{args.mode}] {len(times)} poses, {elapsed:.1f} s "
          f"({est.timing.as_dict()['ms_per_frame']:.0f} ms/frame)")
    print(f"  ATE se3  rmse {a['rmse']:.4f} m   median {a['median']:.4f} m")
    print(f"  ATE sim3 rmse {s['ate_translation_m']['rmse']:.4f} m   "
          f"scale {s['alignment_scale']:.4f}")
    print(f"  map: {len(cloud)} landmarks")
    print(f"  wrote {traj_path}, {map_path} and {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
