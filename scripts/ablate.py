#!/usr/bin/env python
"""The ablations behind the claims made in the code and in docs/results.md.

Every design decision in this repository that is defended with a number is
defended with a number produced here.  Each ablation prints a table and writes
a JSON record, so ``scripts/check_docs.py`` can verify that the documents still
quote what the code actually does.

    python scripts/ablate.py                       # all of them
    python scripts/ablate.py --only integration frontend
    python scripts/ablate.py --only backend --data ~/datasets/euroc/sequences
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio.backend import factors as F  # noqa: E402
from vio.backend.graph import FactorGraph, linearise  # noqa: E402
from vio.backend.groups import ReprojectionGroup  # noqa: E402
from vio.backend.variables import (  # noqa: E402
    PointVariable, PoseVariable, Values,
)
from vio.evaluation.metrics import ate  # noqa: E402
from vio.frontend.tracker import Tracker, TrackerConfig  # noqa: E402
from vio.geometry import Pose, se3, so3  # noqa: E402
from vio.imu.preintegration import preintegrate  # noqa: E402
from vio.io import euroc  # noqa: E402
from vio.pipeline.estimator import Estimator, EstimatorConfig  # noqa: E402
from vio.sim.render import SceneRenderer  # noqa: E402
from vio.sim.simulator import SimConfig, simulate  # noqa: E402
from vio.types import ImuBias, ImuNoise  # noqa: E402

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)


# ---------------------------------------------------------------------------
# 1. Integration order: Euler versus midpoint
# ---------------------------------------------------------------------------


def ablate_integration(args) -> dict:
    """How much the second-order integration scheme is worth.

    Against an analytic trajectory, so the reference has no integration error
    of its own to hide behind.
    """
    seq = simulate(SimConfig(
        duration=8.0, imu_noise=ImuNoise(0, 0, 0, 0),
        initial_bias=ImuBias(), bias_walk=False, seed=1,
    ))
    rows = []
    for scheme in ("euler", "midpoint"):
        for span in (0.05, 0.1, 0.25, 0.5, 1.0):
            errs = []
            for t0 in np.arange(0.5, 6.0, 0.25):
                pre = preintegrate(seq.imu, float(t0), float(t0) + span,
                                   scheme=scheme)
                pred = pre.predict(seq.state_at(float(t0)))
                truth = seq.state_at(float(t0) + span)
                errs.append((
                    np.linalg.norm(pred.p - truth.p),
                    np.linalg.norm(pred.v - truth.v),
                    so3.angle_between(pred.R, truth.R),
                ))
            worst = np.max(errs, axis=0)
            rows.append({
                "scheme": scheme, "span_s": span,
                "position_m": float(worst[0]),
                "velocity_ms": float(worst[1]),
                "rotation_rad": float(worst[2]),
            })

    print("\nIntegration scheme, worst error against the analytic trajectory")
    print(f"{'scheme':10s} {'span':>6s} {'position':>12s} {'velocity':>12s} "
          f"{'rotation':>12s}")
    for r in rows:
        print(f"{r['scheme']:10s} {r['span_s']:6.2f} {r['position_m']:12.3e} "
              f"{r['velocity_ms']:12.3e} {r['rotation_rad']:12.3e}")
    ratios = {}
    for span in (0.05, 1.0):
        e = next(r for r in rows if r["scheme"] == "euler" and r["span_s"] == span)
        m = next(r for r in rows if r["scheme"] == "midpoint" and r["span_s"] == span)
        ratios[f"position_ratio_at_{span}s"] = e["position_m"] / m["position_m"]
    print("  midpoint is better by a factor of "
          + ", ".join(f"{v:.0f} at {k.split('_')[-1]}" for k, v in ratios.items()))
    return {"rows": rows, "ratios": ratios}


# ---------------------------------------------------------------------------
# 2. Front end: gyroscope-predicted initial guess
# ---------------------------------------------------------------------------


def ablate_frontend(args) -> dict:
    """What the gyro-predicted KLT guess is worth, against frame rate.

    Needs the dataset: the whole point is that the answer depends on how much
    the camera moves between frames, and that is a property of the recording.
    """
    root = Path(args.data) / args.frontend_sequence
    if not root.is_dir():
        print(f"\nfront-end ablation skipped: {root} not found")
        return {"skipped": str(root)}

    seq = euroc.load(root, stereo=False)
    cam = seq.cam0
    R_bc = cam.T_body_cam.R
    rows = []
    print(f"\nGyroscope-predicted KLT guess, {seq.name}, "
          f"{args.frontend_frames} source frames")
    print(f"{'stride':>6s} {'eff. fps':>9s} {'no gyro':>9s} {'with gyro':>10s} "
          f"{'gain':>7s} {'guess err no':>13s} {'guess err with':>15s}")
    for stride in (1, 2, 3, 5, 8):
        counts, guesses = [], []
        for use_gyro in (False, True):
            tracker = Tracker(cam.camera, TrackerConfig(use_gyro_prediction=use_gyro))
            prev_t = None
            tracked, errors = [], []
            prev_uv = prev_ids = None
            for i in range(0, args.frontend_frames, stride):
                t = float(cam.times[i])
                rotation = None
                if prev_t is not None and use_gyro:
                    pre = preintegrate(seq.imu, prev_t, t, bias=ImuBias(),
                                       noise=seq.imu_noise)
                    rotation = R_bc.T @ pre.dR @ R_bc
                out = tracker.process(cam.image(i), t, i, gyro_rotation=rotation)
                if prev_t is not None:
                    tracked.append(out.n_tracked)
                    common = np.intersect1d(prev_ids, out.ids)
                    if len(common) > 20:
                        ia = {int(k): j for j, k in enumerate(prev_ids)}
                        ib = {int(k): j for j, k in enumerate(out.ids)}
                        a = np.stack([prev_uv[ia[int(c)]] for c in common])
                        b = np.stack([out.uv[ib[int(c)]] for c in common])
                        if use_gyro and rotation is not None:
                            xn = cam.camera.undistort(a)
                            bearings = np.concatenate(
                                [xn, np.ones((len(xn), 1))], axis=1
                            ) @ rotation
                            z = bearings[:, 2]
                            ok = z > 1e-6
                            guess = a.copy()
                            guess[ok] = cam.camera.project(
                                bearings[ok, :2] / z[ok, None]
                            )
                        else:
                            guess = a
                        errors.append(float(np.median(
                            np.linalg.norm(guess - b, axis=1)
                        )))
                prev_t, prev_uv, prev_ids = t, out.uv, out.ids
            counts.append(float(np.mean(tracked)))
            guesses.append(float(np.mean(errors)) if errors else float("nan"))
        gain = 100.0 * (counts[1] - counts[0]) / max(counts[0], 1e-9)
        rows.append({
            "stride": stride, "fps": 20.0 / stride,
            "tracked_no_gyro": counts[0], "tracked_with_gyro": counts[1],
            "gain_percent": gain,
            "guess_error_no_gyro_px": guesses[0],
            "guess_error_with_gyro_px": guesses[1],
        })
        print(f"{stride:6d} {20.0 / stride:9.1f} {counts[0]:9.1f} {counts[1]:10.1f} "
              f"{gain:+6.1f}% {guesses[0]:13.2f} {guesses[1]:15.2f}")
    return {"sequence": seq.name, "rows": rows}


# ---------------------------------------------------------------------------
# 3. Back end: vectorised groups, and marginalisation
# ---------------------------------------------------------------------------


def ablate_backend(args) -> dict:
    """Two claims: the group is faster, and marginalisation matters."""
    out = {}

    #, speed of the vectorised group -----------------------------------
    rng = np.random.default_rng(4)
    camera = simulate(SimConfig(duration=0.2, seed=0)).config.camera
    T_bc = Pose(se3.exp(np.array([0.1, -0.2, 1.5, 0.0, 0.0, 0.0])).R,
                np.array([0.02, -0.06, 0.01]))
    values = Values()
    for i in range(10):
        values[f"x{i}"] = PoseVariable(se3.exp(rng.normal(0, 0.15, 6)))
    group = ReprojectionGroup(camera, T_bc, 1.0, F.HuberLoss(2.0))
    for j in range(150):
        values[f"l{j}"] = PointVariable(
            rng.normal(0, 1.5, 3) + np.array([0.0, 0.0, 6.0])
        )
        for i in range(10):
            T = values[f"x{i}"].value * T_bc
            pc = T.R.T @ (values[f"l{j}"].value - T.t)
            if pc[2] > 0.5:
                group.add(f"x{i}", f"l{j}", pc[:2] / pc[2] + rng.normal(0, 0.002, 2))

    timings = {}
    for label, graph in (("group", FactorGraph([group])),
                         ("scalar", FactorGraph(group.to_factors()))):
        t0 = time.perf_counter()
        for _ in range(20):
            linearise(graph, values)
        timings[label] = (time.perf_counter() - t0) / 20 * 1e3
    a = linearise(FactorGraph([group]), values)
    b = linearise(FactorGraph(group.to_factors()), values)
    out["group"] = {
        "observations": len(group),
        "ms_group": timings["group"],
        "ms_scalar": timings["scalar"],
        "speedup": timings["scalar"] / timings["group"],
        "max_jacobian_difference": float(
            np.abs(a.J.toarray() - b.J.toarray()).max()
        ),
        "cost_difference": abs(a.cost - b.cost),
    }
    print(f"\nVectorised reprojection group, {len(group)} observations")
    print(f"  scalar factors  {timings['scalar']:7.2f} ms per linearisation")
    print(f"  vectorised      {timings['group']:7.2f} ms  "
          f"({out['group']['speedup']:.1f}x faster)")
    print(f"  they agree to   {out['group']['max_jacobian_difference']:.2e} "
          f"in the Jacobian")

    #, marginalisation -------------------------------------------------
    seq = simulate(SimConfig(duration=args.duration, n_landmarks=1200,
                             pixel_noise=0.4, seed=21))
    renderer = SceneRenderer(seq, noise_sigma=2.0)
    images = [renderer.render(i) for i in range(len(seq.frames))]
    gt_times = seq.true_times
    gt_poses = [s.pose for s in seq.states]

    rows = []
    print("\nMarginalisation, on a generated sequence")
    print(f"{'marginalise':12s} {'ATE rmse':>10s} {'rot rmse':>10s} "
          f"{'scale err':>10s} {'ms/keyframe':>12s}")
    for marginalise in (True, False):
        cfg = EstimatorConfig(use_imu=True, marginalise=marginalise)
        est = Estimator(seq.config.camera, seq.config.T_body_cam, seq.imu,
                        seq.config.imu_noise, cfg)
        for image, frame in zip(images, seq.frames):
            est.process(image, frame.stamp, frame.index)
        times, poses = est.poses(initialised_only=True)
        rigid = ate(times, poses, gt_times, gt_poses, mode="se3")
        similar = ate(times, poses, gt_times, gt_poses, mode="sim3")
        timing = est.timing.as_dict()
        rows.append({
            "marginalise": marginalise,
            "ate_rmse_m": rigid.translation.rmse,
            "rotation_rmse_deg": rigid.rotation_deg.rmse,
            "scale_error_percent": 100.0 * abs(similar.scale - 1.0),
            "ms_per_keyframe": timing["ms_optimise_per_keyframe"]
            + timing["ms_marginalise_per_keyframe"],
        })
        print(f"{str(marginalise):12s} {rows[-1]['ate_rmse_m']:10.4f} "
              f"{rows[-1]['rotation_rmse_deg']:10.3f} "
              f"{rows[-1]['scale_error_percent']:9.2f}% "
              f"{rows[-1]['ms_per_keyframe']:12.1f}")
    out["marginalisation"] = rows
    return out


# ---------------------------------------------------------------------------
# 4. Window size
# ---------------------------------------------------------------------------


def ablate_window(args) -> dict:
    """Accuracy and cost against the number of keyframes held in the window."""
    seq = simulate(SimConfig(duration=args.duration, n_landmarks=1200,
                             pixel_noise=0.4, seed=21))
    renderer = SceneRenderer(seq, noise_sigma=2.0)
    images = [renderer.render(i) for i in range(len(seq.frames))]
    gt_times = seq.true_times
    gt_poses = [s.pose for s in seq.states]

    rows = []
    print("\nSliding-window size, on a generated sequence")
    print(f"{'window':>7s} {'ATE rmse':>10s} {'scale err':>10s} {'ms/frame':>10s}")
    for window in (5, 8, 10, 15):
        cfg = EstimatorConfig(use_imu=True, window_size=window)
        est = Estimator(seq.config.camera, seq.config.T_body_cam, seq.imu,
                        seq.config.imu_noise, cfg)
        for image, frame in zip(images, seq.frames):
            est.process(image, frame.stamp, frame.index)
        times, poses = est.poses(initialised_only=True)
        rigid = ate(times, poses, gt_times, gt_poses, mode="se3")
        similar = ate(times, poses, gt_times, gt_poses, mode="sim3")
        rows.append({
            "window": window,
            "ate_rmse_m": rigid.translation.rmse,
            "scale_error_percent": 100.0 * abs(similar.scale - 1.0),
            "ms_per_frame": est.timing.as_dict()["ms_per_frame"],
        })
        print(f"{window:7d} {rows[-1]['ate_rmse_m']:10.4f} "
              f"{rows[-1]['scale_error_percent']:9.2f}% "
              f"{rows[-1]['ms_per_frame']:10.1f}")
    return {"rows": rows}


ABLATIONS = {
    "integration": ablate_integration,
    "frontend": ablate_frontend,
    "backend": ablate_backend,
    "window": ablate_window,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(ABLATIONS),
                    default=sorted(ABLATIONS))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--frontend-sequence", default="V1_02_medium")
    ap.add_argument("--frontend-frames", type=int, default=600)
    args = ap.parse_args()

    results = {}
    for name in args.only:
        t0 = time.perf_counter()
        results[name] = ABLATIONS[name](args)
        results[name]["seconds"] = round(time.perf_counter() - t0, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ablations.json"
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
