#!/usr/bin/env python
"""Run the estimator with loop closure, then score both the detector and the effect.

Two questions, answered separately, because they are separate:

**Was the detector right?**  Every emitted closure is labelled against ground
truth: two keyframes are genuinely the same place if their true positions are
within ``--radius`` metres and their true viewing directions within
``--bearing`` degrees.  Precision is what fraction of emitted closures were
genuine.

Recall is over **revisit episodes**, not keyframe pairs, and the distinction is
not pedantry.  On V1_02 there are 1356 keyframes in a room a few metres across,
so 1347 of them have some earlier keyframe within 1.5 m; counted that way a
detector that closes the loop cleanly once per pass through the room scores
0.005 and looks broken.  An episode is a contiguous stretch of keyframes
revisiting the same place, and it counts as detected if *any* keyframe in it
produced a verified closure, which is what the pose graph needs: a second
closure over the same stretch adds almost nothing.

**Did closing the loops help?**  ATE before and after the pose graph, from the
same run.  This is the number the brief asks for and it is the one that can
disappoint: a detector with perfect precision still buys nothing if the
trajectory had not drifted enough for the correction to matter.

    python scripts/run_loop_closure.py V1_02_medium
    python scripts/run_loop_closure.py --sim --duration 60
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

from vio.evaluation.metrics import ate, associate, drift_rate  # noqa: E402
from vio.geometry import Pose  # noqa: E402
from vio.io import euroc  # noqa: E402
from vio.io.trajectory import write_tum  # noqa: E402
from vio.loop_closure import pose_graph  # noqa: E402
from vio.loop_closure.detector import LoopConfig, LoopDetector  # noqa: E402
from vio.loop_closure.vocabulary import Vocabulary  # noqa: E402
from vio.pipeline.estimator import Estimator, EstimatorConfig  # noqa: E402

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)


def revisit_episodes(pairs, gap: int = 5):
    """Group revisit pairs into episodes: contiguous stretches of the same place.

    Two pairs belong to the same episode when both their query and their match
    keyframes are close together, so a single pass back through a room counts
    once rather than a hundred times.
    """
    episodes = []
    for query, match in sorted(pairs):
        placed = False
        for episode in episodes:
            near_query = query - episode["query_end"] <= gap
            near_match = abs(match - episode["match_end"]) <= gap * 3
            if near_query and near_match:
                episode["queries"].add(query)
                episode["query_end"] = max(episode["query_end"], query)
                episode["match_end"] = match
                placed = True
                break
        if not placed:
            episodes.append({
                "queries": {query}, "query_end": query, "match_end": match,
            })
    return episodes


def revisit_pairs(ids, stamps, gt_times, gt_poses, radius, bearing_deg, min_gap):
    """Keyframe pairs that ground truth says are the same place.

    A place is not just a position: standing in the same spot facing the other
    way is not a revisit any appearance-based system could or should detect, so
    the viewing direction has to agree too.
    """
    ei, ri = associate(stamps, gt_times, max_difference=0.05)
    if len(ei) < 3:
        return [], {}
    poses = {ids[int(i)]: gt_poses[int(j)] for i, j in zip(ei, ri)}
    keys = sorted(poses)
    cos_limit = np.cos(np.radians(bearing_deg))
    pairs = []
    for a_i, a in enumerate(keys):
        for b in keys[a_i + 1:]:
            if b - a < min_gap:
                continue
            Ta, Tb = poses[a], poses[b]
            if np.linalg.norm(Ta.t - Tb.t) > radius:
                continue
            # Body x axis stands in for the viewing direction; the camera is
            # rigidly attached, so any fixed body axis works as a proxy.
            if float((Ta.R @ np.array([1.0, 0, 0])) @ (Tb.R @ np.array([1.0, 0, 0]))) < cos_limit:
                continue
            pairs.append((a, b))
    return pairs, poses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="?", default="V1_02_medium")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--vocabulary", default=str(ROOT / "models" / "vocabulary.pkl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--mode", choices=["vio", "vo"], default="vio")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--radius", type=float, default=1.5, help="metres")
    ap.add_argument("--bearing", type=float, default=45.0, help="degrees")
    ap.add_argument("--min-gap", type=int, default=30, help="keyframes")
    ap.add_argument("--no-gravity-lock", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    vocab_path = Path(args.vocabulary)
    if not vocab_path.exists():
        print(f"no vocabulary at {vocab_path}; run scripts/build_vocabulary.py")
        return 2
    vocab = Vocabulary.load(vocab_path)

    if args.sim:
        from vio.sim.render import SceneRenderer
        from vio.sim.simulator import SimConfig, simulate
        from vio.sim.trajectory import Sinusoid, Trajectory

        # A closed loop: commensurate periods, so the path repeats exactly.
        looping = Trajectory(
            position=Sinusoid([3.0, 2.4, 0.4],
                              [2 * np.pi / 20.0, 2 * np.pi / 20.0, 2 * np.pi / 10.0],
                              [0.0, np.pi / 2, 0.0], [0.0, 0.0, 1.4]),
            rotation=Sinusoid([0.1, 0.08, 0.3],
                              [2 * np.pi / 20.0, 2 * np.pi / 20.0, 2 * np.pi / 20.0],
                              [0.4, 2.1, 0.0], [0.0, 0.0, 0.0]),
        )
        cfg = SimConfig(duration=args.duration, trajectory=looping, seed=71)
        seq = simulate(cfg)
        renderer = SceneRenderer(seq, noise_sigma=2.0)
        name = "sim_loop"
        camera, T_bc, imu, noise = cfg.camera, cfg.T_body_cam, seq.imu, cfg.imu_noise
        frames = [(i, f.stamp, lambda i=i: renderer.render(i))
                  for i, f in enumerate(seq.frames)]
        gt_times = seq.true_times
        gt_poses = [s.pose for s in seq.states]
        origin_ns = 0
    else:
        loaded = euroc.load(Path(args.data) / args.sequence, stereo=False)
        cam = loaded.cam0
        n = len(cam) if args.frames <= 0 else min(args.frames, len(cam))
        name = loaded.name
        camera, T_bc, imu, noise = cam.camera, cam.T_body_cam, loaded.imu, loaded.imu_noise
        frames = [(i, float(cam.times[i]), lambda i=i: cam.image(i)) for i in range(n)]
        gt = loaded.groundtruth
        gt_times = gt.times
        gt_poses = [Pose(R, p) for R, p in zip(gt.rotations, gt.positions)]
        origin_ns = loaded.origin_ns

    est_cfg = EstimatorConfig(use_imu=(args.mode == "vio"))
    est = Estimator(camera, T_bc, imu if est_cfg.use_imu else None, noise, est_cfg)
    est.attach_loop_detector(
        LoopDetector(vocab, camera, T_bc, LoopConfig(min_gap=args.min_gap))
    )

    t0 = time.perf_counter()
    for k, (index, stamp, get_image) in enumerate(frames):
        est.process(get_image(), stamp, index)
        if not args.quiet and k % 250 == 0:
            print(f"  {k}/{len(frames)} frames, {len(est.closures)} closures",
                  flush=True)
    elapsed = time.perf_counter() - t0

    ids, stamps, poses = est.keyframe_trajectory()
    pairs, gt_by_kf = revisit_pairs(
        ids, stamps, gt_times, gt_poses, args.radius, args.bearing, args.min_gap
    )
    episodes = revisit_episodes(pairs)
    opportunities = {a for a, _ in pairs} | {b for _, b in pairs}

    correct, wrong = [], []
    for c in est.closures:
        a, b = gt_by_kf.get(c.match_id), gt_by_kf.get(c.query_id)
        if a is None or b is None:
            wrong.append((c, float("nan")))
            continue
        distance = float(np.linalg.norm(a.t - b.t))
        (correct if distance <= args.radius else wrong).append((c, distance))

    precision = len(correct) / max(len(est.closures), 1)
    detected = {c.query_id for c, _ in correct}
    hit = sum(1 for e in episodes if e["queries"] & detected)
    recall = hit / max(len(episodes), 1)
    # Kept alongside because it is the number a reader might expect, and
    # because the gap between the two is the point.
    keyframe_recall = len(detected & opportunities) / max(len(opportunities), 1)

    # -- the effect ------------------------------------------------------
    before_times, before_poses = est.poses(initialised_only=est_cfg.use_imu)
    mode = "se3" if est_cfg.use_imu else "sim3"
    before = ate(before_times, before_poses, gt_times, gt_poses, mode=mode)
    drift_before = drift_rate(before_times, before_poses, gt_times, gt_poses, mode=mode)

    result = None
    after = after_drift = None
    corrected_poses = before_poses
    if est.closures and len(ids) > 3:
        cfg_pg = pose_graph.PoseGraphConfig(lock_gravity=not args.no_gravity_lock)
        original = dict(zip(ids, poses))
        result = pose_graph.optimise_graph(ids, original, est.closures, cfg_pg)
        corrected_poses = pose_graph.apply_correction(
            before_times, before_poses, stamps, ids, original, result.poses
        )
        after = ate(before_times, corrected_poses, gt_times, gt_poses, mode=mode)
        after_drift = drift_rate(
            before_times, corrected_poses, gt_times, gt_poses, mode=mode
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tum(out_dir / f"{name}_{args.mode}_loop.tum", before_times, corrected_poses,
              origin_ns, comment=f"{name} {args.mode} with loop closure")

    record = {
        "sequence": name,
        "mode": args.mode,
        "vocabulary": {"path": str(vocab_path), "words": vocab.n_words,
                       "trained_on": list(vocab.trained_on)},
        "wall_seconds": round(elapsed, 1),
        "keyframes": len(ids),
        "revisit_opportunities": len(opportunities),
        "revisit_episodes": len(episodes),
        "episodes_detected": hit,
        "closures_emitted": len(est.closures),
        "closures_correct": len(correct),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "keyframe_recall": round(keyframe_recall, 4),
        "closures": [
            {"query": c.query_id, "match": c.match_id, "score": round(c.score, 4),
             "inliers": c.n_inliers, "matches": c.n_matches,
             "correction_m": round(c.correction, 3),
             "true_distance_m": round(d, 3) if np.isfinite(d) else None}
            for c, d in correct + wrong
        ],
        "ate_before": before.as_dict(),
        "ate_after": None if after is None else after.as_dict(),
        "drift_before": drift_before,
        "drift_after": after_drift,
        "pose_graph": None if result is None else {
            "iterations": result.iterations,
            "seconds": round(result.seconds, 3),
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
            "max_displacement_m": round(result.max_displacement(), 4),
            "gravity_locked": not args.no_gravity_lock,
        },
    }
    path = out_dir / f"{name}_{args.mode}_loop.json"
    path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    print(f"\n{name} [{args.mode}] {len(ids)} keyframes, {elapsed:.0f} s")
    print(f"  revisit episodes {len(episodes)} ({len(opportunities)} keyframes "
          f"have a partner), closures {len(est.closures)} "
          f"({len(correct)} correct)")
    print(f"  precision {precision:.3f}   episode recall {recall:.3f} "
          f"({hit}/{len(episodes)})   keyframe recall {keyframe_recall:.3f}")
    print(f"  ATE before {before.translation.rmse:.4f} m")
    if after is not None:
        change = 100.0 * (after.translation.rmse - before.translation.rmse) \
            / max(before.translation.rmse, 1e-12)
        print(f"  ATE after  {after.translation.rmse:.4f} m  ({change:+.1f}%)")
        print(f"  drift {100*drift_before['final_per_metre']:.3f} -> "
              f"{100*after_drift['final_per_metre']:.3f} %/m")
    else:
        print("  no closures, pose graph not run")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
