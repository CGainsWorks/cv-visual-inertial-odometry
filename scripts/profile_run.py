#!/usr/bin/env python
"""Function-level profile of one run, beyond the per-stage timers.

Every result file already carries a per-stage breakdown, because the estimator
times its own tracking, preintegration, triangulation, solve and
marginalisation.  That answers "which stage", not "which function" and the
second question is the one that tells you what to fix.  This runs cProfile over
a bounded slice of a sequence and prints cumulative time by function, plus the
stage timers alongside so the two views can be compared.

    python scripts/profile_run.py V1_01_easy --frames 600
    python scripts/profile_run.py --sim --duration 20
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio.io import euroc  # noqa: E402
from vio.pipeline.estimator import Estimator, EstimatorConfig  # noqa: E402

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="?", default="V1_01_easy")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--mode", choices=["vio", "vo"], default="vio")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()

    if args.sim:
        from vio.sim.render import SceneRenderer
        from vio.sim.simulator import SimConfig, simulate

        cfg = SimConfig(duration=args.duration, seed=21)
        seq = simulate(cfg)
        renderer = SceneRenderer(seq, noise_sigma=2.0)
        name = "sim"
        camera, T_bc, imu, noise = cfg.camera, cfg.T_body_cam, seq.imu, cfg.imu_noise
        # Images are rendered up front so the profile measures the estimator
        # rather than the renderer.
        frames = [(i, f.stamp, renderer.render(i)) for i, f in enumerate(seq.frames)]
    else:
        loaded = euroc.load(Path(args.data) / args.sequence, stereo=False)
        cam = loaded.cam0
        n = min(args.frames, len(cam)) if args.frames > 0 else len(cam)
        name = loaded.name
        camera, T_bc = cam.camera, cam.T_body_cam
        imu, noise = loaded.imu, loaded.imu_noise
        print(f"loading {n} images from {name}", flush=True)
        frames = [(i, float(cam.times[i]), cam.image(i)) for i in range(n)]

    est_cfg = EstimatorConfig(use_imu=(args.mode == "vio"))
    est = Estimator(camera, T_bc, imu if est_cfg.use_imu else None, noise, est_cfg)

    def run():
        for index, stamp, image in frames:
            est.process(image, stamp, index)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    run()
    profiler.disable()
    elapsed = time.perf_counter() - t0

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats.print_stats(args.top)
    text = buffer.getvalue()

    timing = est.timing.as_dict()
    print(f"\n{name} [{args.mode}] {len(frames)} frames in {elapsed:.1f} s "
          f"({1e3 * elapsed / len(frames):.1f} ms/frame under the profiler)")
    print("\nper-stage timers (unprofiled cost is lower than the above):")
    for key, value in timing.items():
        print(f"  {key:34s} {value}")
    print("\n" + text)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{args.mode}_profile.json"
    path.write_text(json.dumps({
        "sequence": name,
        "mode": args.mode,
        "frames": len(frames),
        "profiled_seconds": round(elapsed, 3),
        "timing": timing,
        "cumulative_top": text,
    }, indent=2, default=float), encoding="utf-8")
    stats.dump_stats(str(out / f"{name}_{args.mode}_profile.prof"))
    print(f"wrote {path} and the raw .prof beside it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
