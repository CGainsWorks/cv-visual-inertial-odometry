#!/usr/bin/env python
"""Write a synthetic sequence to disk in the ASL layout EuRoC uses.

Two reasons this exists.  It lets the dataset loader be tested against data
whose every field is known, rather than against a real sequence where a
misread column just makes the trajectory quietly worse.  And it gives anyone
without the 19 GB download something the whole pipeline will actually run on.

    python scripts/make_sim_dataset.py --out data/sim_sequence --duration 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio.sim.render import write_asl  # noqa: E402
from vio.sim.simulator import SimConfig, simulate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sim_sequence")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--landmarks", type=int, default=1200)
    ap.add_argument("--texture-points", type=int, default=7000)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    seq = simulate(SimConfig(
        duration=args.duration, n_landmarks=args.landmarks,
        pixel_noise=0.4, seed=args.seed,
    ))
    root = write_asl(
        seq, args.out, write_images=not args.no_images,
        texture_points=args.texture_points,
    )
    print(f"wrote {len(seq.frames)} frames and {len(seq.imu)} IMU samples to "
          f"{root} in {time.perf_counter() - t0:.1f} s")
    print(f"path length {seq.path_length():.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
