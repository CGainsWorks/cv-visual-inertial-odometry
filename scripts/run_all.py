#!/usr/bin/env python
"""Run every sequence in both configurations, in parallel, and summarise.

One process per run.  The estimator is single-threaded and numpy releases the
GIL only for the larger array operations, so processes rather than threads is
the only arrangement that actually uses the cores.

    python scripts/run_all.py --jobs 6
    python scripts/run_all.py --sequences V1_01_easy V1_02_medium --modes vio
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)

ALL_SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
]


#: Each worker is pinned to a single thread.  numpy and OpenCV both spawn
#: their own thread pools, so N workers on an N-core machine end up with N^2
#: runnable threads fighting over the cores; measured on a 16-core machine,
#: eight unpinned workers were slower than six pinned ones by about a factor
#: of three.  The estimator is a long chain of small matrix operations, so it
#: gets almost nothing from intra-process threading anyway.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENCV_FOR_THREADS_NUM": "1",
}


def one(job):
    sequence, mode, data, out, extra = job
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_sequence.py"), sequence,
        "--mode", mode, "--data", str(data), "--out", str(out), "--quiet",
    ] + list(extra)
    env = dict(os.environ)
    env.update(SINGLE_THREAD_ENV)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return {
        "sequence": sequence,
        "mode": mode,
        "returncode": proc.returncode,
        "seconds": round(time.perf_counter() - t0, 1),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="*", default=ALL_SEQUENCES)
    ap.add_argument("--modes", nargs="*", default=["vio", "vo"])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--jobs", type=int, default=max((os.cpu_count() or 4) // 2, 1))
    ap.add_argument("--frames", type=int, default=0)
    args, extra = ap.parse_known_args()
    if args.frames:
        extra += ["--frames", str(args.frames)]

    missing = [s for s in args.sequences if not (Path(args.data) / s).is_dir()]
    if missing:
        print(f"missing sequences under {args.data}: {', '.join(missing)}")
        print("run scripts/fetch_euroc.py first")
        return 2

    jobs = [
        (s, m, args.data, args.out, extra)
        for s in args.sequences for m in args.modes
    ]
    print(f"{len(jobs)} runs on {args.jobs} workers", flush=True)

    t0 = time.perf_counter()
    done = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(one, jobs):
            done.append(result)
            flag = "ok " if result["returncode"] == 0 else "FAIL"
            print(f"  {flag} {result['sequence']:18s} {result['mode']:4s} "
                  f"{result['seconds']:7.1f}s", flush=True)
            if result["returncode"] != 0:
                print(result["stderr"][-1500:], flush=True)

    print(f"\nall runs finished in {time.perf_counter() - t0:.0f} s")
    summary = Path(args.out) / "run_all.json"
    summary.write_text(json.dumps(done, indent=2), encoding="utf-8")
    print(f"wrote {summary}")
    return 0 if all(r["returncode"] == 0 for r in done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
