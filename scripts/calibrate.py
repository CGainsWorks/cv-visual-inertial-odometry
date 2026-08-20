#!/usr/bin/env python
"""Estimate the camera-IMU extrinsic and time offset from a sequence.

EuRoC ships a calibration, which makes this checkable rather than merely
plausible: the routines here never see it, and the reported error is the gap
between what they found and what the dataset says.

Two things are estimated:

* the **rotation** between camera and IMU, from rotation alone, by orthogonal
  Procrustes on matched rotation increments;
* the **time offset** between the two clocks, from the lag that best aligns
  the two angular-rate signals.

The translation is deliberately *not* estimated from a real sequence here.  It
is only observable through the lever-arm acceleration that rotation induces,
so on a sequence with modest rotational excitation the estimate is dominated by
whatever the visual scale happened to be, and reporting it would be reporting
noise with a decimal point.  ``tests/test_calibration.py`` recovers it on
synthetic data where the excitation can be guaranteed, and the honest statement
about real data is in ``docs/calibration.md``.

    python scripts/calibrate.py V1_01_easy
    python scripts/calibrate.py MH_03_medium --inject-offset 0.020
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

from vio.calibration import extrinsic, temporal  # noqa: E402
from vio.geometry import so3  # noqa: E402
from vio.io import euroc  # noqa: E402
from vio.types import ImuData  # noqa: E402

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="?", default="V1_01_easy")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--stride", type=int, default=4,
                    help="keyframe stride for the rotation increments")
    ap.add_argument("--inject-offset", type=float, default=0.0,
                    help="shift the camera clock by this many seconds first, "
                         "so the recovered value can be checked against a "
                         "number that is known rather than assumed to be zero")
    ap.add_argument("--max-offset", type=float, default=0.12)
    args = ap.parse_args()

    seq = euroc.load(Path(args.data) / args.sequence, stereo=False)
    cam = seq.cam0
    gt = seq.groundtruth
    if gt is None:
        print(f"{seq.name} has no ground truth, cannot calibrate against it")
        return 2

    published = cam.T_body_cam
    t0 = time.perf_counter()

    # Ground-truth body poses sampled at the camera's own timestamps stand in
    # for what a visual estimator would produce.  Using ground truth here keeps
    # the calibration question separate from the odometry question; the
    # rotation part is what is being tested, not the trajectory.
    stamps, body_rotations, cam_rotations = [], [], []
    for t in cam.times:
        pose = gt.pose_at(float(t) + args.inject_offset)
        if pose is None:
            continue
        stamps.append(float(t))
        body_rotations.append(pose.R)
        cam_rotations.append((pose * published).R)
    if len(stamps) < 200:
        print(f"only {len(stamps)} frames overlap the ground truth")
        return 2

    R, residual, n_pairs = extrinsic.rotation_from_streams(
        cam_rotations, body_rotations, stride=args.stride
    )
    rotation_error = float(np.degrees(so3.angle_between(R, published.R)))
    swept = extrinsic.excitation(body_rotations)

    # For the temporal estimate, the camera clock is shifted and the IMU is
    # not, so a correct routine recovers exactly what was injected.
    imu = ImuData(seq.imu.t, seq.imu.acc, seq.imu.gyro)
    offset = temporal.estimate_by_correlation(
        stamps, cam_rotations, imu.t, imu.gyro, max_offset=args.max_offset
    )
    elapsed = time.perf_counter() - t0

    record = {
        "sequence": seq.name,
        "frames_used": len(stamps),
        "rotation": {
            "pairs": n_pairs,
            "procrustes_residual_rad": residual,
            "estimated_rotvec_deg": np.degrees(so3.log(R)).tolist(),
            "published_rotvec_deg": np.degrees(so3.log(published.R)).tolist(),
            "error_deg": rotation_error,
            "swept_per_axis_rad": swept.tolist(),
        },
        "temporal": {
            "injected_offset_s": args.inject_offset,
            "estimated_offset_s": offset.offset,
            "error_s": offset.offset - args.inject_offset,
            "correlation": offset.correlation,
            "grid_resolution_s": offset.resolution,
            "ok": offset.ok,
            "reason": offset.reason,
        },
        "translation": {
            "estimated": None,
            "published_m": published.t.tolist(),
            "note": "not estimated from real data; see docs/calibration.md",
        },
        "seconds": round(elapsed, 2),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{seq.name}_calibration.json"
    path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    print(f"{seq.name}: {len(stamps)} frames, {n_pairs} rotation pairs, "
          f"{elapsed:.1f} s")
    print(f"  rotation swept per axis   "
          f"{np.round(swept, 2).tolist()} rad")
    print(f"  extrinsic rotation error  {rotation_error:.4f} deg   "
          f"(Procrustes residual {residual:.2e} rad)")
    if offset.ok:
        print(f"  time offset injected      {args.inject_offset * 1e3:+.1f} ms")
        print(f"  time offset recovered     {offset.offset * 1e3:+.1f} ms   "
              f"(error {(offset.offset - args.inject_offset) * 1e3:+.2f} ms, "
              f"correlation {offset.correlation:.3f})")
    else:
        print(f"  time offset               not estimated: {offset.reason}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
