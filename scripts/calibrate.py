#!/usr/bin/env python
"""Estimate the camera-IMU extrinsic rotation and time offset from a sequence.

EuRoC ships a calibration, which makes this checkable rather than merely
plausible: the routines here never see it and the reported error is the gap
between what they found and what the dataset says.

**The inputs are deliberately independent of the answer.**  An earlier version
of this script took ground-truth body poses, multiplied them by the *published*
extrinsic to get camera rotations and then recovered that extrinsic to within
1e-14 radians.  That is not a calibration, it is an identity: the answer had
been used to construct the input.  What runs now is the real procedure:

* camera rotations come from **visual odometry over the images**, with the IMU
  switched off, so nothing about the extrinsic or the IMU enters them;
* body rotations come from **integrating the raw gyroscope** between the same
  instants, so nothing about the camera enters them;
* the published extrinsic is loaded only at the end, to score the result.

The translation is not estimated.  It is only observable through the lever-arm
acceleration that rotation induces and on a sequence with modest rotational
excitation the estimate is dominated by whatever the visual scale happened to
be.  ``tests/test_calibration.py`` recovers it on synthetic data where the
excitation can be guaranteed; ``docs/calibration.md`` says why the real-data
case is left alone.

    python scripts/calibrate.py V1_01_easy
    python scripts/calibrate.py V1_02_medium --inject-offset 0.020
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
from vio.imu.preintegration import preintegrate  # noqa: E402
from vio.io import euroc  # noqa: E402
from vio.pipeline.estimator import Estimator, EstimatorConfig  # noqa: E402
from vio.types import ImuBias  # noqa: E402

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)


def visual_rotations(sequence, frames: int, offset: float, quiet: bool):
    """Camera orientation at each keyframe, from visual odometry alone.

    The estimator is run with ``use_imu`` false, so its rotations come from
    images and nothing else.  Scale is arbitrary and irrelevant here: rotation
    is the one thing a monocular system observes metrically.

    **Keyframes, not frames.**  Between keyframes the visual-only estimator
    holds the last keyframe's pose, because interpolating towards a pose it has
    not computed yet would be inventing information.  Feeding those held poses
    into a correlation against the gyroscope makes the visual signal lag by
    part of a keyframe interval and the lag is read back as a time offset:
    measured on V1_01, that artefact alone accounted for about 7 ms of apparent
    offset on a sequence whose true offset is zero.  Keyframe poses are genuine
    estimates and carry no such lag.
    """
    cam = sequence.cam0
    n = len(cam) if frames <= 0 else min(frames, len(cam))
    est = Estimator(cam.camera, cam.T_body_cam, None, sequence.imu_noise,
                    EstimatorConfig(use_imu=False))
    for i in range(n):
        # The injected offset shifts the camera clock, so the *reported*
        # timestamp of a frame is earlier than the instant it was exposed.
        est.process(cam.image(i), float(cam.times[i]) - offset, i)
        if not quiet and i % 500 == 0:
            print(f"  visual odometry {i}/{n}", flush=True)
    _, stamps, poses = est.keyframe_trajectory()
    rotations = [(T * cam.T_body_cam).R for T in poses]
    return np.asarray(stamps, dtype=float), rotations


def gyro_rotations(imu, stamps, stride: int):
    """Body rotation increments between the given instants, from the gyroscope.

    Preintegration at zero bias.  A few milliradians per second of unmodelled
    bias is well below the extrinsic accuracy this can reach and estimating it
    here would need the very extrinsic being solved for.
    """
    out = []
    for a, b in zip(stamps[:-stride:stride], stamps[stride::stride]):
        out.append(preintegrate(imu, float(a), float(b), bias=ImuBias()).dR)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="?", default="V1_01_easy")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2,
                    help="keyframes per rotation increment")
    ap.add_argument("--inject-offset", type=float, default=0.0,
                    help="shift the camera clock by this many seconds, so the "
                         "recovered value can be checked against a number that "
                         "is known rather than assumed to be zero")
    ap.add_argument("--max-offset", type=float, default=0.12)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sequence = euroc.load(Path(args.data) / args.sequence, stereo=False)
    published = sequence.cam0.T_body_cam
    t0 = time.perf_counter()

    stamps, cam_rotations = visual_rotations(
        sequence, args.frames, args.inject_offset, args.quiet
    )
    if len(stamps) < 200:
        print(f"visual odometry produced only {len(stamps)} poses")
        return 2

    # -- rotation --------------------------------------------------------
    cam_increments = [
        cam_rotations[i].T @ cam_rotations[i + args.stride]
        for i in range(0, len(cam_rotations) - args.stride, args.stride)
    ]
    body_increments = gyro_rotations(sequence.imu, stamps, args.stride)
    n = min(len(cam_increments), len(body_increments))
    R, residual, n_pairs = extrinsic.rotation_from_increments(
        cam_increments[:n], body_increments[:n]
    )
    if R is None:
        print("not enough usable rotation pairs")
        return 2
    rotation_error = float(np.degrees(so3.angle_between(R, published.R)))

    swept = np.zeros(3)
    for dR in body_increments[:n]:
        swept += np.abs(so3.log(dR))

    # -- time offset -----------------------------------------------------
    offset = temporal.estimate_by_correlation(
        stamps, cam_rotations, sequence.imu.t, sequence.imu.gyro,
        max_offset=args.max_offset,
    )
    elapsed = time.perf_counter() - t0

    record = {
        "sequence": sequence.name,
        "method": "visual odometry rotations against raw gyroscope; the "
                  "published extrinsic is used only to score the result",
        "frames_used": int(len(stamps)),
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
    path = out / f"{sequence.name}_calibration.json"
    path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    print(f"\n{sequence.name}: {len(stamps)} visual poses, {n_pairs} rotation "
          f"pairs, {elapsed:.0f} s")
    print(f"  rotation swept per axis   {np.round(swept, 1).tolist()} rad")
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
