"""The same commit and the same seed must give the same trajectory.

This needs its own file because it needs its own *processes*.  The bug it
exists to catch is invisible from inside one: Python randomises string hashing
per process, so a set of variable keys iterates in a different order every
time and if that order reaches the numerics the answer moves.  It did.
``vio.backend.marginalisation`` ordered the columns of its system by iterating
a set of keys, that ordering fixed the eigendecomposition and the
eigendecomposition fixed the prior.  Two runs of the same commit with the same
seed came out at 0.257 m and 0.362 m of absolute trajectory error on EuRoC
V1_01, which is a third of the number and every within-process check passed
happily throughout.

So the test runs the estimator in two subprocesses with *deliberately
different* ``PYTHONHASHSEED`` values and compares the trajectories bit for bit.
Anything that leaks hash order into the arithmetic fails it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

RUNNER = '''
import json, sys
sys.path.insert(0, {src!r})
import cv2
cv2.setNumThreads(1)
import numpy as np
from vio.pipeline.estimator import Estimator, EstimatorConfig
from vio.sim.render import SceneRenderer
from vio.sim.simulator import SimConfig, simulate

cfg = SimConfig(duration=7.0, n_landmarks=1200, pixel_noise=0.4, seed=21)
seq = simulate(cfg)
renderer = SceneRenderer(seq, noise_sigma=2.0)
est = Estimator(cfg.camera, cfg.T_body_cam, seq.imu, cfg.imu_noise,
                EstimatorConfig(use_imu=True, ransac_seed=0))
for i, frame in enumerate(seq.frames):
    est.process(renderer.render(i), frame.stamp, frame.index)
times, poses = est.poses()
positions = np.stack([p.t for p in poses])
print(json.dumps({{
    "n": int(len(times)),
    "checksum": float(np.sum(np.abs(positions))),
    "last": [float(v) for v in positions[-1]],
}}))
'''


def _run(hash_seed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER.format(src=str(ROOT / "src"))],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-3000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_the_trajectory_does_not_depend_on_python_hash_randomisation():
    a = _run("0")
    b = _run("12345")
    assert a["n"] == b["n"], (a["n"], b["n"])
    assert a["checksum"] == b["checksum"], (
        f"two processes disagree: {a['checksum']!r} against {b['checksum']!r}. "
        "Something is iterating a set of string keys and letting the order "
        "reach the arithmetic."
    )
    assert a["last"] == b["last"]
