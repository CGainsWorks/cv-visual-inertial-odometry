"""Reading and writing trajectories in TUM format.

TUM format is one line per pose::

    timestamp tx ty tz qx qy qz qw

Chosen over the EuRoC CSV layout for the *outputs* because it is what ``evo``
and every other trajectory tool reads without a converter, and because a
results file that a reader can drop straight into their own evaluation is worth
more than one in the dataset's own dialect.

Timestamps are written in seconds with nanosecond resolution.  Sequences whose
clock is a Unix epoch in nanoseconds, which is every EuRoC sequence, would
lose about 250 ns of resolution if written as a bare float64, so an offset can
be supplied and is recorded in the file header.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry import Pose


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to ``[qx, qy, qz, qw]``, Shepperd's branching."""
    R = np.asarray(R, dtype=float)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0)) * 2.0
        if s < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        w = (R[k, j] - R[j, k]) / s
        v = np.zeros(3)
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
    q = np.array([v[0], v[1], v[2], w])
    return q / np.linalg.norm(q)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """``[qx, qy, qz, qw]`` to a rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def write_tum(path, times, poses, offset_ns: int = 0, comment: str = "") -> Path:
    """Write a trajectory.  ``poses`` are body-to-world transforms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if comment:
            for line in comment.splitlines():
                fh.write(f"# {line}\n")
        if offset_ns:
            fh.write(f"# timestamp_offset_ns {offset_ns}\n")
        fh.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, T in zip(times, poses):
            q = _matrix_to_quaternion(T.R)
            fh.write(
                f"{float(t):.9f} {T.t[0]:.9f} {T.t[1]:.9f} {T.t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )
    return path


def read_tum(path):
    """Read a trajectory.  Returns ``(times, poses)``."""
    times, poses = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            vals = [float(v) for v in parts[:8]]
            times.append(vals[0])
            poses.append(Pose(_quaternion_to_matrix(np.array(vals[4:8])),
                              np.array(vals[1:4])))
    return np.asarray(times, dtype=float), poses


def write_groundtruth_tum(path, groundtruth, comment: str = "") -> Path:
    """Convert a loaded :class:`vio.io.euroc.GroundTruth` to a TUM file."""
    poses = [Pose(R, p) for R, p in zip(groundtruth.rotations, groundtruth.positions)]
    return write_tum(path, groundtruth.times, poses, comment=comment)
