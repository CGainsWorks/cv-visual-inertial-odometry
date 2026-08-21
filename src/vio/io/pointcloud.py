"""Writing the sparse landmark map, in a format anything can open.

ASCII PLY.  It is the least interesting file format in this repository and the
most widely readable one: MeshLab, CloudCompare, Open3D, Blender and numpy all
take it without a converter and a person can read it in a text editor to check
that the numbers are metres rather than something else.

The map is a by-product of the estimator rather than a maintained structure.
Points are written as they were when they left the sliding window and are never
revisited, so a loop closure corrects the trajectory and not the cloud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HEADER = """ply
format ascii 1.0
comment sparse landmark map from cv-visual-inertial-odometry
comment {note}
element vertex {count}
property float x
property float y
property float z
end_header
"""


def write_ply(path, points, note: str = "") -> Path:
    """Write an ``(N, 3)`` array of world-frame points."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.size == 0:
        points = points.reshape(0, 3)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(HEADER.format(count=len(points), note=note or "no note"))
        for x, y, z in points:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    return path


def read_ply(path):
    """Read back an ASCII PLY written by :func:`write_ply`."""
    rows = []
    started = False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not started:
                started = line.strip() == "end_header"
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(v) for v in parts[:3]])
    return np.asarray(rows, dtype=float).reshape(-1, 3)
