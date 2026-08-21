"""Loader for the ASL directory layout used by EuRoC MAV and TUM VI.

Both datasets ship the same tree, ``mav0/<sensor>/data.csv`` plus a
``sensor.yaml`` per sensor, so one loader covers both.  The differences are
that TUM VI uses an equidistant distortion model where EuRoC uses
radial-tangential and that TUM VI's ground truth is a raw motion-capture
stream rather than a batch-smoothed state estimate.  Both are handled.

Three details in this format catch people out and all three are handled here
rather than left to the caller:

* **Timestamps are integer nanoseconds.**  Converting them to float seconds
  from the Unix epoch loses resolution, 1.4e18 ns needs 61 bits and a float64
  has 53 of mantissa, so you land on roughly 256 ns quantisation.  That is
  small but it is a *systematic* jitter on every IMU interval.  Times are
  therefore rebased to the first camera timestamp before being converted, which
  puts them in the range where float64 is exact to well under a nanosecond.

* **``T_BS`` maps sensor coordinates into the body frame**, so it is
  ``T_body_sensor`` and not its inverse.  For EuRoC's ``imu0`` it is the
  identity, which means body frame and IMU frame coincide and the ground-truth
  poses are directly comparable to the estimator's output.

* **Ground truth does not start when the images do.**  On EuRoC it begins about
  a second late and ends early, so evaluation has to intersect the two time
  ranges rather than assume they match.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..geometry import Pose
from ..geometry.camera import PinholeCamera
from ..types import ImuBias, ImuData, ImuNoise, NavState

NS = 1e-9

_DISTORTION_ALIASES = {
    "radial-tangential": "radtan",
    "radtan": "radtan",
    "plumb_bob": "radtan",
    "equidistant": "equidistant",
    "kannala-brandt": "equidistant",
    "none": "none",
}


@dataclass
class CameraStream:
    """One camera: its calibration, its frame times and where the files are."""

    name: str
    camera: PinholeCamera
    T_body_cam: Pose
    #: Seconds since the sequence origin, ascending.
    times: np.ndarray
    #: Raw nanosecond stamps, kept so results can be written back in the
    #: dataset's own units without a lossy round trip.
    stamps_ns: np.ndarray
    paths: list

    def __len__(self) -> int:
        return len(self.times)

    def image(self, index: int) -> np.ndarray:
        """Read one frame as an 8-bit greyscale array."""
        import cv2

        img = cv2.imread(str(self.paths[index]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.paths[index])
        return img


@dataclass
class GroundTruth:
    """A reference trajectory, with whatever extras the dataset provides."""

    times: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    velocities: np.ndarray | None = None
    bias_acc: np.ndarray | None = None
    bias_gyro: np.ndarray | None = None
    source: str = "unknown"

    def __len__(self) -> int:
        return len(self.times)

    def pose_at(self, t: float) -> Pose | None:
        """Nearest-sample pose, or ``None`` outside the covered interval.

        Nearest rather than interpolated: ground truth is at 200 Hz, so the
        worst nearest-neighbour error is 2.5 ms of motion and interpolating
        rotations invites a convention bug in a function whose whole job is to
        be beyond suspicion.
        """
        if len(self.times) == 0 or t < self.times[0] or t > self.times[-1]:
            return None
        i = int(np.argmin(np.abs(self.times - t)))
        return Pose(self.rotations[i], self.positions[i])

    def state_at(self, t: float) -> NavState | None:
        i = int(np.argmin(np.abs(self.times - t)))
        if len(self.times) == 0 or t < self.times[0] or t > self.times[-1]:
            return None
        v = self.velocities[i] if self.velocities is not None else np.zeros(3)
        return NavState(self.rotations[i], self.positions[i], v)

    def bias_at(self, t: float) -> ImuBias | None:
        if self.bias_acc is None or len(self.times) == 0:
            return None
        i = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        return ImuBias(self.bias_acc[i], self.bias_gyro[i])


@dataclass
class Sequence:
    """A whole visual-inertial sequence."""

    name: str
    root: Path
    cameras: list = field(default_factory=list)
    imu: ImuData = None
    imu_noise: ImuNoise = field(default_factory=ImuNoise)
    T_body_imu: Pose = field(default_factory=Pose.identity)
    groundtruth: GroundTruth | None = None
    #: Nanosecond stamp that ``t = 0`` corresponds to.
    origin_ns: int = 0

    @property
    def cam0(self) -> CameraStream:
        return self.cameras[0]

    @property
    def stereo(self) -> bool:
        return len(self.cameras) > 1

    def duration(self) -> float:
        return float(self.cam0.times[-1] - self.cam0.times[0])

    def groundtruth_length(self) -> float:
        """Distance travelled according to ground truth, for drift per metre."""
        if self.groundtruth is None:
            return float("nan")
        p = self.groundtruth.positions
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _read_T_BS(node: dict) -> Pose:
    data = np.asarray(node["T_BS"]["data"], dtype=float).reshape(4, 4)
    return Pose.from_matrix(data)


def _read_csv(path: Path) -> tuple:
    """Read an ASL ``data.csv`` into an integer stamp column and a float block."""
    stamps = []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            stamps.append(int(row[0]))
            rows.append(row[1:])
    return np.asarray(stamps, dtype=np.int64), rows


def _load_camera(folder: Path, origin_ns: int) -> CameraStream:
    cfg = _read_yaml(folder / "sensor.yaml")
    fu, fv, cu, cv = [float(v) for v in cfg["intrinsics"]]
    w, h = [int(v) for v in cfg["resolution"]]
    model = _DISTORTION_ALIASES.get(
        str(cfg.get("distortion_model", "radial-tangential")).strip().lower()
    )
    if model is None:
        raise ValueError(f"unsupported distortion model in {folder}")
    coeffs = np.asarray(cfg.get("distortion_coefficients", np.zeros(4)), dtype=float)
    cam = PinholeCamera(w, h, fu, fv, cu, cv, coeffs, model)

    stamps, rows = _read_csv(folder / "data.csv")
    paths = [folder / "data" / str(r[0]).strip() for r in rows]
    times = (stamps - origin_ns).astype(np.float64) * NS
    return CameraStream(folder.name, cam, _read_T_BS(cfg), times, stamps, paths)


def _load_imu(folder: Path, origin_ns: int):
    cfg = _read_yaml(folder / "sensor.yaml")
    noise = ImuNoise(
        acc_noise=float(cfg.get("accelerometer_noise_density", 2.0e-3)),
        gyro_noise=float(cfg.get("gyroscope_noise_density", 1.6968e-4)),
        acc_walk=float(cfg.get("accelerometer_random_walk", 3.0e-3)),
        gyro_walk=float(cfg.get("gyroscope_random_walk", 1.9393e-5)),
    )
    stamps, rows = _read_csv(folder / "data.csv")
    block = np.asarray(rows, dtype=float)
    times = (stamps - origin_ns).astype(np.float64) * NS
    # The ASL column order is gyro first, then accelerometer.
    imu = ImuData(times, block[:, 3:6], block[:, 0:3])
    return imu, noise, _read_T_BS(cfg)


def _load_groundtruth(folder: Path, origin_ns: int, source: str) -> GroundTruth:
    stamps, rows = _read_csv(folder / "data.csv")
    block = np.asarray(rows, dtype=float)
    times = (stamps - origin_ns).astype(np.float64) * NS
    positions = block[:, 0:3]
    quats = block[:, 3:7]  # w, x, y, z
    rotations = _quats_to_matrices(quats)
    velocities = block[:, 7:10] if block.shape[1] >= 10 else None
    bias_gyro = block[:, 10:13] if block.shape[1] >= 13 else None
    bias_acc = block[:, 13:16] if block.shape[1] >= 16 else None
    return GroundTruth(
        times, positions, rotations, velocities, bias_acc, bias_gyro, source
    )


def _quats_to_matrices(q: np.ndarray) -> np.ndarray:
    """``(N, 4)`` quaternions in ``w, x, y, z`` order to ``(N, 3, 3)``."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load(root, stereo: bool = True, name: str | None = None) -> Sequence:
    """Load a sequence from an ASL-layout directory.

    ``root`` may point either at the sequence folder or at the ``mav0`` inside
    it, because both are what people have on disk after unzipping.
    """
    root = Path(root)
    mav = root if root.name == "mav0" else root / "mav0"
    if not mav.is_dir():
        raise FileNotFoundError(f"no mav0 directory under {root}")

    cam_dirs = sorted(d for d in mav.glob("cam*") if (d / "data.csv").exists())
    if not cam_dirs:
        raise FileNotFoundError(f"no camera streams under {mav}")
    if not stereo:
        cam_dirs = cam_dirs[:1]

    # Rebase every clock to the first camera stamp; see the module docstring.
    first_stamps, _ = _read_csv(cam_dirs[0] / "data.csv")
    origin_ns = int(first_stamps[0])

    cameras = [_load_camera(d, origin_ns) for d in cam_dirs]
    imu, noise, T_body_imu = _load_imu(mav / "imu0", origin_ns)

    gt = None
    for folder, label in (
        (mav / "state_groundtruth_estimate0", "batch state estimate"),
        (mav / "vicon0", "vicon"),
        (mav / "mocap0", "mocap"),
        (mav / "leica0", "leica"),
    ):
        if (folder / "data.csv").exists():
            gt = _load_groundtruth(folder, origin_ns, label)
            break

    return Sequence(
        name=name or (root.name if root.name != "mav0" else root.parent.name),
        root=root,
        cameras=cameras,
        imu=imu,
        imu_noise=noise,
        T_body_imu=T_body_imu,
        groundtruth=gt,
        origin_ns=origin_ns,
    )


def available(root) -> list:
    """Sequence directories under ``root``, sorted by name."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir() if (d / "mav0" / "cam0" / "data.csv").exists()
    )
