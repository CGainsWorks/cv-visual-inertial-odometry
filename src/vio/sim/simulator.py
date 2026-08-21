"""A synthetic visual-inertial sequence with exact ground truth.

This is the most useful file in the repository during development and the
reason the estimator converged at all.  Real data gives you one number at the
end and no way to tell which of eight components produced it.  This gives every
component an exact answer to be checked against:

* preintegration is checked against the analytic trajectory, not against
  another integrator;
* the back-end Jacobians are checked against finite differences of the
  residuals they claim to differentiate;
* initialisation is checked against a known gravity direction, a known scale
  and a known bias, all three of which it has to recover from nothing;
* the temporal calibration is checked against a time offset that was
  deliberately injected, so a value of zero is a failure rather than a pass.

Everything here is configurable and seeded.  Nothing here is photorealistic --
see :mod:`vio.sim.render` for what the images look like and why that is a
limitation worth naming rather than hiding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose
from ..geometry.camera import PinholeCamera
from ..types import GRAVITY, ImuBias, ImuData, ImuNoise, NavState
from .trajectory import Trajectory

#: Camera mounted looking along body ``+x``, with the usual computer-vision
#: axes: camera ``z`` forward, ``x`` right, ``y`` down.  Matches the sense of
#: the EuRoC ``cam0`` extrinsic without pretending to be the same numbers.
R_BODY_CAM = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def default_camera() -> PinholeCamera:
    """A camera with EuRoC ``cam0`` geometry, so the pixel scales are realistic."""
    return PinholeCamera(
        width=752,
        height=480,
        fx=458.654,
        fy=457.296,
        cx=367.215,
        cy=248.375,
        distortion=np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]),
        model="radtan",
    )


@dataclass
class SimConfig:
    """Everything that decides what a generated sequence looks like."""

    duration: float = 30.0
    imu_rate: float = 200.0
    cam_rate: float = 20.0
    n_landmarks: int = 1200
    pixel_noise: float = 0.5
    max_depth: float = 22.0
    min_depth: float = 1.5
    seed: int = 0

    #: Room half-extents in x and y and height in z.  Landmarks are scattered
    #: on the six faces, which is a closer match to a real indoor scene than a
    #: cloud floating in the middle of the room would be.
    #:
    #: The room size and the ``min_depth`` above are not free parameters.  The
    #: default trajectory and camera together have to put the inter-frame pixel
    #: motion in the range KLT can actually resolve against a field of isolated
    #: blobs, about six pixels at 20 Hz, which is what EuRoC's own sequences
    #: do.  An earlier version of these defaults produced seventeen pixels per
    #: frame, past half the blob spacing and the tracker mismatched
    #: neighbouring blobs on almost every feature.  The assertion in
    #: ``tests/test_io_and_sim.py`` keeps it in range.
    room: tuple = (8.5, 7.0, 3.4)

    imu_noise: ImuNoise = field(default_factory=ImuNoise)
    initial_bias: ImuBias = field(
        default_factory=lambda: ImuBias(
            acc=np.array([0.03, -0.02, 0.05]), gyro=np.array([0.002, -0.003, 0.001])
        )
    )
    bias_walk: bool = True

    camera: PinholeCamera = field(default_factory=default_camera)
    T_body_cam: Pose = field(
        default_factory=lambda: Pose(R_BODY_CAM, np.array([0.05, 0.01, -0.02]))
    )

    #: Ground-truth camera-to-IMU time offset, in seconds.  The true exposure
    #: time of a frame recorded at ``t`` is ``t + time_offset``.  Non-zero by
    #: default so that a calibration routine returning zero is caught.
    time_offset: float = 0.0

    trajectory: Trajectory = field(default_factory=lambda: Trajectory(yaw_rate=0.11))
    gravity: np.ndarray = field(default_factory=lambda: GRAVITY.copy())


@dataclass
class Frame:
    """One camera frame: which landmarks it saw and where."""

    index: int
    #: The timestamp as it would be *recorded*, before any time-offset
    #: correction.  The estimator only ever sees this one.
    stamp: float
    #: The instant the frame was actually exposed, ``stamp + time_offset``.
    true_stamp: float
    ids: np.ndarray
    uv: np.ndarray
    #: Noise-free pixel positions, for measuring how much of the final error is
    #: the estimator and how much is the measurement noise it was given.
    uv_clean: np.ndarray


@dataclass
class SimSequence:
    """A generated sequence and everything true about it."""

    config: SimConfig
    imu: ImuData
    frames: list
    landmarks: np.ndarray
    #: Ground-truth navigation state at each frame's true exposure time.
    states: list
    #: Ground-truth bias at each IMU sample, which drifts if ``bias_walk``.
    bias_acc: np.ndarray
    bias_gyro: np.ndarray

    @property
    def times(self) -> np.ndarray:
        return np.array([f.stamp for f in self.frames])

    @property
    def true_times(self) -> np.ndarray:
        return np.array([f.true_stamp for f in self.frames])

    def positions(self) -> np.ndarray:
        return np.stack([s.p for s in self.states])

    def path_length(self) -> float:
        return float(
            np.linalg.norm(np.diff(self.positions(), axis=0), axis=1).sum()
        )

    def state_at(self, t: float) -> NavState:
        """Exact ground truth at an arbitrary time, straight from the analytic form."""
        tr = self.config.trajectory
        return NavState(tr.R(t), tr.p(t), tr.v(t))

    def bias_at(self, t: float) -> ImuBias:
        i = int(np.clip(np.searchsorted(self.imu.t, t), 0, len(self.imu.t) - 1))
        return ImuBias(self.bias_acc[i], self.bias_gyro[i])


#: Fraction of scene points placed on furniture-sized boxes standing in the
#: room rather than on its walls.
CLUTTER_FRACTION = 0.35

#: Standard deviation of the displacement applied to wall points, in metres.
#:
#: This is not cosmetic.  Six exactly flat walls make the scene *planar over
#: any single view* and a plane is a degenerate configuration for the
#: fundamental matrix: it admits a one-parameter family of solutions, so
#: RANSAC returns a confident and arbitrary one.  With 0.06 m of jitter the
#: front end's two-view check was discarding 34 tracks per frame on synthetic
#: images against 0.2 per frame on EuRoC and mean track length was five frames
#: against thirty.  The scene has to have real depth structure, which is what
#: the clutter boxes and this larger jitter provide.
WALL_ROUGHNESS = 0.35


def _scatter_on_room_faces(rng: np.random.Generator, n: int, room) -> np.ndarray:
    """Scene points on the walls of a room, plus boxes standing in it."""
    hx, hy, hz = float(room[0]), float(room[1]), float(room[2])
    n_clutter = int(n * CLUTTER_FRACTION)
    n_wall = n - n_clutter

    per = int(np.ceil(n_wall / 6))
    u = lambda a, b, k: rng.uniform(a, b, k)  # noqa: E731
    faces = [
        np.stack([np.full(per, hx), u(-hy, hy, per), u(0.0, hz, per)], 1),
        np.stack([np.full(per, -hx), u(-hy, hy, per), u(0.0, hz, per)], 1),
        np.stack([u(-hx, hx, per), np.full(per, hy), u(0.0, hz, per)], 1),
        np.stack([u(-hx, hx, per), np.full(per, -hy), u(0.0, hz, per)], 1),
        np.stack([u(-hx, hx, per), u(-hy, hy, per), np.full(per, hz)], 1),
        np.stack([u(-hx, hx, per), u(-hy, hy, per), np.zeros(per)], 1),
    ]
    wall = np.concatenate(faces, axis=0)[:n_wall]
    wall = wall + rng.normal(0.0, WALL_ROUGHNESS, wall.shape)

    # Boxes: axis-aligned, standing on the floor, scattered around the room but
    # kept clear of the middle where the trajectory runs.
    n_boxes = max(int(n_clutter / 90), 6)
    centres = np.stack([
        rng.uniform(-hx * 0.8, hx * 0.8, n_boxes),
        rng.uniform(-hy * 0.8, hy * 0.8, n_boxes),
        np.zeros(n_boxes),
    ], axis=1)
    sizes = rng.uniform(0.4, 1.6, (n_boxes, 3))
    sizes[:, 2] = rng.uniform(0.5, 2.2, n_boxes)
    which = rng.integers(0, n_boxes, n_clutter)
    local = rng.uniform(-0.5, 0.5, (n_clutter, 3))
    # Push each point onto the nearest face of its box so the clutter is
    # surfaces rather than a fog of points.
    axis = rng.integers(0, 3, n_clutter)
    local[np.arange(n_clutter), axis] = np.sign(
        local[np.arange(n_clutter), axis]
    ) * 0.5
    clutter = centres[which] + local * sizes[which]
    clutter[:, 2] = np.abs(clutter[:, 2])

    pts = np.concatenate([wall, clutter], axis=0)
    rng.shuffle(pts)
    return pts[:n]


def simulate(config: SimConfig | None = None) -> SimSequence:
    """Generate a sequence.  Deterministic given ``config.seed``."""
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    tr = cfg.trajectory

    # -- IMU ---------------------------------------------------------------
    dt = 1.0 / cfg.imu_rate
    # A little lead-in and lead-out so every camera frame has IMU either side
    # of it even after the time offset is applied.
    t_imu = np.arange(-0.05, cfg.duration + 0.05 + 0.5 * dt, dt)
    n = len(t_imu)

    bias_acc = np.empty((n, 3))
    bias_gyro = np.empty((n, 3))
    bias_acc[0] = cfg.initial_bias.acc
    bias_gyro[0] = cfg.initial_bias.gyro
    if cfg.bias_walk:
        wa = rng.normal(0.0, cfg.imu_noise.acc_walk * np.sqrt(dt), (n - 1, 3))
        wg = rng.normal(0.0, cfg.imu_noise.gyro_walk * np.sqrt(dt), (n - 1, 3))
        bias_acc[1:] = bias_acc[0] + np.cumsum(wa, axis=0)
        bias_gyro[1:] = bias_gyro[0] + np.cumsum(wg, axis=0)
    else:
        bias_acc[1:] = bias_acc[0]
        bias_gyro[1:] = bias_gyro[0]

    acc_true = np.stack([tr.specific_force(float(t), cfg.gravity) for t in t_imu])
    gyro_true = np.stack([tr.omega_body(float(t)) for t in t_imu])

    acc = (
        acc_true
        + bias_acc
        + rng.normal(0.0, cfg.imu_noise.acc_noise / np.sqrt(dt), (n, 3))
    )
    gyro = (
        gyro_true
        + bias_gyro
        + rng.normal(0.0, cfg.imu_noise.gyro_noise / np.sqrt(dt), (n, 3))
    )
    imu = ImuData(t_imu, acc, gyro)

    # -- landmarks ---------------------------------------------------------
    landmarks = _scatter_on_room_faces(rng, cfg.n_landmarks, cfg.room)

    # -- camera frames -----------------------------------------------------
    stamps = np.arange(0.0, cfg.duration + 1e-9, 1.0 / cfg.cam_rate)
    frames = []
    states = []
    for i, stamp in enumerate(stamps):
        true_t = float(stamp) + cfg.time_offset
        T_wb = tr.pose(true_t)
        T_wc = T_wb * cfg.T_body_cam
        T_cw = T_wc.inverse()
        p_cam = T_cw.act(landmarks)
        uv, ok = cfg.camera.project_points(p_cam)
        depth = p_cam[:, 2]
        ok &= (depth > cfg.min_depth) & (depth < cfg.max_depth)
        ids = np.nonzero(ok)[0]
        clean = uv[ids]
        noisy = clean + rng.normal(0.0, cfg.pixel_noise, clean.shape)
        frames.append(
            Frame(
                index=i,
                stamp=float(stamp),
                true_stamp=true_t,
                ids=ids,
                uv=noisy,
                uv_clean=clean,
            )
        )
        states.append(NavState(tr.R(true_t), tr.p(true_t), tr.v(true_t)))

    return SimSequence(
        config=cfg,
        imu=imu,
        frames=frames,
        landmarks=landmarks,
        states=states,
        bias_acc=bias_acc,
        bias_gyro=bias_gyro,
    )
