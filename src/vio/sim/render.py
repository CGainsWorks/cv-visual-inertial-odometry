"""Rendering synthetic frames and writing a sequence out in ASL format.

What these images are: a textured scene built from thousands of small patches
stuck to the walls of the simulated room, each patch a different mixture of
spatial frequencies, projected through the same camera model the estimator
uses and stamped with sub-pixel accuracy.

What they are not: photorealistic, or subject to occlusion, motion blur,
rolling shutter, auto-exposure or illumination change.  Every one of those is a
real failure mode of a real front end and none of them is exercised here.
These images exist so continuous integration can run the *whole* pipeline --
detector, tracker, initialiser, back end, evaluation, on a machine with no
dataset and so a front-end regression shows up as a test failure rather than
as a worse number on EuRoC three weeks later.  Every quantitative claim in the
README comes from EuRoC, not from these.

Two properties took several attempts to get right and both are properties a
naive renderer gets wrong in a way that silently breaks tracking rather than
looking wrong:

**Low spatial frequencies have to be present.**  The first version drew
2-pixel Gaussian dots.  At the top of a four-level pyramid those dots are gone,
so KLT's coarse-to-fine search started from noise and converged onto a
*neighbouring* dot, consistently, on almost every feature, while reporting a
healthy status flag.  Patches here mix frequencies from about 3 to 18 pixels
per cycle.

**The whole image has to be textured, not just the tracked landmarks.**  A
field of isolated islands separated by flat background gives a corner detector
nothing to choose between and gives the pyramid nothing to downsample.  The
scene therefore carries several thousand texture points beyond the landmarks
the analytic tests use; the tracker finds its own corners in the result, which
is what it does on a real image too.

There is one thing these images are measurably *not* good enough for and it is
worth stating rather than discovering: **ORB descriptors do not match across
them**.  Between two frames ten apart, EuRoC V1_01 yields 386 ratio-test
matches out of 600 detections; this renderer yields about 25 and no amount of
adjusting the frequency content, the envelope width or the patch statistics
moved it, band-limited noise patches did no better than the sinusoid mixtures
used here.  The cause is structural: the patches are stuck to points but never
warp with viewpoint and they overlap, so the *mixture* a 31-pixel descriptor
window sees changes from frame to frame even though each patch does not.  KLT
is unaffected because it works at a smaller scale and re-anchors every frame,
which is why tracking is fine here and place recognition is not.

The consequence is that loop closure is exercised on EuRoC and on these images
only as far as the plumbing, retrieval, indexing, the pose graph.  Fixing it
properly means rendering textured *surfaces* with a real homography warp per
frame rather than pasting fixed patches, which is a different and much larger
renderer.  ``docs/limitations.md`` says the same thing where a reader will
find it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .simulator import SimSequence, _scatter_on_room_faces

#: Half-width of the patch stamped for each scene point, in pixels.
PATCH = 8

#: Number of sinusoidal components per patch.
_COMPONENTS = 4

#: Spatial frequency range, in cycles per pixel: periods of about 3 to 18 px.
_FREQ_MIN, _FREQ_MAX = 0.055, 0.32

#: Width of the Gaussian envelope, in pixels.
_ENVELOPE_SIGMA = 4.0

#: Extra texture points beyond the tracked landmarks.  Enough that a 752x480
#: frame is covered rather than dotted.
DEFAULT_TEXTURE_POINTS = 7000


class SceneRenderer:
    """A textured room, rendered from a simulated trajectory.

    Patches are precomputed once on an integer grid and stamped with bilinear
    weights, which is both fast enough to render a 500-frame sequence in a few
    seconds and accurate enough that the sub-pixel content is real rather than
    an artefact of rounding every landmark to the nearest pixel.
    """

    def __init__(
        self,
        seq: SimSequence,
        texture_points: int = DEFAULT_TEXTURE_POINTS,
        background: int = 118,
        noise_sigma: float = 2.0,
        contrast: tuple = (26.0, 48.0),
    ):
        self.seq = seq
        self.cfg = seq.config
        self.background = float(background)
        self.noise_sigma = float(noise_sigma)

        rng = np.random.default_rng(self.cfg.seed + 77003)
        extra = _scatter_on_room_faces(rng, int(texture_points), self.cfg.room)
        self.points = np.concatenate([seq.landmarks, extra], axis=0)
        self.patches = _build_patches(len(self.points), self.cfg.seed, contrast)

    def render(self, index: int) -> np.ndarray:
        """Render the frame at position ``index`` in the sequence."""
        T_wc = self.seq.states[index].pose * self.cfg.T_body_cam
        return self.render_pose(T_wc, index)

    def render_pose(self, T_world_cam, salt: int = 0) -> np.ndarray:
        cam = self.cfg.camera
        T_cw = T_world_cam.inverse()
        p_cam = T_cw.act(self.points)
        uv, ok = cam.project_points(p_cam)
        depth = p_cam[:, 2]
        ok &= (depth > self.cfg.min_depth) & (depth < self.cfg.max_depth)
        idx = np.nonzero(ok)[0]

        img = np.full((cam.height, cam.width), self.background)
        _stamp(img, uv[idx], self.patches[idx], cam.width, cam.height)

        if self.noise_sigma > 0.0:
            rng = np.random.default_rng(self.cfg.seed * 100003 + int(salt))
            img += rng.normal(0.0, self.noise_sigma, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def _build_patches(n: int, seed: int, contrast: tuple) -> np.ndarray:
    """One ``(2P+2, 2P+2)`` texture patch per scene point.

    The extra row and column exist so the bilinear stamp can read a shifted
    window without going out of bounds.
    """
    size = 2 * PATCH + 2
    grid = np.arange(size, dtype=float) - PATCH
    rng = np.random.default_rng(seed + 5501)

    angle = rng.uniform(0.0, np.pi, (n, _COMPONENTS))
    freq = rng.uniform(_FREQ_MIN, _FREQ_MAX, (n, _COMPONENTS))
    phase = rng.uniform(0.0, 2.0 * np.pi, (n, _COMPONENTS))
    weight = rng.uniform(0.4, 1.0, (n, _COMPONENTS))
    amp = rng.uniform(contrast[0], contrast[1], n)

    envelope = np.exp(
        -(grid[:, None] ** 2 + grid[None, :] ** 2)
        / (2.0 * _ENVELOPE_SIGMA * _ENVELOPE_SIGMA)
    )
    out = np.zeros((n, size, size))
    for c in range(_COMPONENTS):
        fx = (freq[:, c] * np.cos(angle[:, c]))[:, None, None]
        fy = (freq[:, c] * np.sin(angle[:, c]))[:, None, None]
        out += weight[:, c, None, None] * np.cos(
            2.0 * np.pi * (fx * grid[None, None, :] + fy * grid[None, :, None])
            + phase[:, c, None, None]
        )
    out *= envelope[None]
    # Normalise each patch to unit RMS before scaling, so the contrast
    # parameter means the same thing regardless of how the random frequencies
    # happened to interfere.
    rms = np.sqrt(np.mean(out ** 2, axis=(1, 2)))
    out *= (amp / np.maximum(rms, 1e-9))[:, None, None]
    return out


def _stamp(img: np.ndarray, uv: np.ndarray, patches: np.ndarray,
           width: int, height: int) -> None:
    """Add each patch to the image at its sub-pixel position, bilinearly."""
    size = 2 * PATCH + 1
    for (u, v), P in zip(uv, patches):
        x0 = int(np.floor(u)) - PATCH
        y0 = int(np.floor(v)) - PATCH
        if x0 < 0 or y0 < 0 or x0 + size > width or y0 + size > height:
            continue
        fx = float(u) - np.floor(u)
        fy = float(v) - np.floor(v)
        # The patch was built centred on its own origin, so a fractional
        # position shifts the *sampling* by (-fx, -fy).
        a = P[0:size, 0:size]
        b = P[0:size, 1:size + 1]
        c = P[1:size + 1, 0:size]
        d = P[1:size + 1, 1:size + 1]
        img[y0:y0 + size, x0:x0 + size] += (
            (1.0 - fy) * ((1.0 - fx) * a + fx * b)
            + fy * ((1.0 - fx) * c + fx * d)
        )


def render_sequence(seq: SimSequence, noise_sigma: float = 2.0,
                    texture_points: int = DEFAULT_TEXTURE_POINTS):
    """Every frame of a sequence, as a list of 8-bit images."""
    renderer = SceneRenderer(seq, texture_points, noise_sigma=noise_sigma)
    return [renderer.render(i) for i in range(len(seq.frames))]


def render_frame(seq: SimSequence, index: int, noise_sigma: float = 2.0,
                 texture_points: int = DEFAULT_TEXTURE_POINTS) -> np.ndarray:
    """A single frame.  Builds the scene each call, so use the class in a loop."""
    return SceneRenderer(seq, texture_points, noise_sigma=noise_sigma).render(index)


# ---------------------------------------------------------------------------
# ASL export
# ---------------------------------------------------------------------------

_CAM_YAML = """# Generated by vio.sim.render.  Synthetic sequence, not real data.
sensor_type: camera
comment: synthetic camera from vio.sim

T_BS:
  cols: 4
  rows: 4
  data: [{t00}, {t01}, {t02}, {t03},
         {t10}, {t11}, {t12}, {t13},
         {t20}, {t21}, {t22}, {t23},
         0.0, 0.0, 0.0, 1.0]

rate_hz: {rate}
resolution: [{width}, {height}]
camera_model: pinhole
intrinsics: [{fx}, {fy}, {cx}, {cy}]
distortion_model: {model}
distortion_coefficients: [{d0}, {d1}, {d2}, {d3}]
"""

_IMU_YAML = """# Generated by vio.sim.render.  Synthetic sequence, not real data.
sensor_type: imu
comment: synthetic IMU from vio.sim

T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
rate_hz: {rate}

gyroscope_noise_density: {gn}
gyroscope_random_walk: {gw}
accelerometer_noise_density: {an}
accelerometer_random_walk: {aw}
"""

_GT_YAML = """# Generated by vio.sim.render.  Synthetic sequence, not real data.
sensor_type: visual-inertial
comment: exact analytic ground truth

T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
"""


def _quaternion(R: np.ndarray):
    from ..io.trajectory import _matrix_to_quaternion

    q = _matrix_to_quaternion(R)
    return q[3], q[0], q[1], q[2]


def write_asl(
    seq: SimSequence,
    root,
    write_images: bool = True,
    noise_sigma: float = 2.0,
    groundtruth_rate: float = 200.0,
    texture_points: int = DEFAULT_TEXTURE_POINTS,
) -> Path:
    """Write a simulated sequence in the ASL layout the loader reads.

    Round-tripping the simulator through the same loader the real datasets use
    is what makes the loader itself testable: a field read out of place shows
    up as an error against a value the test already knows.
    """
    import cv2

    root = Path(root)
    mav = root / "mav0"
    cam_dir = mav / "cam0"
    (cam_dir / "data").mkdir(parents=True, exist_ok=True)
    (mav / "imu0").mkdir(parents=True, exist_ok=True)
    (mav / "state_groundtruth_estimate0").mkdir(parents=True, exist_ok=True)

    cfg = seq.config
    cam = cfg.camera
    T = cfg.T_body_cam.matrix()
    (cam_dir / "sensor.yaml").write_text(
        _CAM_YAML.format(
            rate=int(cfg.cam_rate), width=cam.width, height=cam.height,
            fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
            model={"radtan": "radial-tangential"}.get(cam.model, cam.model),
            d0=cam.distortion[0], d1=cam.distortion[1],
            d2=cam.distortion[2], d3=cam.distortion[3],
            **{f"t{r}{c}": T[r, c] for r in range(3) for c in range(4)},
        ),
        encoding="utf-8",
    )
    (mav / "imu0" / "sensor.yaml").write_text(
        _IMU_YAML.format(
            rate=int(cfg.imu_rate), gn=cfg.imu_noise.gyro_noise,
            gw=cfg.imu_noise.gyro_walk, an=cfg.imu_noise.acc_noise,
            aw=cfg.imu_noise.acc_walk,
        ),
        encoding="utf-8",
    )
    (mav / "state_groundtruth_estimate0" / "sensor.yaml").write_text(
        _GT_YAML, encoding="utf-8"
    )

    renderer = SceneRenderer(seq, texture_points, noise_sigma=noise_sigma)

    def ns(t):
        return int(round(float(t) * 1e9))

    with open(cam_dir / "data.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["#timestamp [ns]", "filename"])
        for i, f in enumerate(seq.frames):
            name = f"{ns(f.stamp)}.png"
            w.writerow([ns(f.stamp), name])
            if write_images:
                cv2.imwrite(str(cam_dir / "data" / name), renderer.render(i))

    with open(mav / "imu0" / "data.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["#timestamp [ns]", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
        for t, g, a in zip(seq.imu.t, seq.imu.gyro, seq.imu.acc):
            if t < 0:
                continue
            w.writerow([ns(t)] + [f"{v:.12g}" for v in (*g, *a)])

    tr = cfg.trajectory
    step = 1.0 / groundtruth_rate
    times = np.arange(0.0, cfg.duration + 1e-9, step)
    with open(mav / "state_groundtruth_estimate0" / "data.csv", "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["#timestamp", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z",
                    "v_x", "v_y", "v_z", "b_w_x", "b_w_y", "b_w_z",
                    "b_a_x", "b_a_y", "b_a_z"])
        for t in times:
            t = float(t)
            b = seq.bias_at(t)
            qw, qx, qy, qz = _quaternion(tr.R(t))
            w.writerow([ns(t)] + [f"{v:.12g}" for v in (
                *tr.p(t), qw, qx, qy, qz, *tr.v(t), *b.gyro, *b.acc
            )])
    return root
