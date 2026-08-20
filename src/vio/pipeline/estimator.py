"""The sliding-window estimator: front end, initialisation, back end, output.

One class runs both configurations the project compares.  With ``use_imu``
false it is a monocular visual odometry that knows nothing about scale; with it
true the same code path gains IMU factors, an initialisation step and a bias
state.  Keeping them in one class rather than two is deliberate: the point of
the comparison is what the inertial measurements buy, and that answer is worth
nothing if the visual baseline is a different, weaker program.

Life cycle
----------

``BOOTSTRAP``
    Accumulate keyframes and wait for enough parallax between the first and the
    latest to run a two-view initialisation.  Triangulate, run a visual bundle
    adjustment, then keep adding keyframes by PnP.  Scale is arbitrary here even
    when the IMU is enabled, because nothing has told it otherwise yet.

``ALIGNING`` (inertial only)
    Once there are enough keyframes, hand the up-to-scale trajectory and the
    preintegrated blocks to :func:`vio.imu.initialisation.initialise`.  On
    success the whole window, poses, velocities, landmarks, is rescaled and
    rotated into a gravity-aligned frame.  On failure it keeps collecting and
    retries, which is the right behaviour when the platform has not yet moved
    enough for scale to be observable.

``RUNNING``
    Per keyframe: predict, associate, triangulate what is new, optimise the
    window, marginalise the oldest keyframe into a linear prior.

Output is at *frame* rate, not keyframe rate.  Between keyframes the pose is
propagated from the last optimised state, by IMU preintegration when it is
available and by constant velocity when it is not.  Reporting only keyframe
poses would make the trajectory error look better than it is, because
keyframes are exactly the instants the estimator has just fitted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..backend import factors as F
from ..backend.graph import FactorGraph
from ..backend.groups import ReprojectionGroup
from ..backend.marginalisation import marginalise
from ..backend.solver import SolverOptions, optimise
from ..backend.variables import (
    BiasVariable,
    PointVariable,
    PoseVariable,
    Values,
    VectorVariable,
)
from ..frontend import geometry as G
from ..frontend.tracker import Tracker, TrackerConfig
from ..geometry import Pose, so3
from ..geometry.camera import PinholeCamera
from ..imu.initialisation import initialise
from ..imu.preintegration import Preintegrated, preintegrate
from ..types import GRAVITY, ImuBias, ImuData, ImuNoise, NavState


@dataclass
class EstimatorConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    use_imu: bool = True
    #: Keyframes held in the sliding window.  Ten is the usual compromise:
    #: fewer and the accelerometer bias is not observable within it, more and
    #: the solve time grows faster than the accuracy does.
    window_size: int = 10

    #, keyframe policy ---------------------------------------------------
    #: Median normalised-coordinate parallax against the previous keyframe.
    #: 0.02 is roughly nine pixels at EuRoC's focal length.
    keyframe_parallax: float = 0.02
    #: Force a keyframe if tracking has decayed this far, even without parallax.
    keyframe_track_ratio: float = 0.55
    keyframe_min_interval: float = 0.045
    keyframe_max_interval: float = 0.5

    #, landmarks ---------------------------------------------------------
    #: Keyframe observations a track needs before it is triangulated.  Two
    #: is the geometric minimum and the parallax test in
    #: vio.frontend.geometry is what actually guards quality; requiring
    #: three starves the map on sequences where tracks are short.
    min_track_length: int = 2
    min_triangulation_angle: float = 1.0
    max_landmarks: int = 150
    #: Metres, applied once the map is metric.
    max_landmark_distance: float = 60.0
    #: Before the inertial alignment the map has no units, so the same cap is
    #: expressed as a multiple of the median landmark depth, which the
    #: bootstrap normalises to one.
    bootstrap_max_depth_ratio: float = 15.0
    #: Reprojection error above which a landmark is dropped from the window,
    #: in pixels, applied after each optimisation.
    outlier_pixels: float = 3.0

    #, weighting ---------------------------------------------------------
    pixel_sigma: float = 1.0
    huber_pixels: float = 2.0

    #, initialisation ----------------------------------------------------
    bootstrap_keyframes: int = 12
    bootstrap_parallax: float = 0.035
    max_bootstrap_keyframes: int = 60

    #, solver ------------------------------------------------------------
    window_iterations: int = 8
    bootstrap_iterations: int = 15
    #: Turning this off replaces the marginal prior with a fixed-covariance
    #: prior on the oldest kept keyframe, which is what most people reach for
    #: first.  ``scripts/ablate_backend.py`` measures the difference.
    marginalise: bool = True
    #: Standard deviations for the fallback prior: rotation (rad), position (m).
    fallback_prior_sigma: tuple = (0.02, 0.05)

    gravity: np.ndarray = field(default_factory=lambda: GRAVITY.copy())


@dataclass
class Keyframe:
    kf_id: int
    frame_index: int
    stamp: float
    #: track id -> undistorted normalised coordinate.
    features: dict = field(default_factory=dict)
    preintegrated: Preintegrated | None = None

    @property
    def pose_key(self) -> str:
        return f"x{self.kf_id}"

    @property
    def vel_key(self) -> str:
        return f"v{self.kf_id}"

    @property
    def bias_key(self) -> str:
        return f"b{self.kf_id}"


@dataclass
class FrameEstimate:
    index: int
    stamp: float
    pose: Pose
    keyframe: bool
    initialised: bool


@dataclass
class Timing:
    """Wall-clock per stage, accumulated over a run."""

    track: float = 0.0
    triangulate: float = 0.0
    optimise: float = 0.0
    marginalise: float = 0.0
    preintegrate: float = 0.0
    other: float = 0.0
    frames: int = 0
    keyframes: int = 0
    solver_iterations: int = 0

    def total(self) -> float:
        return (
            self.track + self.triangulate + self.optimise
            + self.marginalise + self.preintegrate + self.other
        )

    def as_dict(self) -> dict:
        n = max(self.frames, 1)
        k = max(self.keyframes, 1)
        return {
            "frames": self.frames,
            "keyframes": self.keyframes,
            "total_s": round(self.total(), 3),
            "ms_per_frame": round(1e3 * self.total() / n, 3),
            "ms_track_per_frame": round(1e3 * self.track / n, 3),
            "ms_preintegrate_per_frame": round(1e3 * self.preintegrate / n, 3),
            "ms_triangulate_per_keyframe": round(1e3 * self.triangulate / k, 3),
            "ms_optimise_per_keyframe": round(1e3 * self.optimise / k, 3),
            "ms_marginalise_per_keyframe": round(1e3 * self.marginalise / k, 3),
            "solver_iterations_per_keyframe": round(self.solver_iterations / k, 2),
            "effective_hz": round(n / max(self.total(), 1e-9), 2),
        }


class Estimator:
    """Monocular visual or visual-inertial odometry over an image stream."""

    def __init__(
        self,
        camera: PinholeCamera,
        T_body_cam: Pose,
        imu: ImuData | None = None,
        imu_noise: ImuNoise | None = None,
        config: EstimatorConfig | None = None,
    ):
        self.cfg = config or EstimatorConfig()
        self.camera = camera
        self.T_body_cam = T_body_cam.copy()
        self.imu = imu
        self.imu_noise = imu_noise or ImuNoise()
        if self.cfg.use_imu and imu is None:
            raise ValueError("use_imu is set but no IMU data was supplied")

        self.tracker = Tracker(camera, self.cfg.tracker)
        self.values = Values()
        self.keyframes: list = []
        self.landmarks: dict = {}
        self.state = "BOOTSTRAP"
        self.trajectory: list = []
        self.timing = Timing()
        self.events: list = []
        self.event_counts: dict = {}

        self._next_kf_id = 0
        self._prior = None
        self._prior_keys: list = []
        self._last_stamp = None
        self._bias = ImuBias()
        self._velocity = np.zeros(3)
        self._last_features = None
        self._scale_applied = 1.0
        self._n_bootstrap_attempts = 0

        # Loop closure runs alongside rather than inside the sliding window,
        # which is how VINS-Mono arranges it too: the window is a fixed-lag
        # smoother with a bounded cost per keyframe, and a loop correction is a
        # global operation over every keyframe ever seen.  Mixing them would
        # make the per-keyframe cost unbounded.
        self.loop_detector = None
        self.closures: list = []
        #: Final pose of every keyframe, recorded as it leaves the window and
        #: kept current for those still inside it.
        self.keyframe_poses: dict = {}
        self.keyframe_stamps: dict = {}

    #, public -----------------------------------------------------------

    @property
    def initialised(self) -> bool:
        return self.state == "RUNNING"

    def process(self, image: np.ndarray, stamp: float, index: int | None = None):
        """Feed one image.  Returns a :class:`FrameEstimate`, always."""
        t0 = time.perf_counter()
        gyro_rotation = self._gyro_rotation(stamp)
        self.timing.preintegrate += time.perf_counter() - t0

        t0 = time.perf_counter()
        feats = self.tracker.process(image, stamp, index, gyro_rotation)
        self.timing.track += time.perf_counter() - t0
        self.timing.frames += 1

        is_kf = self._is_keyframe(feats)
        if is_kf:
            self._add_keyframe(feats)
            self.timing.keyframes += 1
            if self.keyframes:
                self._record_keyframe_pose(self.keyframes[-1])
                self._offer_to_loop_detector(self.keyframes[-1], image)

        pose = self._current_pose(stamp)
        est = FrameEstimate(feats.index, float(stamp), pose, is_kf, self.initialised)
        # self.initialised is read after the keyframe has been processed, so a
        # frame that triggered the alignment is correctly marked initialised.
        self.trajectory.append(est)
        self._last_stamp = stamp
        self._last_features = feats
        return est

    def poses(self, initialised_only: bool = False):
        """``(stamps, poses)`` for every frame that has one.

        With ``initialised_only`` the bootstrap frames are dropped.  They are
        in the same coordinate frame as everything else, the alignment is
        applied retrospectively, but they were produced by visual odometry
        with no scale, so including them measures a different system from the
        one being reported on.
        """
        rows = [e for e in self.trajectory if e.pose is not None]
        if initialised_only:
            rows = [e for e in rows if e.initialised]
        return np.array([e.stamp for e in rows]), [e.pose for e in rows]

    def keyframe_trajectory(self):
        """``(ids, stamps, poses)`` for every keyframe, in order."""
        for kf in self.keyframes:
            self._record_keyframe_pose(kf)
        ids = sorted(self.keyframe_poses)
        return (
            ids,
            [self.keyframe_stamps[k] for k in ids],
            [self.keyframe_poses[k] for k in ids],
        )

    def landmark_cloud(self) -> np.ndarray:
        pts = [
            self.values.vector(f"l{k}")
            for k in self.landmarks
            if f"l{k}" in self.values
        ]
        return np.stack(pts) if pts else np.zeros((0, 3))

    #, keyframe policy ---------------------------------------------------

    def _is_keyframe(self, feats) -> bool:
        cfg = self.cfg
        if not self.keyframes:
            return len(feats.ids) >= cfg.tracker.min_tracks
        last = self.keyframes[-1]
        dt = feats.stamp - last.stamp
        if dt < cfg.keyframe_min_interval:
            return False
        if dt >= cfg.keyframe_max_interval:
            return True

        common = [i for i in feats.ids if int(i) in last.features]
        if len(common) < cfg.tracker.min_tracks:
            return True
        ratio = len(common) / max(len(last.features), 1)
        if ratio < cfg.keyframe_track_ratio:
            return True

        index = {int(i): j for j, i in enumerate(feats.ids)}
        disp = [
            np.linalg.norm(feats.xn[index[int(i)]] - last.features[int(i)])
            for i in common
        ]
        return float(np.median(disp)) > cfg.keyframe_parallax

    #, keyframe handling -------------------------------------------------

    def attach_loop_detector(self, detector) -> None:
        """Feed every keyframe to a :class:`vio.loop_closure.detector.LoopDetector`.

        The estimator does not act on the closures itself.  It records them,
        and ``scripts/run_sequence.py --loop`` hands them to the pose graph
        once the sequence is finished.  Correcting inside the loop would mean
        the trajectory reported for frame *n* depends on frames after it, which
        is a batch result presented as an online one.
        """
        self.loop_detector = detector

    def _record_keyframe_pose(self, kf) -> None:
        if kf.pose_key in self.values:
            self.keyframe_poses[kf.kf_id] = self.values.pose(kf.pose_key).copy()
            self.keyframe_stamps[kf.kf_id] = kf.stamp

    def _offer_to_loop_detector(self, kf, image) -> None:
        if self.loop_detector is None or kf.pose_key not in self.values:
            return
        T_wc = self._camera_pose(kf)
        T_cw = T_wc.inverse()
        uv, points_cam = [], []
        for lid in kf.features:
            key = f"l{lid}"
            if key not in self.values:
                continue
            p_cam = T_cw.act(self.values.vector(key))
            if p_cam[2] <= 0.05:
                continue
            pix, ok = self.camera.project_points(p_cam[None, :])
            if not ok[0]:
                continue
            uv.append(pix[0])
            points_cam.append(p_cam)
        self.loop_detector.add(
            kf.kf_id, kf.stamp, self.values.pose(kf.pose_key), image,
            np.stack(uv) if uv else None,
            np.stack(points_cam) if points_cam else None,
        )
        closure = self.loop_detector.detect(kf.kf_id)
        if closure is not None:
            self.closures.append(closure)
            self._log("loop closure", query=closure.query_id, match=closure.match_id,
                      score=round(closure.score, 4), inliers=closure.n_inliers,
                      correction_m=round(closure.correction, 3))

    def _add_keyframe(self, feats) -> None:
        kf = Keyframe(
            kf_id=self._next_kf_id,
            frame_index=feats.index,
            stamp=float(feats.stamp),
            features={int(i): np.asarray(x) for i, x in zip(feats.ids, feats.xn)},
        )
        self._next_kf_id += 1

        if self.keyframes and self.imu is not None:
            t0 = time.perf_counter()
            kf.preintegrated = preintegrate(
                self.imu, self.keyframes[-1].stamp, kf.stamp,
                bias=self._bias, noise=self.imu_noise,
            )
            self.timing.preintegrate += time.perf_counter() - t0

        self.keyframes.append(kf)

        if self.state == "BOOTSTRAP":
            self._bootstrap_step(kf)
        else:
            self._running_step(kf)

        # The window is allowed to grow past its steady-state size while
        # bootstrapping, because the inertial alignment needs a longer span
        # than the running window does, with a bootstrap target larger than
        # window_size and no exception here, the window slides first and the
        # alignment is never attempted at all.
        limit = (
            self.cfg.window_size
            if self.state == "RUNNING"
            else max(self.cfg.window_size, self.cfg.bootstrap_keyframes)
        )
        while len(self.keyframes) > limit:
            self._slide()
        self.tracker.prune(
            self.keyframes[0].frame_index if self.keyframes else feats.index
        )

    #, bootstrap ---------------------------------------------------------

    def _bootstrap_step(self, kf: Keyframe) -> None:
        cfg = self.cfg
        if len(self.keyframes) < 2:
            return

        if not self.values:
            if not self._two_view_init():
                # Keep only the newest keyframe as the new reference, otherwise
                # a stationary start accumulates keyframes that will never have
                # parallax against each other.
                if len(self.keyframes) > cfg.max_bootstrap_keyframes:
                    self.keyframes = self.keyframes[-2:]
                return
            self._log("visual init", keyframes=len(self.keyframes))
            return

        # Structure exists: locate the new keyframe against it, then refine.
        if not self._locate_by_pnp(kf):
            self._log("pnp failed during bootstrap", kf=kf.kf_id)
            # Drop the keyframe rather than leave an unposed one in the window.
            # An unposed keyframe contributes nothing, blocks triangulation of
            # everything it sees, and each one makes the next PnP harder, which
            # turns one failure into a permanent one.
            self.keyframes.pop()
            return
        t0 = time.perf_counter()
        self._triangulate_new()
        self.timing.triangulate += time.perf_counter() - t0
        self._optimise(cfg.bootstrap_iterations, visual_only=True)

        if len(self.keyframes) >= cfg.bootstrap_keyframes:
            if cfg.use_imu:
                self._try_inertial_alignment()
            else:
                # Visual-only has nothing to align.  Once there is a stable
                # window of structure it is running, and saying so is what
                # switches it to the cheaper per-keyframe iteration budget and
                # the steady-state window size.
                self.state = "RUNNING"
                self._log("visual bootstrap complete",
                          keyframes=len(self.keyframes),
                          landmarks=len(self.landmarks))

    def _two_view_init(self) -> bool:
        cfg = self.cfg
        first = self.keyframes[0]
        last = self.keyframes[-1]
        common = [i for i in last.features if i in first.features]
        if len(common) < 30:
            return False
        x0 = np.stack([first.features[i] for i in common])
        x1 = np.stack([last.features[i] for i in common])
        disp = float(np.median(np.linalg.norm(x1 - x0, axis=1)))
        if disp < cfg.bootstrap_parallax:
            return False

        rel = G.relative_pose(x0, x1)
        if not rel.ok or rel.parallax_deg < cfg.min_triangulation_angle:
            return False

        # Camera 0 is the origin.  The scale is whatever recoverPose's unit
        # translation implies; the inertial alignment fixes it later, and for
        # the visual-only baseline it never gets fixed at all, which is the
        # honest state of a monocular system.
        T_wc0 = Pose.identity()
        T_wc1 = rel.T_0_1
        self.values = Values()
        self._set_pose(first, T_wc0)
        self._set_pose(last, T_wc1)
        self.keyframes = [first, last]
        if self.imu is not None:
            last.preintegrated = preintegrate(
                self.imu, first.stamp, last.stamp, bias=self._bias, noise=self.imu_noise
            )

        points = {}
        for i, keep in zip(common, rel.inliers):
            if not keep:
                continue
            p, ok = G.triangulate_two_view(
                T_wc0, T_wc1, first.features[i], last.features[i],
                cfg.min_triangulation_angle,
            )
            if ok:
                points[i] = p
        if len(points) < 20:
            self.values = Values()
            self.landmarks = {}
            return False

        # Normalise the map so the median landmark depth is one.
        #
        # recoverPose returns a unit translation, so the bootstrap map is built
        # at whatever scale that implies, on this data the first keyframe
        # pair is about 4 cm apart, which makes every distance in the map
        # twenty-five times too large.  Any threshold expressed in metres is
        # then meaningless until the inertial alignment arrives: with a
        # 60-metre landmark cap the bootstrap was silently discarding
        # everything past 2.4 real metres, in an eight-metre room.  Fixing the
        # median depth at one makes the bootstrap thresholds ratios, which is
        # the only thing a monocular map can meaningfully express, and it
        # conditions the bundle adjustment better as a side effect.
        depths = np.array([
            float((T_wc0.R.T @ (p - T_wc0.t))[2]) for p in points.values()
        ])
        median_depth = float(np.median(depths[depths > 0]))
        if not np.isfinite(median_depth) or median_depth <= 1e-9:
            self.values = Values()
            self.landmarks = {}
            return False
        inv = 1.0 / median_depth
        T_wc1 = Pose(T_wc1.R, T_wc1.t * inv)
        self._set_pose(last, T_wc1)

        n = 0
        for i, p in points.items():
            q = p * inv
            if np.linalg.norm(q - T_wc0.t) > cfg.bootstrap_max_depth_ratio:
                continue
            self.landmarks[i] = {first.kf_id, last.kf_id}
            self.values[f"l{i}"] = PointVariable(q)
            n += 1
        if n < 20:
            self.values = Values()
            self.landmarks = {}
            return False
        self._optimise(cfg.bootstrap_iterations, visual_only=True)
        return True

    def _try_inertial_alignment(self) -> None:
        self._n_bootstrap_attempts += 1
        poses_wc = [self.values.pose(k.pose_key) * self.T_body_cam
                    for k in self.keyframes]
        blocks = []
        for a, b in zip(self.keyframes[:-1], self.keyframes[1:]):
            blocks.append(
                preintegrate(self.imu, a.stamp, b.stamp, noise=self.imu_noise)
            )
        result = initialise(poses_wc, blocks, self.T_body_cam)
        if not result.ok:
            self._log("alignment rejected", reason=result.reason,
                      keyframes=len(self.keyframes))
            if len(self.keyframes) > self.cfg.max_bootstrap_keyframes:
                self.keyframes = self.keyframes[-self.cfg.window_size:]
                self.values = Values()
                self.landmarks = {}
            return

        self._apply_alignment(result)
        self.state = "RUNNING"
        self._log(
            "inertial alignment", scale=round(result.scale, 4),
            gravity=np.round(result.gravity, 3).tolist(),
            gyro_bias=np.round(result.gyro_bias, 5).tolist(),
            excitation=round(result.excitation, 3),
            condition=round(result.condition_number, 1),
            keyframes=len(self.keyframes),
        )

    def _apply_alignment(self, result) -> None:
        """Rescale and rotate the window into a metric, gravity-aligned frame."""
        s = result.scale
        Ra = result.R_align
        for kf, v in zip(self.keyframes, result.velocities):
            T = self.values.pose(kf.pose_key)
            new = Pose(so3.project(Ra @ T.R), Ra @ (s * T.t))
            self.values[kf.pose_key] = PoseVariable(new)
            self.values[kf.vel_key] = VectorVariable(Ra @ v)
            self.values[kf.bias_key] = BiasVariable(
                np.concatenate([np.zeros(3), result.gyro_bias])
            )
        for lid in list(self.landmarks):
            key = f"l{lid}"
            if key in self.values:
                self.values[key] = PointVariable(Ra @ (s * self.values.vector(key)))
        # Frames already emitted are in the bootstrap frame, which is neither
        # metric nor gravity aligned.  Leaving them there makes the output
        # trajectory a mixture of two coordinate systems, and any global
        # alignment done during evaluation then tries to fit both at once --
        # which showed up as a constant 15 degrees of apparent rotation error
        # on a run whose actual gravity tilt was under two.  The same
        # transform is applied retrospectively so the whole trajectory is in
        # one frame; ``poses(initialised_only=True)`` is there for the stricter
        # comparison that reports only what the initialised estimator produced.
        for entry in self.trajectory:
            if entry.pose is not None:
                entry.pose = Pose(
                    so3.project(Ra @ entry.pose.R), Ra @ (s * entry.pose.t)
                )
        self._bias = ImuBias(np.zeros(3), result.gyro_bias)
        self._scale_applied = s
        # Re-preintegrate at the newly known gyro bias before the first
        # inertial solve, so the factors are not linearised about zero.
        for a, b in zip(self.keyframes[:-1], self.keyframes[1:]):
            b.preintegrated = preintegrate(
                self.imu, a.stamp, b.stamp, bias=self._bias, noise=self.imu_noise
            )
        self._optimise(self.cfg.bootstrap_iterations)

    #, running -----------------------------------------------------------

    def _running_step(self, kf: Keyframe) -> None:
        prev = self.keyframes[-2]
        if self.cfg.use_imu and kf.preintegrated is not None:
            prev_state = NavState(
                self.values.pose(prev.pose_key).R,
                self.values.pose(prev.pose_key).t,
                self.values.vector(prev.vel_key),
            )
            pred = kf.preintegrated.predict(prev_state, self._bias, self.cfg.gravity)
            self.values[kf.pose_key] = PoseVariable(Pose(pred.R, pred.p))
            self.values[kf.vel_key] = VectorVariable(pred.v)
            self.values[kf.bias_key] = BiasVariable(self._bias.vector())
        else:
            if not self._locate_by_pnp(kf):
                self._log("pnp failed", kf=kf.kf_id)
                self._set_pose(kf, self.values.pose(prev.pose_key) * self.T_body_cam)

        t0 = time.perf_counter()
        self._triangulate_new()
        self.timing.triangulate += time.perf_counter() - t0
        self._optimise(self.cfg.window_iterations)
        self._reject_outliers()

        if self.cfg.use_imu:
            self._bias = self.values.bias(kf.bias_key)
        self._velocity = (
            self.values.vector(kf.vel_key)
            if kf.vel_key in self.values
            else self._velocity
        )

    def _locate_by_pnp(self, kf: Keyframe) -> bool:
        pts, obs, ids = [], [], []
        for lid, xn in kf.features.items():
            key = f"l{lid}"
            if key in self.values:
                pts.append(self.values.vector(key))
                obs.append(xn)
                ids.append(lid)
        if len(pts) < 8:
            return False
        prev = self.keyframes[-2] if len(self.keyframes) > 1 else None
        guess = None
        if prev is not None and prev.pose_key in self.values:
            guess = self.values.pose(prev.pose_key) * self.T_body_cam
        res = G.solve_pnp(np.stack(pts), np.stack(obs), guess)
        if not res.ok:
            return False
        self._set_pose(kf, res.T_world_cam)
        return True

    #, map ---------------------------------------------------------------

    def _set_pose(self, kf: Keyframe, T_world_cam: Pose) -> None:
        T_wb = T_world_cam * self.T_body_cam.inverse()
        self.values[kf.pose_key] = PoseVariable(T_wb)
        if self.cfg.use_imu and kf.vel_key not in self.values:
            self.values[kf.vel_key] = VectorVariable(self._velocity.copy())
            self.values[kf.bias_key] = BiasVariable(self._bias.vector())

    def _camera_pose(self, kf: Keyframe) -> Pose:
        return self.values.pose(kf.pose_key) * self.T_body_cam

    def _triangulate_new(self) -> None:
        cfg = self.cfg
        window = self.keyframes
        poses = {k.kf_id: self._camera_pose(k) for k in window
                 if k.pose_key in self.values}
        counts = {}
        for kf in window:
            if kf.kf_id not in poses:
                continue
            for lid in kf.features:
                counts.setdefault(lid, []).append(kf.kf_id)

        # Prefer the longest tracks: they have the widest baseline and they are
        # the ones that will still be alive after the window slides.
        candidates = sorted(
            (lid for lid, ks in counts.items()
             if f"l{lid}" not in self.values and len(ks) >= cfg.min_track_length),
            key=lambda lid: -len(counts[lid]),
        )
        room = cfg.max_landmarks - sum(
            1 for lid in self.landmarks if f"l{lid}" in self.values
        )
        for lid in candidates[: max(room, 0)]:
            kf_ids = counts[lid]
            obs_poses = [poses[k] for k in kf_ids]
            obs_x = np.stack(
                [next(kf.features[lid] for kf in window if kf.kf_id == k)
                 for k in kf_ids]
            )
            p, ok = G.triangulate_dlt(obs_poses, obs_x, cfg.min_triangulation_angle)
            if not ok:
                continue
            p = G.refine_point(obs_poses, obs_x, p)
            depth = min(
                float((T.R.T @ (p - T.t))[2]) for T in obs_poses
            )
            if self.state == "RUNNING" and self.cfg.use_imu:
                lo, hi = 0.1, cfg.max_landmark_distance
            else:
                lo, hi = 0.02, cfg.bootstrap_max_depth_ratio
            if depth <= lo or depth > hi:
                continue
            self.values[f"l{lid}"] = PointVariable(p)
            self.landmarks[lid] = set(kf_ids)

    def _reject_outliers(self) -> None:
        cfg = self.cfg
        focal = self.camera.focal
        drop = []
        for lid in list(self.landmarks):
            key = f"l{lid}"
            if key not in self.values:
                continue
            p = self.values.vector(key)
            worst = 0.0
            seen = 0
            for kf in self.keyframes:
                if lid not in kf.features or kf.pose_key not in self.values:
                    continue
                T = self._camera_pose(kf)
                pc = T.R.T @ (p - T.t)
                if pc[2] <= 1e-6:
                    worst = np.inf
                    break
                err = focal * np.linalg.norm(pc[:2] / pc[2] - kf.features[lid])
                worst = max(worst, float(err))
                seen += 1
            if seen < 2 or worst > cfg.outlier_pixels:
                drop.append(lid)
        for lid in drop:
            self.values.pop(f"l{lid}", None)
            self.landmarks.pop(lid, None)

    #, optimisation ------------------------------------------------------

    def _build_graph(self, visual_only: bool = False) -> FactorGraph:
        cfg = self.cfg
        g = FactorGraph()
        loss = F.HuberLoss(cfg.huber_pixels)

        # One vectorised group for every landmark observation in the window.
        # See vio.backend.groups for why: the scalar factors are twelve times
        # slower here for identical output.
        group = ReprojectionGroup(
            self.camera, self.T_body_cam, cfg.pixel_sigma, loss
        )
        for kf in self.keyframes:
            if kf.pose_key not in self.values:
                continue
            for lid, xn in kf.features.items():
                key = f"l{lid}"
                if key in self.values:
                    group.add(kf.pose_key, key, xn)
        if len(group):
            g.add(group)

        if cfg.use_imu and not visual_only:
            for a, b in zip(self.keyframes[:-1], self.keyframes[1:]):
                if b.preintegrated is None or a.vel_key not in self.values:
                    continue
                if b.preintegrated.needs_repropagation(self.values.bias(a.bias_key)):
                    b.preintegrated = b.preintegrated.repropagate(
                        self.values.bias(a.bias_key)
                    )
                g.add(F.ImuFactor(
                    a.pose_key, a.vel_key, a.bias_key, b.pose_key, b.vel_key,
                    b.preintegrated, cfg.gravity,
                ))
                g.add(F.BiasRandomWalkFactor(
                    a.bias_key, b.bias_key, b.preintegrated.dt, self.imu_noise
                ))

        if self._prior is not None and all(
            k in self.values for k in self._prior.keys
        ):
            g.add(self._prior)
        elif self.keyframes:
            # No prior yet.  Anchor the oldest keyframe so the window is not
            # free to translate and rotate as a rigid body.  In the visual-only
            # configuration this is also what pins the arbitrary scale, since
            # the second keyframe's baseline is what the structure was built on.
            oldest = self.keyframes[0]
            if oldest.pose_key in self.values:
                sr, sp = cfg.fallback_prior_sigma
                info = np.diag(np.concatenate([np.full(3, 1.0 / sr), np.full(3, 1.0 / sp)]))
                g.add(F.PosePriorFactor(
                    oldest.pose_key, self.values.pose(oldest.pose_key), info
                ))
        return g

    def _optimise(self, iterations: int, visual_only: bool = False) -> None:
        if not self.values:
            return
        t0 = time.perf_counter()
        g = self._build_graph(visual_only)
        if not len(g):
            self.timing.optimise += time.perf_counter() - t0
            return
        res = optimise(
            g, self.values,
            options=SolverOptions(max_iterations=iterations, verbose=False),
        )
        self.values = res.values
        self.timing.optimise += time.perf_counter() - t0
        self.timing.solver_iterations += res.iterations

    #, window management -------------------------------------------------

    def _slide(self) -> None:
        cfg = self.cfg
        old = self.keyframes[0]
        # Its value now is its final one: nothing after this refines it.
        self._record_keyframe_pose(old)
        self.keyframes = self.keyframes[1:]

        drop = [old.pose_key]
        if cfg.use_imu and old.vel_key in self.values:
            drop += [old.vel_key, old.bias_key]

        # Landmarks that would be left with fewer than two observations go too.
        survivors = {k.kf_id for k in self.keyframes}
        for lid in list(self.landmarks):
            key = f"l{lid}"
            if key not in self.values:
                self.landmarks.pop(lid, None)
                continue
            remaining = sum(1 for kf in self.keyframes if lid in kf.features)
            if remaining < 2:
                drop.append(key)
        del survivors

        if cfg.marginalise and cfg.use_imu and self.state == "RUNNING":
            t0 = time.perf_counter()
            self._prior, self._prior_keys = self._marginalise_states(old, drop)
            self.timing.marginalise += time.perf_counter() - t0
        else:
            self._prior = None

        for key in drop:
            self.values.pop(key, None)
            if key.startswith("l"):
                self.landmarks.pop(int(key[1:]), None)

    def _marginalise_states(self, old, drop):
        """Marginalise the departing keyframe's inertial states into a prior.

        Only the *inertial* factors are marginalised, the IMU block, the bias
        random walk, and the previous prior, not the departing keyframe's
        landmark observations.

        That is a deliberate simplification and it is worth being straight
        about.  Marginalising the visual factors too is the textbook thing to
        do, and it is what VINS-Mono does with an inverse-depth
        parameterisation that lets most landmarks be dropped outright.  Doing
        it naively here means every landmark the old keyframe shared with the
        window enters the prior, which turns a 15 x 15 dense factor into a
        550 x 550 one: measured on the synthetic sequence that took the
        per-keyframe solve from 26 ms to 1042 ms, for an accuracy change
        smaller than the run-to-run spread.  The IMU block is the factor that
        actually carries scale and gravity across the window boundary, and that
        one is kept.

        ``scripts/ablate_backend.py`` compares this against no marginalisation
        at all.
        """
        first = self.keyframes[0] if self.keyframes else None
        if first is None or first.preintegrated is None:
            return None, []
        if old.vel_key not in self.values or first.vel_key not in self.values:
            return None, []

        g = FactorGraph()
        g.add(F.ImuFactor(
            old.pose_key, old.vel_key, old.bias_key,
            first.pose_key, first.vel_key, first.preintegrated, self.cfg.gravity,
        ))
        g.add(F.BiasRandomWalkFactor(
            old.bias_key, first.bias_key, first.preintegrated.dt, self.imu_noise
        ))
        if self._prior is not None and all(k in self.values for k in self._prior.keys):
            g.add(self._prior)
        state_drop = [k for k in drop if not k.startswith("l")]
        prior, _, keep = marginalise(g, self.values, state_drop)
        return prior, keep

    #, propagation and output -------------------------------------------

    def _gyro_rotation(self, stamp: float):
        if (
            self.imu is None
            or self._last_stamp is None
            or not self.cfg.tracker.use_gyro_prediction
        ):
            return None
        pre = preintegrate(
            self.imu, self._last_stamp, stamp, bias=self._bias, noise=self.imu_noise
        )
        R_bc = self.T_body_cam.R
        return R_bc.T @ pre.dR @ R_bc

    def _current_pose(self, stamp: float):
        """Body pose at ``stamp``, propagated from the last optimised keyframe."""
        if not self.keyframes or not self.values:
            return None
        kf = self.keyframes[-1]
        if kf.pose_key not in self.values:
            return None
        base = self.values.pose(kf.pose_key)
        dt = float(stamp) - kf.stamp
        if abs(dt) < 1e-9:
            return base.copy()
        if self.cfg.use_imu and self.state == "RUNNING" and self.imu is not None:
            pre = preintegrate(
                self.imu, kf.stamp, float(stamp),
                bias=self.values.bias(kf.bias_key), noise=self.imu_noise,
            )
            state = NavState(base.R, base.t, self.values.vector(kf.vel_key))
            pred = pre.predict(state, gravity=self.cfg.gravity)
            return Pose(pred.R, pred.p)
        # Visual-only: hold the last keyframe pose.  Interpolating towards a
        # pose that does not exist yet would be inventing information.
        return base.copy()

    #: Repeats of the same kind of event beyond this are counted rather than
    #: listed.  A run that fails PnP on every keyframe was otherwise writing a
    #: quarter of a megabyte of near-identical lines into its result file,
    #: which buries the three events anyone actually wants to read.
    MAX_EVENTS_PER_KIND = 12

    def _log(self, event: str, **fields) -> None:
        seen = self.event_counts.get(event, 0)
        self.event_counts[event] = seen + 1
        if seen < self.MAX_EVENTS_PER_KIND:
            record = {"event": event, "frame": self.timing.frames}
            record.update(fields)
            self.events.append(record)
        elif seen == self.MAX_EVENTS_PER_KIND:
            self.events.append({
                "event": event, "frame": self.timing.frames,
                "note": "further occurrences counted only, see event_counts",
            })

    def event_summary(self) -> dict:
        """How many times each kind of event happened over the whole run."""
        return dict(self.event_counts)
