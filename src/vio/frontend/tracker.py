"""Feature tracking front end: KLT with track management.

What is borrowed and what is not, stated plainly because it is the honest part
of a "from scratch" claim.  The pyramidal Lucas-Kanade solver itself is
OpenCV's ``calcOpticalFlowPyrLK``, and so is Shi-Tomasi corner detection.
Rewriting either in Python would run two orders of magnitude slower and teach
nothing that is not in the original 1991 paper.  Everything around them is
here: the forward-backward consistency test, the spacing mask that stops every
corner piling onto one high-contrast object, the track lifecycle, the two-view
outlier rejection, the keyframe policy, and the gyroscope-predicted initial
guess.

That last one is worth a paragraph, and the paragraph is not the one it
started as.  KLT begins its search at the feature's previous pixel position,
which is a poor guess when the camera is rotating fast.  Because this is a
*visual-inertial* system there is a gyroscope available, so integrating it
across the frame interval predicts where each feature will land, up to
translation parallax.  Measured on EuRoC, that prediction does more than
halve the initial guess error, 11.6 px to 5.3 px on V1_02, but it buys **no
additional tracked features at all** at the native 20 fps, because the
displacement is already comfortably inside a 21x21 window at the top of a
four-level pyramid.  Decimating the stream to simulate faster motion shows
where it starts to pay:

===========  =========  ==================  ==============  ======
stride       eff. fps   tracked, no gyro    with gyro       gain
===========  =========  ==================  ==============  ======
1            20.0       151.2               151.3           +0.0%
2            10.0       115.2               117.7           +2.2%
3             6.7        93.7                99.1           +5.7%
5             4.0        70.7                78.1          +10.4%
8             2.5        53.3                59.8          +12.2%
===========  =========  ==================  ==============  ======

So it is off by default: on this dataset it costs about a quarter of the front
end's runtime and returns nothing.  It is kept, and kept tested, because the
conclusion is a property of EuRoC's frame rate rather than of the idea, and a
rolling-shutter phone camera at 5 fps would want it on.
``scripts/ablate.py --only frontend`` regenerates that table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.camera import PinholeCamera


@dataclass
class TrackerConfig:
    max_features: int = 200
    #: Minimum spacing between features, in pixels.  Enforced by a mask at
    #: detection time rather than by non-maximum suppression afterwards, so
    #: existing tracks always win against new detections.
    min_distance: float = 25.0
    quality_level: float = 0.01
    #: KLT window and pyramid depth.  21x21 over four levels covers about 160
    #: pixels of motion at the coarsest level on a 752x480 image.
    win_size: int = 21
    pyramid_levels: int = 3
    #: A track is dropped if the round trip back to the previous frame lands
    #: more than this far from where it started.
    fb_threshold: float = 0.5
    border: int = 8
    #: Two-view RANSAC threshold in pixels, for the fundamental-matrix check.
    ransac_threshold: float = 1.0
    ransac_confidence: float = 0.999
    #: Reject the two-view model, not the tracks, if it keeps less than this.
    min_ransac_inlier_ratio: float = 0.65
    #: See the module docstring: measured to be worth nothing at 20 fps and
    #: about ten per cent of tracking yield at 2.5 fps.
    use_gyro_prediction: bool = False
    #: Below this many surviving tracks the frame is treated as a tracking
    #: failure and reported, rather than quietly producing a bad pose.
    min_tracks: int = 20


@dataclass
class Track:
    """One feature followed across frames."""

    id: int
    #: Frame indices, ascending.
    frames: list = field(default_factory=list)
    #: Raw pixel positions, one per entry in ``frames``.
    uv: list = field(default_factory=list)
    #: Undistorted normalised coordinates, computed once per observation.
    xn: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def last_frame(self) -> int:
        return self.frames[-1]

    def observation(self, frame: int):
        try:
            i = self.frames.index(frame)
        except ValueError:
            return None
        return self.uv[i], self.xn[i]

    def parallax(self) -> float:
        """Normalised-coordinate spread between first and last observation.

        Used by the keyframe policy.  Measured in normalised units rather than
        pixels so the threshold means the same thing on any camera.
        """
        if self.length < 2:
            return 0.0
        return float(np.linalg.norm(np.asarray(self.xn[-1]) - np.asarray(self.xn[0])))


@dataclass
class FrameFeatures:
    """What the tracker produced for one image."""

    index: int
    stamp: float
    ids: np.ndarray
    uv: np.ndarray
    xn: np.ndarray
    n_new: int
    n_tracked: int
    n_rejected: int
    ok: bool = True

    def as_dict(self) -> dict:
        return {int(i): x for i, x in zip(self.ids, self.xn)}


class Tracker:
    """Stateful KLT tracker over a stream of images."""

    def __init__(self, camera: PinholeCamera, config: TrackerConfig | None = None):
        self.camera = camera
        self.cfg = config or TrackerConfig()
        self.tracks: dict = {}
        self._next_id = 0
        self._prev_image = None
        self._prev_ids = np.zeros(0, dtype=np.int64)
        self._prev_uv = np.zeros((0, 2), dtype=np.float32)
        self._prev_stamp = None
        self.frame_index = -1

    # -- main entry point --------------------------------------------------
    def process(
        self,
        image: np.ndarray,
        stamp: float,
        index: int | None = None,
        gyro_rotation: np.ndarray | None = None,
    ) -> FrameFeatures:
        """Track into ``image`` and top up with new detections.

        ``gyro_rotation`` is ``R_prev_cur`` in the *camera* frame, the rotation
        the camera underwent since the previous image.  Optional; without it
        the tracker falls back to a zero-motion prior.
        """
        cfg = self.cfg
        self.frame_index = self.frame_index + 1 if index is None else int(index)
        image = np.ascontiguousarray(image)

        ids = np.zeros(0, dtype=np.int64)
        uv = np.zeros((0, 2), dtype=np.float32)
        n_rejected = 0

        if self._prev_image is not None and len(self._prev_uv) > 0:
            ids, uv, n_rejected = self._track(image, gyro_rotation)

        n_tracked = len(ids)
        new_uv = self._detect(image, uv)
        n_new = len(new_uv)
        if n_new:
            new_ids = np.arange(self._next_id, self._next_id + n_new, dtype=np.int64)
            self._next_id += n_new
            ids = np.concatenate([ids, new_ids])
            uv = np.concatenate([uv, new_uv.astype(np.float32)])

        xn = (
            self.camera.undistort(uv.astype(np.float64))
            if len(uv)
            else np.zeros((0, 2))
        )
        for tid, p, q in zip(ids, uv, xn):
            tid = int(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(tid)
            tr.frames.append(self.frame_index)
            tr.uv.append(np.asarray(p, dtype=float))
            tr.xn.append(np.asarray(q, dtype=float))

        self._prev_image = image
        self._prev_ids = ids
        self._prev_uv = uv.astype(np.float32)
        self._prev_stamp = stamp

        return FrameFeatures(
            index=self.frame_index,
            stamp=float(stamp),
            ids=ids,
            uv=uv.astype(np.float64),
            xn=xn,
            n_new=n_new,
            n_tracked=n_tracked,
            n_rejected=n_rejected,
            ok=len(ids) >= cfg.min_tracks,
        )

    # -- internals ---------------------------------------------------------
    def _predict(self, gyro_rotation) -> np.ndarray:
        """Where the previous features should land, given the gyro rotation."""
        prev = self._prev_uv.astype(np.float64)
        if gyro_rotation is None or not self.cfg.use_gyro_prediction or len(prev) == 0:
            return prev.astype(np.float32)
        xn = self.camera.undistort(prev)
        bearings = np.concatenate([xn, np.ones((len(xn), 1))], axis=1)
        # R_prev_cur maps a direction in the previous camera frame to the
        # current one; a pure rotation moves the bearing and nothing else.
        rotated = bearings @ np.asarray(gyro_rotation, dtype=float)
        z = rotated[:, 2]
        ok = z > 1e-6
        out = prev.copy()
        if np.any(ok):
            out[ok] = self.camera.project(rotated[ok, :2] / z[ok, None])
        return out.astype(np.float32)

    def _track(self, image, gyro_rotation):
        cfg = self.cfg
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        win = (cfg.win_size, cfg.win_size)
        guess = self._predict(gyro_rotation)

        fwd, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_image, image, self._prev_uv, guess.copy(),
            winSize=win, maxLevel=cfg.pyramid_levels, criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        back, status_b, _ = cv2.calcOpticalFlowPyrLK(
            image, self._prev_image, fwd, self._prev_uv.copy(),
            winSize=win, maxLevel=cfg.pyramid_levels, criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )

        status = status.reshape(-1).astype(bool) & status_b.reshape(-1).astype(bool)
        fb = np.linalg.norm(back - self._prev_uv, axis=1)
        keep = status & (fb < cfg.fb_threshold)

        h, w = image.shape[:2]
        b = cfg.border
        keep &= (
            (fwd[:, 0] >= b) & (fwd[:, 0] < w - b)
            & (fwd[:, 1] >= b) & (fwd[:, 1] < h - b)
        )
        n_before = int(keep.sum())

        ids = self._prev_ids[keep]
        uv = fwd[keep]
        prev_uv = self._prev_uv[keep]

        # Two-view geometric check.  A fundamental matrix rather than an
        # essential one, because at this point the pair may be nearly pure
        # rotation and the essential matrix is degenerate there; the
        # fundamental matrix still removes independently moving points and
        # gross mismatches, which is all this stage is for.
        #
        # The result is only accepted if it keeps most of its input.  A
        # fundamental matrix is itself degenerate when the scene is planar or
        # the baseline is tiny, and in that case RANSAC does not fail loudly --
        # it returns a confident, arbitrary model and throws away most of the
        # correspondences.  Measured on a deliberately planar synthetic scene
        # it discarded 34 tracks per frame while the forward-backward test was
        # discarding three, and mean track length collapsed from thirty frames
        # to five.  When the model rejects more than this fraction, the model
        # is the thing that is wrong, not the tracks.
        if len(uv) >= 12:
            _, mask = cv2.findFundamentalMat(
                prev_uv, uv, cv2.FM_RANSAC,
                cfg.ransac_threshold, cfg.ransac_confidence,
            )
            if mask is not None:
                inl = mask.reshape(-1).astype(bool)
                ratio = inl.sum() / max(len(inl), 1)
                if inl.sum() >= cfg.min_tracks and ratio >= cfg.min_ransac_inlier_ratio:
                    ids, uv = ids[inl], uv[inl]

        return ids, uv, n_before - len(ids)

    def _detect(self, image: np.ndarray, existing: np.ndarray) -> np.ndarray:
        """New corners, at least ``min_distance`` from anything already tracked."""
        cfg = self.cfg
        want = cfg.max_features - len(existing)
        if want <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        r = int(cfg.min_distance)
        for x, y in existing:
            cv2.circle(mask, (int(round(float(x))), int(round(float(y)))), r, 0, -1)
        mask[: cfg.border, :] = 0
        mask[-cfg.border:, :] = 0
        mask[:, : cfg.border] = 0
        mask[:, -cfg.border:] = 0

        pts = cv2.goodFeaturesToTrack(
            image, want, cfg.quality_level, cfg.min_distance, mask=mask
        )
        if pts is None:
            return np.zeros((0, 2), dtype=np.float32)
        return pts.reshape(-1, 2)

    # -- housekeeping ------------------------------------------------------
    def prune(self, before_frame: int) -> int:
        """Drop tracks last seen before ``before_frame``.

        Without this the dictionary grows without bound: EuRoC MH_01 produces
        about 90 000 tracks over 3600 frames, and holding every dead one costs
        several hundred megabytes by the end.
        """
        dead = [k for k, t in self.tracks.items() if t.last_frame < before_frame]
        for k in dead:
            del self.tracks[k]
        return len(dead)

    def active(self, frame: int) -> dict:
        return {
            k: t for k, t in self.tracks.items() if t.frames and t.last_frame == frame
        }
