"""Loop detection: bag-of-words retrieval followed by geometric verification.

Appearance alone is not evidence.  A bag-of-words score says two images contain
similar *stuff*, which in a machine hall full of identical girders it very
often does when the camera has never been near that place.  Every candidate
here therefore has to survive a geometric test, descriptor matching against
the candidate keyframe's own three-dimensional points, then PnP under RANSAC --
before it is allowed to become a constraint.  A false loop closure is far worse
than a missed one: a missed loop leaves the drift you already had, while a
false one folds the map onto itself and nothing downstream can recover.

Three filters, in increasing cost:

1. **Temporal exclusion.**  Keyframes within ``min_gap`` of the query are the
   ones the estimator already constrains directly and they always score
   highest.  They are removed before anything else happens.

2. **Relative scoring.**  An absolute bag-of-words threshold means different
   things in a textured room and a bare corridor.  Following DBoW2, candidate
   scores are normalised by the score against the query's own immediate
   predecessor, which is the best available estimate of "what a good match
   looks like right now".

3. **Geometric verification.**  Match descriptors, run PnP with RANSAC against
   the candidate's landmarks, require a minimum inlier count *and* a minimum
   inlier fraction.  The fraction matters: a hundred inliers out of two
   thousand matches is a coincidence, not a place.

Each verified loop is also required to repeat.  A single frame agreeing with a
distant keyframe is more likely to be an accident than a place revisit, so a
candidate must be confirmed on ``consistency`` consecutive queries against
nearby keyframes before it is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..frontend import geometry as G
from ..geometry import Pose
from ..geometry.camera import PinholeCamera
from .vocabulary import Vocabulary


@dataclass
class LoopConfig:
    #: Keyframes closer than this in index are never loop candidates.
    min_gap: int = 30
    #: Retain candidates scoring above this fraction of the reference score.
    relative_score: float = 0.35
    #: ...and above this absolute score, so a featureless reference cannot
    #: drag the bar to zero.
    absolute_score: float = 0.015
    #: Candidates considered for geometric verification, best first.
    max_candidates: int = 4
    #: ORB features detected per keyframe for retrieval.
    orb_features: int = 600
    #: Lowe ratio for descriptor matching.
    match_ratio: float = 0.75
    min_matches: int = 25
    min_inliers: int = 20
    min_inlier_ratio: float = 0.30
    pnp_threshold_norm: float = 4.0e-3
    #: A detection is treated as sitting on a landmark if it is within this
    #: many pixels of the landmark's projection.
    association_px: float = 3.0
    #: Consecutive confirmations required before a loop is emitted.
    consistency: int = 2
    #: Two closures may not be emitted closer together than this.
    cooldown: int = 10


@dataclass
class LoopRecord:
    """Everything retained about a keyframe for the purpose of finding it again."""

    kf_id: int
    stamp: float
    #: Body pose in the world at the time the record was made.
    pose: Pose
    bow: dict
    descriptors: np.ndarray
    keypoints: np.ndarray
    #: Descriptors at the positions of landmarks this keyframe observed.
    map_descriptors: np.ndarray = field(default_factory=lambda: np.zeros((0, 32), np.uint8))
    #: Those landmarks in *this keyframe's camera frame*, so a PnP against them
    #: returns the relative pose directly rather than a pose in a world frame
    #: that has drifted since.
    map_points_cam: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class LoopClosure:
    query_id: int
    match_id: int
    score: float
    n_matches: int
    n_inliers: int
    #: Body-frame transform: the query keyframe expressed in the matched one.
    relative_pose: Pose
    #: Position discrepancy against the current estimate, metres.  Reported so
    #: a closure that claims to fix eight metres of drift can be recognised as
    #: either a triumph or a blunder.
    correction: float = 0.0


class LoopDetector:
    """Bag-of-words place recognition with geometric verification."""

    def __init__(self, vocabulary: Vocabulary, camera: PinholeCamera,
                 T_body_cam: Pose, config: LoopConfig | None = None):
        self.vocab = vocabulary
        self.camera = camera
        self.T_body_cam = T_body_cam.copy()
        self.cfg = config or LoopConfig()
        self.records: dict = {}
        self.order: list = []
        self._orb = cv2.ORB_create(self.cfg.orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        #: word -> set of keyframe ids, the inverted index.
        self.index: dict = {}
        self._pending: dict = {}
        self._last_closure = -10 ** 9

    # -- population --------------------------------------------------------
    def describe(self, image: np.ndarray, landmark_uv=None, landmark_points_cam=None):
        """ORB features for retrieval and which of them have a 3-D position.

        The map descriptors are a *subset of the detections*, not descriptors
        computed at the landmark positions.  That distinction is the difference
        between the geometric verification working and not working and the
        first version got it wrong.  A descriptor computed at an arbitrary
        pixel, wherever a tracked landmark happens to project, describes a
        patch that no detector would have chosen, so it does not match the
        detections in the query frame, which are at corners.  Matching like
        with like means detecting normally and then asking which detections
        happen to sit on a landmark.
        """
        keypoints, descriptors = self._orb.detectAndCompute(image, None)
        if descriptors is None or not len(keypoints):
            return (np.zeros((0, 2)), np.zeros((0, 32), np.uint8),
                    np.zeros((0, 32), np.uint8), np.zeros((0, 3)))
        kp = np.array([k.pt for k in keypoints], dtype=float)

        map_desc = np.zeros((0, 32), np.uint8)
        map_pts = np.zeros((0, 3))
        if landmark_uv is not None and len(landmark_uv):
            uv = np.asarray(landmark_uv, dtype=float)
            points = np.asarray(landmark_points_cam, dtype=float)
            distance = np.linalg.norm(kp[:, None, :] - uv[None, :, :], axis=2)
            nearest = np.argmin(distance, axis=1)
            close = distance[np.arange(len(kp)), nearest] <= self.cfg.association_px
            if np.any(close):
                map_desc = descriptors[close]
                map_pts = points[nearest[close]]
        return kp, descriptors, map_desc, map_pts

    def add(self, kf_id: int, stamp: float, pose: Pose, image: np.ndarray,
            landmark_uv=None, landmark_points_cam=None) -> LoopRecord:
        kp, desc, map_desc, map_pts = self.describe(
            image, landmark_uv, landmark_points_cam
        )
        record = LoopRecord(
            kf_id=int(kf_id), stamp=float(stamp), pose=pose.copy(),
            bow=self.vocab.vector(desc), descriptors=desc, keypoints=kp,
            map_descriptors=map_desc, map_points_cam=map_pts,
        )
        self.records[record.kf_id] = record
        self.order.append(record.kf_id)
        for word in record.bow:
            self.index.setdefault(word, set()).add(record.kf_id)
        return record

    # -- retrieval ---------------------------------------------------------
    def candidates(self, kf_id: int):
        """Scored, temporally filtered candidates for a keyframe."""
        cfg = self.cfg
        query = self.records.get(kf_id)
        if query is None or not query.bow:
            return []

        # The inverted index turns "score against everything" into "score
        # against everything sharing at least one word", which on a EuRoC-sized
        # database is a factor of ten or so.
        touched = set()
        for word in query.bow:
            touched.update(self.index.get(word, ()))
        touched.discard(kf_id)

        reference = 0.0
        for other in self.order[::-1]:
            if other < kf_id and kf_id - other <= 2:
                reference = max(reference, self.vocab.score(query.bow, self.records[other].bow))
        if reference <= 0.0:
            reference = cfg.absolute_score / max(cfg.relative_score, 1e-9)

        out = []
        # Sorted so the candidate order does not depend on Python's per-process
        # string hash randomisation; see vio.backend.marginalisation for what
        # that costs when it leaks into the numerics.
        for other in sorted(touched):
            if abs(other - kf_id) < cfg.min_gap:
                continue
            s = self.vocab.score(query.bow, self.records[other].bow)
            if s >= cfg.absolute_score and s >= cfg.relative_score * reference:
                out.append((s, other))
        out.sort(reverse=True)
        return out[: cfg.max_candidates]

    def verify(self, kf_id: int, match_id: int):
        """Geometric verification of one candidate pair."""
        cfg = self.cfg
        q = self.records[kf_id]
        m = self.records[match_id]
        if len(m.map_descriptors) < 8 or len(q.descriptors) < 8:
            return None

        raw = self._matcher.knnMatch(q.descriptors, m.map_descriptors, k=2)
        pairs = [
            (a.queryIdx, a.trainIdx)
            for a, b in (p for p in raw if len(p) == 2)
            if a.distance < cfg.match_ratio * b.distance
        ]
        if len(pairs) < cfg.min_matches:
            return None

        qi = np.array([p[0] for p in pairs])
        mi = np.array([p[1] for p in pairs])
        bearings = self.camera.undistort(q.keypoints[qi])
        points = m.map_points_cam[mi]

        res = G.solve_pnp(points, bearings, None, cfg.pnp_threshold_norm,
                          cfg.min_inliers)
        n_in = int(res.inliers.sum())
        if not res.ok or n_in < cfg.min_inliers:
            return None
        if n_in / max(len(pairs), 1) < cfg.min_inlier_ratio:
            return None

        # solve_pnp returns the query *camera* pose in the frame the points
        # were given in, which is the matched keyframe's camera frame.
        T_mcam_qcam = res.T_world_cam
        T_mbody_qbody = self.T_body_cam * T_mcam_qcam * self.T_body_cam.inverse()
        return LoopClosure(
            query_id=kf_id, match_id=match_id, score=0.0,
            n_matches=len(pairs), n_inliers=n_in, relative_pose=T_mbody_qbody,
        )

    def detect(self, kf_id: int):
        """Full pipeline for one keyframe.  Returns a closure or ``None``."""
        cfg = self.cfg
        if kf_id - self._last_closure < cfg.cooldown:
            return None
        for score, match_id in self.candidates(kf_id):
            closure = self.verify(kf_id, match_id)
            if closure is None:
                continue
            closure.score = float(score)

            # Temporal consistency: the same *place* has to come back, though
            # not necessarily the same keyframe, so nearby ids count as the
            # same group.
            group = match_id // max(cfg.min_gap, 1)
            seen = self._pending.get(group, [])
            seen = [k for k in seen if kf_id - k <= 3] + [kf_id]
            self._pending[group] = seen
            if len(seen) < cfg.consistency:
                return None

            q = self.records[kf_id]
            m = self.records[match_id]
            predicted = m.pose.inverse() * q.pose
            closure.correction = float(
                np.linalg.norm(predicted.t - closure.relative_pose.t)
            )
            self._last_closure = kf_id
            self._pending.pop(group, None)
            return closure
        return None
