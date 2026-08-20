"""Bag-of-words vocabulary, place recognition and the pose graph."""

from __future__ import annotations

import numpy as np
import pytest

from vio.backend import factors as F
from vio.geometry import Pose, se3, so3
from vio.loop_closure import pose_graph
from vio.loop_closure.detector import LoopConfig, LoopDetector
from vio.loop_closure.vocabulary import Vocabulary, hamming, majority_bits
from vio.types import GRAVITY


# ---------------------------------------------------------------- vocabulary


def test_hamming_counts_differing_bits():
    a = np.array([[0b00000000, 0b11111111]], dtype=np.uint8)
    b = np.array([[0b11111111, 0b11111111]], dtype=np.uint8)
    assert hamming(a, b)[0, 0] == 8
    assert hamming(a, a)[0, 0] == 0


def test_hamming_is_symmetric_and_zero_on_the_diagonal():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 256, (20, 32), dtype=np.uint8)
    D = hamming(d, d)
    assert np.array_equal(D, D.T)
    assert np.all(np.diag(D) == 0)


def test_majority_bits_minimises_total_hamming_distance():
    """The binary analogue of the median, and the reason it is the centroid."""
    rng = np.random.default_rng(1)
    block = rng.integers(0, 256, (25, 4), dtype=np.uint8)
    centre = majority_bits(block)
    best = int(hamming(block, centre[None, :]).sum())
    for candidate in block:
        assert int(hamming(block, candidate[None, :]).sum()) >= best


def test_vocabulary_separates_distinct_clusters():
    """Descriptors drawn from well-separated seeds must land in different words."""
    rng = np.random.default_rng(2)
    seeds = rng.integers(0, 256, (8, 32), dtype=np.uint8)
    data, labels = [], []
    for k, seed in enumerate(seeds):
        bits = np.unpackbits(np.repeat(seed[None, :], 60, axis=0), axis=1)
        flip = rng.random(bits.shape) < 0.02
        data.append(np.packbits(np.bitwise_xor(bits, flip.astype(np.uint8)), axis=1))
        labels += [k] * 60
    data = np.concatenate(data)
    labels = np.array(labels)

    vocab = Vocabulary(branching=8, depth=2).train(data, seed=0)
    words = vocab.words(data)
    # Each true cluster should be dominated by one word.
    for k in range(8):
        counts = np.bincount(words[labels == k])
        # Not 1.0: with two per cent of the bits flipped a cluster spreads over
        # a couple of neighbouring leaves, which is what a vocabulary tree does
        # and not a defect.
        assert counts.max() / counts.sum() > 0.8


def test_bow_score_is_one_for_identical_and_zero_for_disjoint():
    rng = np.random.default_rng(3)
    data = rng.integers(0, 256, (600, 32), dtype=np.uint8)
    vocab = Vocabulary(branching=6, depth=2).train(data, seed=0)
    vocab.set_weights(np.ones(vocab.n_words), 10)
    v = vocab.vector(data[:100])
    assert Vocabulary.score(v, v) == pytest.approx(1.0, abs=1e-9)
    assert Vocabulary.score(v, {}) == 0.0
    assert Vocabulary.score({}, {}) == 0.0


def test_bow_vectors_are_l1_normalised():
    rng = np.random.default_rng(4)
    data = rng.integers(0, 256, (400, 32), dtype=np.uint8)
    vocab = Vocabulary(branching=5, depth=2).train(data, seed=0)
    vocab.set_weights(np.full(vocab.n_words, 3.0), 40)
    v = vocab.vector(data[:80])
    assert sum(v.values()) == pytest.approx(1.0, abs=1e-9)


def test_idf_silences_a_word_that_fires_everywhere():
    vocab = Vocabulary(branching=2, depth=1)
    vocab.n_words = 3
    vocab.set_weights(np.array([100.0, 5.0, 0.0]), 100)
    assert vocab.idf[0] == pytest.approx(0.0)
    assert vocab.idf[1] > 1.0
    assert vocab.idf[2] == 0.0


def test_vocabulary_round_trips_through_disk(tmp_path):
    rng = np.random.default_rng(5)
    data = rng.integers(0, 256, (500, 32), dtype=np.uint8)
    vocab = Vocabulary(branching=5, depth=2).train(data, seed=0)
    vocab.set_weights(np.ones(vocab.n_words) * 2.0, 20)
    vocab.trained_on = ["a_sequence"]
    path = vocab.save(tmp_path / "v.pkl")
    back = Vocabulary.load(path)
    assert back.n_words == vocab.n_words
    assert back.trained_on == ["a_sequence"]
    assert np.array_equal(back.words(data[:50]), vocab.words(data[:50]))


def test_training_is_deterministic():
    rng = np.random.default_rng(6)
    data = rng.integers(0, 256, (400, 32), dtype=np.uint8)
    a = Vocabulary(4, 2).train(data, seed=3)
    b = Vocabulary(4, 2).train(data, seed=3)
    assert a.n_words == b.n_words
    assert np.array_equal(a.words(data), b.words(data))


# ---------------------------------------------------------------- detection


@pytest.fixture(scope="module")
def loop_scene(rendered_sequence):
    """A vocabulary trained on the same imagery, and a detector on top of it."""
    import cv2

    seq, images = rendered_sequence
    orb = cv2.ORB_create(400)
    blocks = []
    for image in images[::4]:
        _, desc = orb.detectAndCompute(image, None)
        if desc is not None:
            blocks.append(desc)
    data = np.concatenate(blocks)
    vocab = Vocabulary(branching=8, depth=3).train(data, seed=0)
    counts = np.zeros(vocab.n_words)
    for start in range(0, len(data), 400):
        counts[np.unique(vocab.words(data[start:start + 400]))] += 1
    vocab.set_weights(counts, max(len(data) // 400, 1))
    return seq, images, vocab


def test_the_same_image_scores_near_one(loop_scene):
    seq, images, vocab = loop_scene
    detector = LoopDetector(vocab, seq.config.camera, seq.config.T_body_cam)
    _, desc, _, _ = detector.describe(images[10])
    v = vocab.vector(desc)
    assert Vocabulary.score(v, v) == pytest.approx(1.0, abs=1e-9)


def test_nearby_images_score_higher_than_distant_ones(loop_scene):
    """The property the whole retrieval stage depends on."""
    seq, images, vocab = loop_scene
    detector = LoopDetector(vocab, seq.config.camera, seq.config.T_body_cam)
    vectors = [vocab.vector(detector.describe(img)[1]) for img in images[::5]]
    near, far = [], []
    for i in range(len(vectors) - 12):
        near.append(Vocabulary.score(vectors[i], vectors[i + 1]))
        far.append(Vocabulary.score(vectors[i], vectors[i + 12]))
    assert np.median(near) > np.median(far)


def test_temporal_exclusion_removes_recent_keyframes(loop_scene):
    seq, images, vocab = loop_scene
    detector = LoopDetector(vocab, seq.config.camera, seq.config.T_body_cam,
                            LoopConfig(min_gap=25))
    for i, image in enumerate(images[:60]):
        detector.add(i, float(i) * 0.05, seq.states[i].pose, image)
    for score, other in detector.candidates(59):
        assert abs(59 - other) >= 25


def test_verification_rejects_a_pair_with_no_map_points(loop_scene):
    seq, images, vocab = loop_scene
    detector = LoopDetector(vocab, seq.config.camera, seq.config.T_body_cam)
    detector.add(0, 0.0, seq.states[0].pose, images[0])
    detector.add(50, 2.5, seq.states[50].pose, images[50])
    assert detector.verify(50, 0) is None


def test_verification_recovers_the_relative_pose_from_correspondences():
    """The geometric half of verification, with the appearance half removed.

    Records are built by hand so the descriptors match perfectly and the
    landmarks are exact.  What is under test is everything after matching: the
    PnP, and the conversion from a camera-frame PnP result to the body-frame
    relative pose the pose graph consumes.  That conversion is two inversions
    and a conjugation and it is the easiest thing here to get backwards.

    The appearance half cannot be tested on synthetic imagery, see
    :mod:`vio.sim.render` for the measurement, so it is tested on EuRoC by
    ``scripts/run_loop_closure.py``.
    """
    from vio.loop_closure.detector import LoopRecord

    rng = np.random.default_rng(11)
    from vio.geometry.camera import PinholeCamera

    camera = PinholeCamera(752, 480, 458.654, 457.296, 367.215, 248.375,
                           [-0.283, 0.0739, 0.0002, 1.76e-5], "radtan")
    T_bc = Pose(se3.exp(np.array([0.1, -0.2, 1.5, 0.0, 0.0, 0.0])).R,
                np.array([0.02, -0.06, 0.01]))
    detector = LoopDetector(vocabulary_stub(), camera, T_bc,
                            LoopConfig(min_gap=1, consistency=1, min_matches=20))

    # A known body-frame motion between the two keyframes.
    T_body_rel = Pose(so3.exp(np.array([0.05, -0.03, 0.35])),
                      np.array([0.9, -0.4, 0.1]))
    T_cam_rel = T_bc.inverse() * T_body_rel * T_bc

    n = 120
    points_anchor = np.stack([
        rng.uniform(-1.6, 1.6, n), rng.uniform(-1.2, 1.2, n),
        rng.uniform(2.5, 9.0, n),
    ], axis=1)
    points_query = T_cam_rel.inverse().act(points_anchor)
    uv_query, ok = camera.project_points(points_query)
    keep = ok
    assert keep.sum() > 60

    descriptors = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    anchor = LoopRecord(
        kf_id=0, stamp=0.0, pose=Pose.identity(), bow={},
        descriptors=descriptors, keypoints=np.zeros((n, 2)),
        map_descriptors=descriptors, map_points_cam=points_anchor,
    )
    query = LoopRecord(
        kf_id=40, stamp=2.0, pose=Pose.identity(), bow={},
        descriptors=descriptors[keep],
        keypoints=uv_query[keep] + rng.normal(0, 0.3, (int(keep.sum()), 2)),
        map_descriptors=np.zeros((0, 32), np.uint8),
        map_points_cam=np.zeros((0, 3)),
    )
    detector.records = {0: anchor, 40: query}
    detector.order = [0, 40]

    closure = detector.verify(40, 0)
    assert closure is not None
    assert closure.n_inliers > 50
    assert np.linalg.norm(closure.relative_pose.t - T_body_rel.t) < 0.03
    assert np.degrees(
        so3.angle_between(closure.relative_pose.R, T_body_rel.R)
    ) < 0.5


def vocabulary_stub():
    """A vocabulary that answers nothing; the geometric test does not use it."""
    vocab = Vocabulary(branching=2, depth=1)
    vocab.n_words = 1
    vocab.idf = np.ones(1)
    return vocab


def test_retrieval_plumbing_runs_on_synthetic_imagery(loop_scene):
    """Indexing, scoring and temporal exclusion, which do work on these images.

    The appearance *quality* does not; see :mod:`vio.sim.render`.  What this
    asserts is that the inverted index is populated, that querying it returns
    only temporally distant keyframes, and that nothing raises.
    """
    seq, images, vocab = loop_scene
    detector = LoopDetector(vocab, seq.config.camera, seq.config.T_body_cam,
                            LoopConfig(min_gap=20))
    for i, image in enumerate(images[:80]):
        detector.add(i, float(i) * 0.05, seq.states[i].pose, image)
    assert len(detector.index) > 50
    assert all(len(v) > 0 for v in detector.index.values())
    candidates = detector.candidates(79)
    assert all(abs(79 - other) >= 20 for _, other in candidates)
    assert detector.detect(79) is None or True   # must not raise either way


# ---------------------------------------------------------------- pose graph


def _drifting_chain(n=60, drift=0.02):
    """A straight-ish loop whose estimate accumulates a yaw error."""
    truth, estimate = [], []
    angle = np.linspace(0, 2 * np.pi, n)
    for i, a in enumerate(angle):
        truth.append(Pose(so3.exp(np.array([0.0, 0.0, a])),
                          np.array([3 * np.cos(a), 3 * np.sin(a), 1.0])))
        bad = a + drift * i / n * a
        estimate.append(Pose(so3.exp(np.array([0.0, 0.0, bad])),
                             np.array([3 * np.cos(bad), 3 * np.sin(bad), 1.0])))
    return truth, estimate


def test_pose_graph_reduces_the_gap_at_a_loop_closure():
    truth, estimate = _drifting_chain()
    ids = list(range(len(truth)))
    poses = dict(zip(ids, estimate))
    closure = type("C", (), {})()
    closure.query_id = len(truth) - 1
    closure.match_id = 0
    closure.relative_pose = truth[0].inverse() * truth[-1]

    before = np.linalg.norm(
        (estimate[0].inverse() * estimate[-1]).t - closure.relative_pose.t
    )
    result = pose_graph.optimise_graph(ids, poses, [closure])
    after_rel = result.poses[0].inverse() * result.poses[ids[-1]]
    after = np.linalg.norm(after_rel.t - closure.relative_pose.t)
    assert after < before / 3.0
    assert result.final_cost < result.initial_cost
    assert result.max_displacement() > 0.0


def test_pose_graph_without_closures_is_a_no_op():
    truth, estimate = _drifting_chain(n=10)
    ids = list(range(10))
    result = pose_graph.optimise_graph(ids, dict(zip(ids, estimate)), [])
    for k in ids:
        assert np.allclose(result.poses[k].matrix(), estimate[k].matrix())


def test_gravity_lock_stops_the_map_tilting():
    """A six-degree-of-freedom pose graph will tilt the world to close a loop.

    Roll and pitch are observable from gravity and do not drift, so trading
    them for a better loop fit is trading a good measurement for a bad one.
    """
    truth, estimate = _drifting_chain(drift=0.06)
    ids = list(range(len(truth)))
    poses = dict(zip(ids, estimate))
    closure = type("C", (), {})()
    closure.query_id = len(truth) - 1
    closure.match_id = 0
    # A loop measurement that is itself slightly tilted, which is what an
    # imperfect PnP produces.
    tilted = truth[0].inverse() * truth[-1]
    closure.relative_pose = Pose(
        so3.project(tilted.R @ so3.exp(np.array([0.05, -0.04, 0.0]))), tilted.t
    )

    g_hat = GRAVITY / np.linalg.norm(GRAVITY)

    def worst_tilt(result):
        out = 0.0
        for k in ids:
            cos = float((result.poses[k].R.T @ g_hat) @ (estimate[k].R.T @ g_hat))
            out = max(out, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return out

    locked = pose_graph.optimise_graph(
        ids, poses, [closure], pose_graph.PoseGraphConfig(lock_gravity=True)
    )
    free = pose_graph.optimise_graph(
        ids, poses, [closure], pose_graph.PoseGraphConfig(lock_gravity=False)
    )
    assert worst_tilt(locked) < worst_tilt(free)
    assert worst_tilt(locked) < 1.5


def test_correction_is_applied_to_every_frame():
    truth, estimate = _drifting_chain(n=20)
    ids = list(range(20))
    stamps = [0.1 * i for i in ids]
    original = dict(zip(ids, estimate))
    corrected = {k: Pose(v.R, v.t + np.array([0.5, 0.0, 0.0]))
                 for k, v in original.items()}

    frame_times = np.linspace(0.0, 1.9, 60)
    frame_poses = [estimate[min(int(t / 0.1), 19)] for t in frame_times]
    out = pose_graph.apply_correction(
        frame_times, frame_poses, stamps, ids, original, corrected
    )
    assert len(out) == len(frame_poses)
    shifts = [np.linalg.norm(a.t - b.t) for a, b in zip(out, frame_poses)]
    assert np.allclose(shifts, 0.5, atol=1e-9)


def test_correction_handles_missing_poses():
    truth, estimate = _drifting_chain(n=5)
    ids = list(range(5))
    stamps = [0.1 * i for i in ids]
    original = dict(zip(ids, estimate))
    out = pose_graph.apply_correction(
        [0.0, 0.15], [None, estimate[1]], stamps, ids, original, original
    )
    assert out[0] is None
    assert out[1] is not None


def test_only_the_loop_edge_is_robustified():
    truth, estimate = _drifting_chain(n=12)
    ids = list(range(12))
    closure = type("C", (), {})()
    closure.query_id, closure.match_id = 11, 0
    closure.relative_pose = truth[0].inverse() * truth[-1]
    graph, _ = pose_graph.build(ids, dict(zip(ids, estimate)), [closure])
    losses = [
        type(f.loss).__name__ for f in graph.factors
        if isinstance(f, F.RelativePoseFactor)
    ]
    # Huber by default, on the loop edge only.  See the module docstring for
    # why a redescending kernel is the wrong choice here.
    assert losses.count("HuberLoss") == 1
    assert losses.count("TrivialLoss") == len(ids) - 1
