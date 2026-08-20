"""A vocabulary tree over binary descriptors, built by hierarchical k-means.

This is the DBoW2 construction, written out.  DBoW2 itself is C++ and there is
no maintained Python binding, but more to the point the algorithm is short and
the interesting parts, how a binary centroid is defined, how the tree is
descended, how the inverted index is weighted, are exactly the parts worth
being able to answer questions about.

The pieces:

**Binary k-means.**  ORB descriptors are 256-bit strings and the natural
distance is Hamming.  The mean of a set of bit strings is not a bit string, so
the centroid is the *majority* bit in each position, which is the point
minimising total Hamming distance, the binary analogue of the median rather
than of the mean.

**The tree.**  Descriptors are clustered into ``branching`` groups, each group
is clustered again, and so on for ``depth`` levels.  The leaves are the words.
Assigning a descriptor to a word is then ``branching * depth`` comparisons
instead of comparing against every word, which is what makes a vocabulary of
tens of thousands of words queryable at frame rate.

**Weighting.**  Words are weighted by inverse document frequency, computed over
the training set: a word that fires in every image says nothing about which
image you are looking at.

Popcount is done with a 256-entry lookup table over bytes.  ``numpy`` gained
``bitwise_count`` in 2.0 and it is used when available, which is roughly twice
as fast; the table is kept as a fallback so the module works on older numpy.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two ``(N, D)`` and ``(M, D)`` uint8 sets."""
    a = np.atleast_2d(np.asarray(a, dtype=np.uint8))
    b = np.atleast_2d(np.asarray(b, dtype=np.uint8))
    xor = a[:, None, :] ^ b[None, :, :]
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(xor).sum(axis=2, dtype=np.int32)
    return _POPCOUNT[xor].sum(axis=2, dtype=np.int32)


def majority_bits(block: np.ndarray) -> np.ndarray:
    """The bitwise majority of a set of descriptors: the Hamming centroid."""
    bits = np.unpackbits(np.asarray(block, dtype=np.uint8), axis=1)
    return np.packbits((bits.mean(axis=0) >= 0.5).astype(np.uint8))


def _kmeans_binary(data: np.ndarray, k: int, rng, iterations: int = 12):
    """k-means with Hamming distance and majority-bit centroids.

    Seeded k-means++ style: the first centre is random, each subsequent one is
    the descriptor furthest from those already chosen.  Plain random seeding
    routinely produced clusters that were empty on the first assignment, which
    then had to be resurrected somehow, and the resurrection rule is exactly
    the sort of arbitrary detail that makes a vocabulary irreproducible.
    """
    n = len(data)
    if n <= k:
        return data.copy(), np.arange(n)

    centres = [data[rng.integers(n)]]
    dist = hamming(data, centres[0][None, :]).reshape(-1)
    for _ in range(1, k):
        centres.append(data[int(np.argmax(dist))])
        dist = np.minimum(dist, hamming(data, centres[-1][None, :]).reshape(-1))
    C = np.stack(centres)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iterations):
        d = hamming(data, C)
        new_labels = np.argmin(d, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(len(C)):
            member = data[labels == j]
            if len(member):
                C[j] = majority_bits(member)
    return C, labels


@dataclass
class _Node:
    centres: np.ndarray | None = None
    children: list = None
    #: Word index, set on leaves only.
    word: int = -1


class Vocabulary:
    """A trained vocabulary tree.

    Parameters
    ----------
    branching:
        Children per node.  DBoW2's default is 10.
    depth:
        Tree depth.  ``branching ** depth`` words at most.
    """

    def __init__(self, branching: int = 10, depth: int = 4):
        self.branching = int(branching)
        self.depth = int(depth)
        self.root: _Node | None = None
        self.n_words = 0
        #: Inverse document frequency per word, filled by :meth:`set_weights`.
        self.idf = np.zeros(0)
        self.descriptor_bytes = 32
        self.trained_on: list = []

    # -- training ----------------------------------------------------------
    def train(self, descriptors: np.ndarray, seed: int = 0,
              max_descriptors: int = 400_000) -> "Vocabulary":
        """Build the tree from a pile of descriptors."""
        data = np.asarray(descriptors, dtype=np.uint8)
        if data.ndim != 2:
            raise ValueError("descriptors must be (N, bytes)")
        self.descriptor_bytes = data.shape[1]
        rng = np.random.default_rng(seed)
        if len(data) > max_descriptors:
            data = data[rng.choice(len(data), max_descriptors, replace=False)]

        self.n_words = 0
        self.root = self._build(data, 0, rng)
        self.idf = np.ones(self.n_words)
        return self

    def _build(self, data: np.ndarray, level: int, rng) -> _Node:
        if level >= self.depth or len(data) <= self.branching:
            node = _Node(word=self.n_words)
            self.n_words += 1
            return node
        C, labels = _kmeans_binary(data, self.branching, rng)
        children = []
        for j in range(len(C)):
            member = data[labels == j]
            if len(member) == 0:
                node = _Node(word=self.n_words)
                self.n_words += 1
                children.append(node)
            else:
                children.append(self._build(member, level + 1, rng))
        return _Node(centres=C, children=children)

    def set_weights(self, word_document_counts: np.ndarray, n_documents: int) -> None:
        """Inverse document frequency from a corpus."""
        counts = np.asarray(word_document_counts, dtype=float)
        with np.errstate(divide="ignore"):
            idf = np.log(max(n_documents, 1) / np.maximum(counts, 1e-9))
        self.idf = np.where(counts > 0, idf, 0.0)

    # -- use ---------------------------------------------------------------
    def words(self, descriptors: np.ndarray) -> np.ndarray:
        """Word index for each descriptor."""
        data = np.atleast_2d(np.asarray(descriptors, dtype=np.uint8))
        out = np.empty(len(data), dtype=np.int32)
        for i, d in enumerate(data):
            node = self.root
            while node.word < 0:
                j = int(np.argmin(hamming(d[None, :], node.centres).reshape(-1)))
                node = node.children[j]
            out[i] = node.word
        return out

    def vector(self, descriptors: np.ndarray) -> dict:
        """TF-IDF bag-of-words vector, L1-normalised, as ``{word: weight}``.

        L1 rather than L2 because the similarity score below is DBoW2's L1
        score, and the two have to agree for the score to lie in ``[0, 1]``.
        """
        if len(descriptors) == 0:
            return {}
        w = self.words(descriptors)
        counts = np.bincount(w, minlength=self.n_words).astype(float)
        weights = counts * self.idf
        total = weights.sum()
        if total <= 0.0:
            return {}
        weights /= total
        nz = np.nonzero(weights)[0]
        return {int(k): float(weights[k]) for k in nz}

    @staticmethod
    def score(a: dict, b: dict) -> float:
        """DBoW2's L1 score: 1 for identical vectors, 0 for disjoint ones."""
        if not a or not b:
            return 0.0
        total = 0.0
        for word, va in a.items():
            vb = b.get(word)
            if vb is not None:
                total += abs(va) + abs(vb) - abs(va - vb)
        return float(0.5 * total)

    # -- persistence -------------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=4)
        return path

    @staticmethod
    def load(path) -> "Vocabulary":
        with open(path, "rb") as fh:
            return pickle.load(fh)
