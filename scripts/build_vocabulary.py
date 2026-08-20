#!/usr/bin/env python
"""Train a bag-of-words vocabulary from ORB descriptors.

Which sequences the vocabulary is trained on matters, and the default here is
deliberate: **Machine Hall 01 and 02 only**.  Loop closure is then evaluated on
Vicon Room, which is a different room, and on Machine Hall 03 to 05, which are
different runs.  Training the vocabulary on the sequence you then report
loop-closure recall for is a way to make a place recogniser look better than it
is, and the split is recorded in the vocabulary file so a reader can check.

    python scripts/build_vocabulary.py --sequences MH_01_easy MH_02_easy
    python scripts/build_vocabulary.py --sim          # for CI, no dataset needed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from vio.io import euroc  # noqa: E402
from vio.loop_closure.vocabulary import Vocabulary  # noqa: E402

DEFAULT_DATA = Path(
    os.environ.get("EUROC_ROOT", Path.home() / "datasets" / "euroc" / "sequences")
)


def descriptors_from_sequence(root, stride: int, features: int, quiet: bool):
    orb = cv2.ORB_create(features)
    seq = euroc.load(root, stereo=False)
    out = []
    for i in range(0, len(seq.cam0), stride):
        _, desc = orb.detectAndCompute(seq.cam0.image(i), None)
        if desc is not None:
            out.append(desc)
        if not quiet and i % (stride * 100) == 0:
            print(f"    frame {i}/{len(seq.cam0)}", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 32), np.uint8)


def descriptors_from_simulation(features: int, quiet: bool):
    from vio.sim.render import SceneRenderer
    from vio.sim.simulator import SimConfig, simulate

    orb = cv2.ORB_create(features)
    out = []
    for seed in (101, 202):
        seq = simulate(SimConfig(duration=12.0, seed=seed))
        renderer = SceneRenderer(seq, noise_sigma=2.0)
        for i in range(0, len(seq.frames), 3):
            _, desc = orb.detectAndCompute(renderer.render(i), None)
            if desc is not None:
                out.append(desc)
        if not quiet:
            print(f"    simulated seed {seed} done", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 32), np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="*", default=["MH_01_easy", "MH_02_easy"])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(ROOT / "models" / "vocabulary.pkl"))
    ap.add_argument("--branching", type=int, default=10)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--features", type=int, default=500)
    ap.add_argument("--max-descriptors", type=int, default=300_000)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.sim:
        data = descriptors_from_simulation(args.features, args.quiet)
        trained_on = ["simulated"]
    else:
        blocks = []
        for name in args.sequences:
            root = Path(args.data) / name
            if not root.is_dir():
                print(f"missing sequence {root}")
                return 2
            if not args.quiet:
                print(f"  {name}", flush=True)
            blocks.append(descriptors_from_sequence(
                root, args.stride, args.features, args.quiet
            ))
        data = np.concatenate(blocks)
        trained_on = list(args.sequences)

    print(f"{len(data)} descriptors in {time.perf_counter() - t0:.0f} s", flush=True)
    vocab = Vocabulary(args.branching, args.depth)
    t1 = time.perf_counter()
    vocab.train(data, seed=0, max_descriptors=args.max_descriptors)
    vocab.trained_on = trained_on
    print(f"{vocab.n_words} words in {time.perf_counter() - t1:.0f} s", flush=True)

    # Inverse document frequency over the training images themselves.
    counts = np.zeros(vocab.n_words)
    n_docs = 0
    chunk = max(args.features, 1)
    for start in range(0, len(data), chunk):
        words = vocab.words(data[start:start + chunk])
        counts[np.unique(words)] += 1
        n_docs += 1
    vocab.set_weights(counts, n_docs)

    out = Path(args.out)
    vocab.save(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"trained on {', '.join(trained_on)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
