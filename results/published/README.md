# Published results

These are the result records behind every number in the README and in
[docs/results.md](../../docs/results.md). Each one carries the metrics, the
timing breakdown, the configuration the run used, the events the estimator
logged, and the commit it was produced from.

`make tables` regenerates every table in the documents from these files, and
`make docs-check` fails the build if a document has drifted from them.

## What is here

| file | what it is |
| --- | --- |
| `<sequence>_vio.json` | visual-inertial run: metrics under both alignments, RPE at 1, 5 and 10 m, drift, per-stage timing |
| `<sequence>_vo.json` | the same sequence with the IMU switched off |
| `<sequence>_vio_loop.json` | loop closure: detections, precision, episode recall, ATE before and after the pose graph |
| `<sequence>_calibration.json` | camera-IMU extrinsic rotation and time offset, against the published values |
| `ablations.json` | every ablation the documents quote |

## What is not here

**The trajectories and the maps.** A `.tum` file for one sequence is a few
hundred kilobytes, its ground-truth companion is a few megabytes, and the
sparse map beside it is another few hundred kilobytes; the whole set is about
24 MB of text that two commands regenerate. Committing it would
make a clone heavier than the code by an order of magnitude for no information
a reader cannot produce.

To regenerate them:

```bash
python scripts/fetch_euroc.py     # about 19 GB, one time
make experiments                  # writes results/*.tum alongside these records
make evo-check                    # re-scores every file with evo
```

The metrics in these records were computed *from* those trajectory files
rather than from the estimator's in-memory state, so recomputing them with
`evaluation/evaluate.py` or with `evo` returns the same numbers rather than
merely similar ones.

## A note on the timing figures

`workers` in each record says how many runs shared the machine. Accuracy does
not care, because the estimator is deterministic. Milliseconds per frame cares
a great deal: an eight-way parallel sweep reads about seventy per cent slower
than the same code alone. The timing table only uses records with
`workers: 1`, which come from `make timing`.
