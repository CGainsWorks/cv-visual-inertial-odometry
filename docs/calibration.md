# Camera-IMU calibration

What is estimated, what is not and why the distinction is about observability
rather than effort.

Run it with:

```bash
make calibrate SEQUENCE=V1_01_easy
python scripts/calibrate.py MH_03_medium --inject-offset 0.020
```

---

## Rotation: observable from rotation alone

The camera and the IMU are bolted to the same rigid body, so between any two
instants they turn through related amounts:

```
dR_body = R_bc  dR_cam  R_bc^T
```

Taking logarithms gives `theta_body = R_bc theta_cam` for every pair, which is
an orthogonal Procrustes problem: stack the two sets of rotation vectors and
take an SVD.  This is the classical hand-eye result specialised to the case
where only rotation is wanted and it is unusually well behaved: it needs no
scale, no translation, no structure and no landmarks, only a gyroscope and a
visual rotation estimate.

Two details in
[`vio/calibration/extrinsic.py`](../src/vio/calibration/extrinsic.py) matter:

* **Small rotations are excluded, not averaged in.**  Below about 0.02 rad the
  logarithm is dominated by noise and Procrustes weights each pair by its
  magnitude squared, so a noise-only pair contributes pure noise at full
  weight.  A test feeds in forty good pairs and two hundred junk ones and
  checks that the junk is dropped and the answer is unaffected.
* **The determinant is corrected.**  Without it a noisy set can fit better with
  a reflection, which is not a rotation.

### The inputs have to be independent of the answer

The first version of `scripts/calibrate.py` took ground-truth body poses,
multiplied them by the *published* extrinsic to get camera rotations and then
recovered that extrinsic to within 1e-14 radians.  That is not a calibration,
it is an identity: the answer had been used to construct the input and the
1e-14 was the round-off in a matrix product.

What runs now is the real procedure.  Camera rotations come from **visual
odometry over the images** with the IMU switched off, so nothing about the
extrinsic or the IMU enters them.  Body rotations come from **integrating the
raw gyroscope** between the same instants, so nothing about the camera enters
them.  The published extrinsic is loaded only at the end, to score the result.

That is slower, because it means running the estimator.  The answer is worse
too, because it now carries the visual odometry's own rotation error.  Both are the
correct outcome: the number means something now.

The result on real sequences is in [results.md](results.md) §9, compared
against EuRoC's published extrinsic, which the routine never sees.

## Time offset: observable from correlation

Cameras and IMUs are rarely on the same clock.  A few milliseconds does not
sound like much until you notice that at 200 degrees per second it injects a
degree of attitude error into every preintegrated block and that the error is
*correlated with motion* rather than random, so it does not average out; it
becomes drift.

Both sensors see the same angular rate: the gyroscope directly, the camera as
the rotation between consecutive frames divided by the interval.  The offset is
the lag that maximises their correlation.  A parabolic fit through the
correlation peak recovers a fraction of the grid step, which is what lets a
2 ms resolution give a sub-millisecond answer.

This is the method Kalibr uses to *initialise* and it is robust for the same
reason the rotation estimate is: it uses no structure, no scale and no
translation.

**The evaluation injects a known offset.**  A routine that always returns zero
would pass a test that checks it recovers zero on a well-synchronised dataset,
so `scripts/calibrate.py --inject-offset` shifts the camera clock by a known
amount first and reports the error against that.

A finer, continuous refinement is also implemented,
[`refine_by_feature_velocity`](../src/vio/calibration/temporal.py), following
VINS-Mono: if a feature crosses the image at velocity `v` and the frame was
really exposed `td` later than its timestamp says, the feature was at
`u + v * td`, so `td` gets a gradient.  Its cost has local minima about one
frame interval apart, which is why the correlation estimate initialises it.

## Translation: deliberately not estimated from real data

The lever arm between camera and IMU is only observable through the
acceleration that *rotation* induces at a displaced point.  Under pure
translation it is not observable at all and under modest rotation it is
observable but badly conditioned, because the estimate is dominated by whatever the
visual scale happened to be and on a monocular sequence that scale is itself
being estimated.

So `scripts/calibrate.py` reports the published translation and estimates
nothing.  Reporting a number there would be reporting noise with a decimal
point.

The machinery to estimate it exists and is exercised where the excitation can
be guaranteed:
[`ExtrinsicReprojectionFactor`](../src/vio/backend/factors.py) puts the
extrinsic into the bundle adjustment as a free variable, and
[`tests/test_calibration.py`](../tests/test_calibration.py) starts it 3 degrees
and 4 cm away from the truth on a synthetic sequence and converges back to
within 0.3 degrees and 1 cm.  `extrinsic.excitation()` reports the total
rotation swept about each body axis, which is the number to look at before
believing a translation estimate from any dataset.

## What is not calibrated at all

* **Camera intrinsics.**  Taken from the dataset.  Estimating them needs a
  target and a target is hardware.
* **IMU intrinsics**: scale factor, axis misalignment, g-sensitivity.  Only
  additive biases are estimated.  Reasonable for the ADIS16448 in EuRoC,
  not reasonable for a phone.
* **Rolling shutter line delay.**  EuRoC is global shutter.

## Where it sits in the estimator

Nowhere, by choice.  The estimator holds the extrinsic fixed and assumes the
timestamps are already aligned.  Leaving six extra degrees of freedom free
inside a ten-keyframe sliding window makes the solve slower and less stable for
no gain once the calibration has converged and the calibration is a
once-per-rig operation rather than a per-frame one.  Online estimation is the
right answer for a system whose mounting flexes; this is not one.
