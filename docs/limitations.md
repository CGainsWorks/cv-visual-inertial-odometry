# Limitations

Everything this system does not do, stated plainly.  The list is longer than
the results section, which is the correct proportion for a from-scratch
implementation of a problem that took the field two decades.

---

## It does not run in real time

This is the largest limitation and it is not fixable within Python.  The front
end costs about 8 ms per frame and the window solve about 60 to 150 ms per
keyframe on one core of a modern desktop; with keyframes at roughly a third of
frames that lands between 60 and 110 ms per frame, against the 50 ms a 20 Hz
camera allows.  The per-stage numbers are in [results.md](results.md) and
`make profile` regenerates them.

Nothing in the design prevents real-time operation: the algorithms are the
ones real-time systems use, the window is bounded and the solve is a
Schur-complemented Levenberg-Marquardt like everyone else's.  The constant
factor is the language.  A C++ back end over the same graph would be
comfortably inside budget.

**The system is therefore run offline, over recorded sequences and no
real-time claim is made anywhere.**

## It is monocular and only monocular

EuRoC ships stereo and the loader reads both cameras but the estimator uses
`cam0` only.  Stereo would fix scale at initialisation instead of requiring
motion, would make triangulation well conditioned from the first frame, and
would remove the excitation gate entirely.  It is the single largest accuracy
improvement available and it is not implemented.

## TUM VI is supported but not run

The loader reads the ASL directory layout that both EuRoC and TUM VI use, and
the camera model implements the equidistant distortion TUM VI ships, with a
round-trip test to machine precision against a real TUM VI intrinsic set.  But
no TUM VI sequence has been run end to end here, so there is no TUM VI number
anywhere and none should be inferred.  The fisheye field of view is wide enough
that the keyframe and triangulation thresholds, which are expressed in
normalised coordinates and tuned on a 78-degree camera, would want revisiting.

## It has never touched hardware

Every number comes from EuRoC or from the simulator.  There is no rig, no
recording of my own and no claim about how it behaves on a camera it has not
seen.  The camera-IMU calibration is evaluated against EuRoC's published
extrinsic; the time offset is evaluated by injecting a known offset into a
sequence and recovering it, not by calibrating an uncharacterised device.

## The simulated images are not photorealistic

[`vio/sim/render.py`](../src/vio/sim/render.py) draws textured patches stuck to
scene points.  It does not model occlusion, motion blur, rolling shutter,
auto-exposure, illumination change, or the fact that a real surface warps under
viewpoint change.  Two consequences:

* **ORB descriptors do not match across these images.**  Measured: 25
  ratio-test matches between frames ten apart, against 386 on EuRoC V1_01.
  The cause is structural, since patches never warp and their overlaps change, so
  no parameter adjustment fixes it.  End to end the effect is unambiguous.  The
  generated circuit used by continuous integration returns to the same place
  and the same attitude once per lap, which gives the detector 17 revisit
  episodes across 598 keyframes.  It converts none of them.  The comparable
  figures on real data are 6 episodes of 21 on MH_05 and 38 of 317 on V1_02.
  Loop closure is therefore evaluated on EuRoC; the synthetic run checks
  that the pipeline runs rather than that it works.
* **Tracking is easier than it should be.**  Track lifetimes and outlier rates
  on synthetic data are not predictive of real data, in either direction.

The simulator exists to give each component an exact answer to check against
and to let continuous integration run the whole pipeline with no dataset.  It
is not a benchmark and no README number comes from it.

## Initialisation needs motion and says so

Scale is unobservable without translational excitation.  The initialiser
refuses below 0.10 m/s² of world-frame acceleration spread and keeps
collecting keyframes until the condition is met.  On a sequence that begins
with a long hover, as MH_01 and MH_02 both do, that means the first seconds
produce no metric trajectory at all and `poses(initialised_only=True)` drops
them.  A stereo pair or a wheel odometer would remove the problem; refusing is
the correct behaviour without one.

The accelerometer bias is not estimated at initialisation, so the first metric
scale carries whatever error the bias contributes, typically two to five per
cent on the synthetic sequence with a 0.06 m/s² bias.  It converges over the
next few seconds through the sliding window.

## Marginalisation is partial

Only the inertial factors are marginalised when a keyframe leaves the window;
the departing keyframe's landmark observations are dropped rather than
absorbed.  This is a real approximation and it discards information.  It was
made deliberately after measuring the alternative, a 550x550 dense prior and a
40x slower solve for no measurable accuracy change and the ablation is in
[results.md](results.md).  A proper treatment would use inverse-depth landmarks
anchored to their first observing keyframe, which is what makes the full
marginalisation cheap in VINS-Mono and that is not implemented.

## The pose graph is four degrees of freedom by construction, not by design

Roll and pitch are pinned by a gravity-direction factor on every node rather
than by a purpose-built four-degree-of-freedom parameterisation.  The effect is
the same and it costs no new variable type but the null space is enforced
numerically through a rank-two residual rather than structurally, so a
sufficiently large loop error can still tilt the map by an amount the
gravity sigma allows.

## The visual-only baseline reports at keyframe rate, not frame rate

With the IMU switched off there is nothing to propagate a pose between
keyframes, so the estimator holds the last keyframe's pose until the next one.
Interpolating towards a pose it has not computed yet would be inventing
information and extrapolating at constant velocity would be inventing a
velocity a monocular system does not have.  The consequence is that the
visual-only trajectory is a staircase at the frame level, which adds a small
amount to its error and, more visibly, makes it lag: feeding those held poses
into the camera-IMU time-offset correlation produced about 7 ms of apparent
offset on a sequence whose true offset is zero.  ``scripts/calibrate.py`` uses
keyframe poses for that reason.

The visual-inertial configuration does not have the problem, because
preintegration propagates the state between keyframes exactly.

## Loop closure is not a full SLAM back end

Verified loops correct the *trajectory* through a pose graph.  They do not
merge landmarks, do not re-triangulate across the loop and do not feed
anything back into the sliding window.

The sparse map written beside each trajectory is a by-product of that, not a
maintained global structure.  Each landmark's position is recorded as it leaves
the sliding window and is never revisited, so a loop closure that moves the
trajectory by twenty centimetres leaves the cloud where it was and the two
disagree by that much after the closure.  Fixing it means either keeping every
landmark in the optimisation, which is what makes a full SLAM back end
expensive, or rigidly transforming each landmark with its anchoring keyframe's
correction, which is cheap and is not implemented.  The map is written for
inspection rather than for use.

There is no relocalisation after tracking loss either: if the front end fails,
the estimator restarts its bootstrap from the present rather than recovering
its old frame, so the trajectory continues in a new one.

## No IMU intrinsic calibration

Scale factor, axis misalignment and g-sensitivity are all assumed ideal.  Only
additive biases are estimated.  For the ADIS16448 in EuRoC this is a reasonable
approximation; for a phone-grade IMU it is not.

## Rolling shutter is not modelled

EuRoC's cameras are global shutter, so this costs nothing here.  On a
rolling-shutter camera every row is exposed at a different instant and the
reprojection model above is wrong in a motion-dependent way.

## The headline numbers are single runs, which is enough

Given a commit and a machine the estimator is bit-for-bit reproducible; a
test runs it in two processes with different hash seeds to keep it so.  Across
machines it is not: the same generated sequence scores 0.2573 m here and
0.2391 m on the continuous integration runner, which has different BLAS and
OpenCV binaries underneath.  It is also insensitive
to the one input that looked stochastic.  `make seed-spread` varies nothing but
the seed OpenCV's generator starts from and across five seeds on each of two
sequences the ATE is identical to every digit printed.  `cv2.setRNGSeed` does
not reach `findFundamentalMat` or `findEssentialMat` under OpenCV 5, which
sample through USAC instead.

That makes a single run the answer rather than one draw from a distribution.
The limitation worth stating is narrower than the one this section used to
claim: the result holds for the OpenCV version pinned in `requirements.txt`,
and a build whose RANSAC does consume the global generator would need the sweep
run again before its numbers could be compared with these.

Beyond that, the numbers still describe one configuration on one dataset, and
the sensitivity to the dozen thresholds in `EstimatorConfig` is characterised
only for the window size; see the ablation in [results.md](results.md).

## What would move the numbers most, in order

1. **Stereo.**  Removes the scale problem entirely and fixes initialisation.
2. **A C++ back end.**  Turns an offline system into a real-time one.
3. **Full marginalisation with inverse-depth landmarks.**  Recovers the
   information currently dropped at the window boundary.
4. **Learned features** in place of Shi-Tomasi and KLT, for the sequences where
   the front end is what fails. V1_03 and MH_04 are both front-end limited.
5. **A photometric or warp-aware renderer**, so the synthetic tests could cover
   place recognition too.
