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
ones real-time systems use, the window is bounded, and the solve is a
Schur-complemented Levenberg-Marquardt like everyone else's.  The constant
factor is the language.  A C++ back end over the same graph would be
comfortably inside budget.

**The system is therefore run offline, over recorded sequences, and no
real-time claim is made anywhere.**

## It is monocular, and only monocular

EuRoC ships stereo and the loader reads both cameras, but the estimator uses
`cam0` only.  Stereo would fix scale at initialisation instead of requiring
motion, would make triangulation well conditioned from the first frame, and
would remove the excitation gate entirely.  It is the single largest accuracy
improvement available and it is not implemented.

## It has never touched hardware

Every number comes from EuRoC or from the simulator.  There is no rig, no
recording of my own, and no claim about how it behaves on a camera it has not
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
  no parameter adjustment fixes it.  Loop closure is therefore evaluated on
  EuRoC, and on synthetic imagery only as far as the plumbing.
* **Tracking is easier than it should be.**  Track lifetimes and outlier rates
  on synthetic data are not predictive of real data, in either direction.

The simulator exists to give each component an exact answer to check against
and to let continuous integration run the whole pipeline with no dataset.  It
is not a benchmark and no README number comes from it.

## Initialisation needs motion, and says so

Scale is unobservable without translational excitation.  The initialiser
refuses below 0.10 m/s² of world-frame acceleration spread and keeps
collecting keyframes until the condition is met.  On a sequence that begins
with a long hover, and MH_01 and MH_02 both do, that means the first seconds
produce no metric trajectory at all, and `poses(initialised_only=True)` drops
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
40x slower solve for no measurable accuracy change, and the ablation is in
[results.md](results.md).  A proper treatment would use inverse-depth landmarks
anchored to their first observing keyframe, which is what makes the full
marginalisation cheap in VINS-Mono, and that is not implemented.

## The pose graph is four degrees of freedom by construction, not by design

Roll and pitch are pinned by a gravity-direction factor on every node rather
than by a purpose-built four-degree-of-freedom parameterisation.  The effect is
the same and it costs no new variable type, but the null space is enforced
numerically through a rank-two residual rather than structurally, so a
sufficiently large loop error can still tilt the map by an amount the
gravity sigma allows.

## The visual-only baseline reports at keyframe rate, not frame rate

With the IMU switched off there is nothing to propagate a pose between
keyframes, so the estimator holds the last keyframe's pose until the next one.
Interpolating towards a pose it has not computed yet would be inventing
information, and extrapolating at constant velocity would be inventing a
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
merge landmarks, do not re-triangulate across the loop, and do not feed
anything back into the sliding window.  The map is a by-product, not a
maintained global structure.  There is no relocalisation after tracking loss:
if the front end fails, the estimator reports the failure and continues from
whatever it can, rather than recovering its old frame.

## No IMU intrinsic calibration

Scale factor, axis misalignment and g-sensitivity are all assumed ideal.  Only
additive biases are estimated.  For the ADIS16448 in EuRoC this is a reasonable
approximation; for a phone-grade IMU it is not.

## Rolling shutter is not modelled

EuRoC's cameras are global shutter, so this costs nothing here.  On a
rolling-shutter camera every row is exposed at a different instant and the
reprojection model above is wrong in a motion-dependent way.

## The evaluation uses a single run per sequence

The estimator is deterministic given its inputs, so there is no seed variance
to report on EuRoC.  But it also means the numbers describe one configuration
on one dataset, and the sensitivity to the dozen thresholds in
`EstimatorConfig` is characterised only for the window size; see the ablation
in [results.md](results.md).

## What would move the numbers most, in order

1. **Stereo.**  Removes the scale problem entirely and fixes initialisation.
2. **A C++ back end.**  Turns an offline system into a real-time one.
3. **Full marginalisation with inverse-depth landmarks.**  Recovers the
   information currently dropped at the window boundary.
4. **Learned features** in place of Shi-Tomasi and KLT, for the sequences where
   the front end is what fails. V1_03 and MH_04 are both front-end limited.
5. **A photometric or warp-aware renderer**, so the synthetic tests could cover
   place recognition too.
