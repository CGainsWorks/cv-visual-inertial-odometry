# Design

How the pieces fit together, why each convention is what it is, and which
decisions were made from measurement rather than taste.

---

## The shape of the system

```
image ──▶ Tracker ──▶ keyframe? ──▶ Estimator ──▶ pose per frame
             │                          │
        gyro (optional)                 ├── triangulate new landmarks
                                        ├── build the window graph
                                        ├── Levenberg-Marquardt with Schur
                                        └── marginalise the oldest keyframe
                                                    │
IMU ──▶ preintegration ─────────────────────────────┘

                        (separately, per keyframe)
                    ORB ──▶ bag of words ──▶ candidates
                                              │
                                       geometric verification
                                              │
                                          pose graph
```

The estimator is a **fixed-lag smoother**: a sliding window of ten keyframes,
optimised jointly, with everything older collapsed into a linear prior.  Loop
closure runs beside it rather than inside it, because a window has a bounded
per-keyframe cost and a loop correction is global.

## Conventions, fixed once

These are the things that quietly ruin a visual-inertial system when they are
inconsistent, so each is decided in exactly one place and used everywhere.

| Thing | Choice | Where |
| --- | --- | --- |
| Rotation perturbation | right: `R ⊞ δ = R Exp(δ)` | [`vio/geometry/so3.py`](../src/vio/geometry/so3.py) |
| Pose retraction | `SO(3) x R³`, translation in the **world** frame | [`vio/backend/variables.py`](../src/vio/backend/variables.py) |
| Tangent ordering | `[rotation, translation]` | same |
| Bias ordering | `[accelerometer, gyroscope]` | [`vio/types.py`](../src/vio/types.py) |
| IMU residual ordering | `[rotation, velocity, position]` | [`vio/imu/preintegration.py`](../src/vio/imu/preintegration.py) |
| Gravity | `[0, 0, -9.81]`, world `z` up | [`vio/types.py`](../src/vio/types.py) |
| Extrinsic | `T_body_cam` maps camera coordinates into the body frame | [`vio/io/euroc.py`](../src/vio/io/euroc.py) |
| Image coordinates in the back end | undistorted, normalised; residuals scaled to pixels | [`vio/geometry/camera.py`](../src/vio/geometry/camera.py) |

The pose retraction deserves its paragraph.  It is **not** the SE(3)
exponential.  Treating a pose as a point on the product manifold `SO(3) x R³`
means a translation step means the same thing regardless of the current
attitude, which both simplifies every Jacobian in
[`vio/backend/factors.py`](../src/vio/backend/factors.py) and improves the
conditioning of the Hessian's translation block.  `vio.geometry.se3` keeps the
exponential form for the places that genuinely need the group structure.

## Layout, and where it departs from the brief

The brief proposed `src/{frontend,imu,backend,loop_closure,calibration,io}`.
That is most of what is here, plus three additions, each of which earned its
place:

* **`geometry/`**: Lie group maths and the camera model.  These are used by
  every other module including the simulator, and putting them under
  `frontend/` would have made `sim` depend on `frontend`, which is backwards.
* **`sim/`**: the analytic trajectory, the measurement generator and the
  renderer.  This is the single most useful directory in the repository during
  development: it is what gives every component an exact answer to be checked
  against, and it is what lets continuous integration run the whole pipeline
  with no dataset.
* **`pipeline/`**: the estimator that ties the others together.  It is neither
  front end nor back end and giving it its own name stops it accreting into
  one of them.

`evaluation/` sits at the top level as the brief asked, and additionally as
`src/vio/evaluation/` for the metric implementations the estimator imports.

## The three pieces that are actually hard

### 1. Preintegration ([`vio/imu/preintegration.py`](../src/vio/imu/preintegration.py))

Forster et al.'s on-manifold formulation, written out: nine-dimensional deltas,
a nine-by-nine covariance propagated alongside, and first-order bias Jacobians
so the block can be corrected as the bias estimate moves without re-integrating.

The default integration scheme is **midpoint, not the Euler scheme in the
paper**, and getting midpoint right was not obvious.  Averaging the two raw
accelerometer readings while still rotating them by the start-of-interval
attitude makes velocity *worse* than plain Euler.  Each acceleration has to be
rotated by the attitude at its own end of the interval before averaging.  With
that fixed, midpoint is between two and three orders of magnitude more accurate
for the same cost; the table is in [results.md](results.md).

### 2. The back end ([`vio/backend/`](../src/vio/backend/))

A factor graph, analytic Jacobians, Levenberg-Marquardt with Marquardt damping,
and a Schur complement that eliminates landmarks before the linear solve.  For
a ten-keyframe window with 150 landmarks that turns a 600-dimensional solve
into a 150-dimensional one.

Two things make it fast enough to be usable in Python:

* **Vectorised factor groups.**  Every landmark observation in the window is
  evaluated in one array pass by
  [`ReprojectionGroup`](../src/vio/backend/groups.py) rather than a thousand
  separate `Factor` objects.  Measured: 3.5 ms against 45 ms per linearisation
  for identical output, agreeing to 1e-11 in the Jacobian.  The scalar factors
  remain as the readable definition, and a test asserts the two agree.
* **A small marginal prior.**  Only the inertial factors are marginalised, not
  the departing keyframe's landmark observations.  Marginalising the visual
  factors naively turns a 15x15 dense prior into a 550x550 one, which took the
  per-keyframe solve from 26 ms to 1042 ms for no measurable accuracy gain.
  The IMU block is what carries scale and gravity across the window boundary
  and that is kept.  See [`marginalisation.py`](../src/vio/backend/marginalisation.py).

Every analytic Jacobian is checked against a central finite difference of its
own residual, at randomised linearisation points, in
[`tests/test_factors.py`](../tests/test_factors.py).  That test is the reason
to believe the back end.

### 3. Initialisation ([`vio/imu/initialisation.py`](../src/vio/imu/initialisation.py))

The step where from-scratch visual-inertial odometry usually dies.  Gyroscope
bias from rotation alone, then a single linear least-squares problem for
gravity, metric scale and the velocity at every keyframe, then a tangent-plane
refinement that enforces `|g| = 9.81`.

It **refuses** rather than guessing when the window has no translational
excitation, because without it the scale column of the linear system is nearly
collinear with the gravity columns and the solve returns a confident, arbitrary
number.  Measuring that excitation correctly matters too: the spread of the
*world-frame* specific force, not the body-frame one, because gravity swings
around the body frame whenever the platform rotates and a body-frame measure
mostly reports rotation.

Accelerometer bias is deliberately not estimated here.  Over a two-second
window it is nearly indistinguishable from a tilt of gravity, and trying to
separate them makes the system worse conditioned rather than more informative.
It is left at zero and picked up by the sliding window over the following
seconds.

## What is borrowed

| Component | Library | Why not written out |
| --- | --- | --- |
| Pyramidal Lucas-Kanade | OpenCV | A Python version is two orders of magnitude slower and adds nothing beyond the 1991 paper |
| Shi-Tomasi corners, ORB descriptors | OpenCV | Same |
| RANSAC essential matrix, PnP | OpenCV | Well-tested minimal solvers; the back end is the part worth judging |
| Sparse factorisation | SciPy | Building the normal equations is the structural work, not factorising them |
| Independent ATE/RPE referee | evo | A metric that agrees only with itself is not evidence |

Everything else, meaning Lie group maths, camera models and their inverses,
preintegration, the factor graph, the solver, marginalisation, initialisation,
triangulation, the keyframe policy, the bag-of-words vocabulary and index, the
pose graph, the calibration, the metrics and the simulator, is in this
repository.

## Decisions that came from measurement

Each of these started as an assumption and was changed by a number.  The
numbers are regenerated by `make ablate` and recorded in [results.md](results.md).

| Assumed | Measured | Now |
| --- | --- | --- |
| Midpoint integration is a small refinement | 2400x more accurate over a 1 s block | default |
| Gyro-predicted KLT guess helps tracking | halves the guess error, gains **zero** features at 20 Hz | off by default, documented with the frame-rate table |
| A redescending kernel protects the pose graph from bad loops | crushed a genuine 0.37 m correction to 0.04 mm | Huber with a generous threshold; robustness lives in the verification |
| Marginalising everything is the textbook thing to do | 40x slower for no accuracy change | inertial factors only |
| Descriptors can be computed wherever a landmark projects | they match nothing; detections are at corners | map descriptors are a subset of the detections |
| Synthetic imagery is good enough for place recognition | 25 ORB matches against EuRoC's 386 | loop closure is evaluated on EuRoC; CI tests the plumbing |
