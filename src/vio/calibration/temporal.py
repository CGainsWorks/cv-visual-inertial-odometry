"""Camera-IMU time offset estimation.

Cameras and IMUs are almost never on the same clock.  A few milliseconds of
offset does not sound like much until you notice that at 200 degrees per second
it is a degree of attitude error injected into every preintegrated block, and
that the error is *correlated with motion* rather than random, so it does not
average out, it becomes drift.

Two estimators are here, and they answer the same question at different
resolutions.

**Correlation**, :func:`estimate_by_correlation`.  The gyroscope measures
angular rate directly.  The camera measures it too, indirectly, as the rotation
between consecutive frames divided by the interval.  Both are the same physical
signal seen through different clocks, so the offset is the lag that maximises
their correlation.  Resolution is limited by the frame interval, which a
parabolic fit through the correlation peak improves to a fraction of a frame.
This is the method Kalibr uses to *initialise*, and it is robust because it
uses no structure, no scale and no extrinsic translation, only the fact that
both sensors are attached to the same rigid body.

**Refinement**, :func:`refine_by_feature_velocity`.  Following VINS-Mono: if a
feature is moving across the image at velocity ``v`` and the frame was actually
exposed ``td`` later than its timestamp says, then the feature was really at
``u + v * td``.  Making ``td`` a variable in the reprojection residual gives a
gradient, and the estimate becomes continuous rather than quantised by the
frame rate.

The correlation estimate is used to initialise the refinement, because the
refinement's cost has local minima roughly one frame interval apart and will
happily settle into the wrong one from a cold start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import so3


@dataclass
class TimeOffsetResult:
    ok: bool
    reason: str = ""
    #: Seconds to add to a camera timestamp to put it on the IMU clock.
    offset: float = 0.0
    #: Peak normalised correlation, in [-1, 1].  Below about 0.5 the answer is
    #: not to be trusted whatever it says.
    correlation: float = 0.0
    #: Search resolution actually achieved, seconds.
    resolution: float = 0.0
    n_samples: int = 0
    refined: bool = False


def angular_rate_from_rotations(times, rotations) -> tuple:
    """Angular speed between consecutive absolute rotations."""
    times = np.asarray(times, dtype=float)
    rate = np.empty(len(times) - 1)
    mid = 0.5 * (times[:-1] + times[1:])
    for i in range(len(rate)):
        dt = times[i + 1] - times[i]
        if dt <= 0:
            rate[i] = 0.0
            continue
        rate[i] = np.linalg.norm(so3.log(rotations[i].T @ rotations[i + 1])) / dt
    return mid, rate


def estimate_by_correlation(
    cam_times,
    cam_rotations,
    imu_times,
    imu_gyro,
    max_offset: float = 0.15,
    steps: int = 601,
) -> TimeOffsetResult:
    """Time offset from the lag that best aligns the two angular-rate signals.

    Positive means the camera's timestamps are *early*: the frame labelled
    ``t`` was actually exposed at ``t + offset``.
    """
    cam_times = np.asarray(cam_times, dtype=float)
    imu_times = np.asarray(imu_times, dtype=float)
    if len(cam_times) < 20 or len(imu_times) < 20:
        return TimeOffsetResult(False, "not enough samples")

    mid, cam_rate = angular_rate_from_rotations(cam_times, cam_rotations)
    imu_rate = np.linalg.norm(np.asarray(imu_gyro, dtype=float), axis=1)

    if float(np.std(cam_rate)) < 1e-4:
        return TimeOffsetResult(False, "no rotational excitation")

    lo = max(mid[0], imu_times[0] + max_offset) + 1e-6
    hi = min(mid[-1], imu_times[-1] - max_offset) - 1e-6
    inside = (mid >= lo) & (mid <= hi)
    if inside.sum() < 20:
        return TimeOffsetResult(False, "overlap too short")
    mid, cam_rate = mid[inside], cam_rate[inside]
    cam_centred = cam_rate - cam_rate.mean()
    cam_norm = float(np.linalg.norm(cam_centred))

    offsets = np.linspace(-max_offset, max_offset, steps)
    scores = np.empty(steps)
    for k, off in enumerate(offsets):
        sampled = np.interp(mid + off, imu_times, imu_rate)
        s = sampled - sampled.mean()
        denom = cam_norm * float(np.linalg.norm(s))
        scores[k] = float(cam_centred @ s) / denom if denom > 1e-12 else 0.0

    k = int(np.argmax(scores))
    best = float(offsets[k])
    # Parabolic interpolation through the peak: the correlation is smooth in
    # the offset, so this recovers a fraction of the grid step for free.
    if 0 < k < steps - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            best += 0.5 * (y0 - y2) / denom * (offsets[1] - offsets[0])

    return TimeOffsetResult(
        ok=True,
        offset=best,
        correlation=float(scores[k]),
        resolution=float(offsets[1] - offsets[0]),
        n_samples=int(len(mid)),
    )


def feature_velocities(previous, current, dt: float) -> dict:
    """Normalised-coordinate velocity of each feature shared by two frames."""
    if dt <= 1e-9:
        return {}
    out = {}
    for tid, xn in current.items():
        prev = previous.get(tid)
        if prev is not None:
            out[tid] = (np.asarray(xn) - np.asarray(prev)) / dt
    return out


def refine_by_feature_velocity(
    frames,
    predictions,
    initial: float = 0.0,
    max_offset: float = 0.15,
    iterations: int = 20,
) -> TimeOffsetResult:
    """One-dimensional Gauss-Newton on the time offset.

    ``frames`` is a list of ``(stamp, {track: normalised_xy})``.
    ``predictions`` supplies, for each frame index, ``{track: predicted_xy}``
    from the current geometry, where the estimator thinks each landmark
    should appear.  The residual is::

        r = observed + velocity * td - predicted

    so the derivative with respect to ``td`` is just the feature velocity,
    which is why this is cheap enough to run every keyframe.
    """
    td = float(initial)
    n_used = 0
    for _ in range(iterations):
        H = 0.0
        b = 0.0
        n_used = 0
        for i in range(1, len(frames)):
            t_prev, prev = frames[i - 1]
            t_cur, cur = frames[i]
            pred = predictions.get(i)
            if not pred:
                continue
            vel = feature_velocities(prev, cur, t_cur - t_prev)
            for tid, v in vel.items():
                p = pred.get(tid)
                if p is None:
                    continue
                r = np.asarray(cur[tid]) + v * td - np.asarray(p)
                H += float(v @ v)
                b += float(v @ r)
                n_used += 1
        if n_used < 20 or H <= 1e-12:
            return TimeOffsetResult(False, "not enough moving features",
                                    offset=td, n_samples=n_used)
        step = -b / H
        td += step
        td = float(np.clip(td, -max_offset, max_offset))
        if abs(step) < 1e-7:
            break
    return TimeOffsetResult(True, offset=td, n_samples=n_used, refined=True)
