"""Shared fixtures.

The simulated sequences here are module-scoped because generating and
rendering one costs a few seconds and a dozen tests want the same one.  They
are all seeded, so a failure is reproducible from the test name alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio.sim.simulator import SimConfig, simulate  # noqa: E402
from vio.types import ImuBias, ImuNoise  # noqa: E402


@pytest.fixture(scope="session")
def clean_sequence():
    """No sensor noise and no bias: everything should be exact against this."""
    cfg = SimConfig(
        duration=6.0,
        imu_noise=ImuNoise(0.0, 0.0, 0.0, 0.0),
        initial_bias=ImuBias(),
        bias_walk=False,
        seed=1,
    )
    return simulate(cfg)


@pytest.fixture(scope="session")
def biased_sequence():
    """Noise-free but with a constant bias, for the bias-Jacobian tests."""
    cfg = SimConfig(
        duration=6.0,
        imu_noise=ImuNoise(0.0, 0.0, 0.0, 0.0),
        initial_bias=ImuBias([0.04, -0.02, 0.03], [0.003, -0.004, 0.002]),
        bias_walk=False,
        seed=2,
    )
    return simulate(cfg)


@pytest.fixture(scope="session")
def noisy_sequence():
    """Realistic noise and a drifting bias."""
    cfg = SimConfig(
        duration=12.0,
        initial_bias=ImuBias([0.04, -0.02, 0.03], [0.003, -0.004, 0.002]),
        seed=3,
    )
    return simulate(cfg)


@pytest.fixture(scope="session")
def rendered_sequence():
    """A short sequence with images, for front-end and pipeline tests."""
    from vio.sim.render import SceneRenderer

    cfg = SimConfig(duration=9.0, n_landmarks=1200, pixel_noise=0.4, seed=21)
    seq = simulate(cfg)
    renderer = SceneRenderer(seq, noise_sigma=2.0)
    images = [renderer.render(i) for i in range(len(seq.frames))]
    return seq, images


def finite_difference_jacobians(factor, values, h: float = 1e-6):
    """Central-difference Jacobians of a factor, for comparison with analytic ones."""
    from vio.backend.variables import Values

    out = []
    for key in factor.keys:
        var = values[key]
        J = np.zeros((factor.dim, var.dim))
        for i in range(var.dim):
            step = np.zeros(var.dim)
            step[i] = h
            plus = Values(values)
            plus[key] = var.retract(step)
            minus = Values(values)
            minus[key] = var.retract(-step)
            J[:, i] = (factor.residual(plus) - factor.residual(minus)) / (2.0 * h)
        out.append(J)
    return out


def assert_jacobians_match(factor, values, tolerance: float = 2e-5) -> None:
    analytic = factor.jacobians(values)
    numeric = finite_difference_jacobians(factor, values)
    for key, a, n in zip(factor.keys, analytic, numeric):
        scale = max(float(np.abs(n).max()), 1e-6)
        err = float(np.abs(a - n).max()) / scale
        assert err < tolerance, (
            f"{type(factor).__name__} Jacobian for {key} differs by {err:.2e}"
        )
