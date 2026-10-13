"""Deterministic problem generation for the Jacobi experiments."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def validate_real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def validate_integer(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def validate_grid_array(array, name):
    """Accept square float32/float64 arrays without casting or broadcasting."""
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 3:
        raise ValueError(f"{name} must be a square 2D grid of size at least 3")
    if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
        raise ValueError(f"{name} must have a real float32 or float64 dtype")
    return array


@dataclass(frozen=True)
class ProblemConfig:
    """All pre-runtime features used by the experiment and ML pipeline."""

    N: int
    source: str
    r: float
    seed: int = 52
    boundary: str = "dirichlet"
    source_strength: float = 1.0

    def validate(self) -> None:
        validate_integer(self.N, "N", minimum=3)
        validate_integer(self.seed, "seed")
        if not isinstance(self.source, str) or self.source not in {"point", "gaussian"}:
            raise ValueError("source must be 'point' or 'gaussian'")
        validate_real(self.r, "r", positive=True)
        validate_real(self.source_strength, "source_strength")
        if self.boundary != "dirichlet":
            raise ValueError("the reduced project scope supports Dirichlet boundaries only")


def make_problem(config: ProblemConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(initial_grid, dt_times_source)`` as float32 arrays.

    The fixed boundary matches the supplied synchronous implementation: the top
    edge is 100 and the other edges are zero.  The source array represents the
    complete ``dt * f`` term in the agreed update equation.  Its peak magnitude
    is ``source_strength`` for both source types.
    """

    config.validate()
    n = config.N
    grid = np.zeros((n, n), dtype=np.float32)
    grid[0, :] = np.float32(100.0)

    source = np.zeros_like(grid)
    center = n // 2
    with np.errstate(over="ignore", invalid="ignore"):
        if config.source == "point":
            source[center, center] = np.float32(config.source_strength)
        else:
            y, x = np.mgrid[0:n, 0:n]
            sigma = max(n / 16.0, 1.0)
            exponent = -((x - center) ** 2 + (y - center) ** 2) / (2.0 * sigma**2)
            source = (config.source_strength * np.exp(exponent)).astype(np.float32)
    if not np.isfinite(source).all():
        raise ValueError("generated source must contain only finite float32 values")

    # Sources never overwrite fixed boundaries.
    source[0, :] = 0.0
    source[-1, :] = 0.0
    source[:, 0] = 0.0
    source[:, -1] = 0.0
    return grid, source


def deterministic_seed(config: ProblemConfig, run_id: int) -> int:
    """Stable seed recorded by the harness for reproducible repeated runs."""

    source_offset = 0 if config.source == "point" else 10_000
    return int(config.seed + config.N * 100 + round(config.r) * 10 + source_offset + run_id)
