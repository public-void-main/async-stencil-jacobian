"""Deterministic problem generation for the Jacobi experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        if self.N < 3:
            raise ValueError("N must be at least 3")
        if self.source not in {"point", "gaussian"}:
            raise ValueError("source must be 'point' or 'gaussian'")
        if self.r <= 0:
            raise ValueError("r must be positive")
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
    if config.source == "point":
        source[center, center] = np.float32(config.source_strength)
    else:
        y, x = np.mgrid[0:n, 0:n]
        sigma = max(n / 16.0, 1.0)
        exponent = -((x - center) ** 2 + (y - center) ** 2) / (2.0 * sigma**2)
        source = (config.source_strength * np.exp(exponent)).astype(np.float32)

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
