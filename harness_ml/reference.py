"""NumPy reference implementation of the agreed implicit Jacobi update."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .problem import ProblemConfig, make_problem


@dataclass
class SolveResult:
    iterations: int
    time_ms: float
    final_residual: float
    converged: bool
    residual_history: list[tuple[int, float]]
    final_grid: np.ndarray


def grid_after_iterations(config: ProblemConfig, iterations: int) -> np.ndarray:
    """Return the reference grid after an exact iteration count for validation."""

    if iterations < 0:
        raise ValueError("iterations cannot be negative")
    old, source = make_problem(config)
    new = old.copy()
    r = np.float32(config.r)
    denominator = np.float32(1.0 + 4.0 * config.r)
    for _ in range(iterations):
        new[1:-1, 1:-1] = (
            old[1:-1, 1:-1]
            + r * (old[:-2, 1:-1] + old[2:, 1:-1] + old[1:-1, :-2] + old[1:-1, 2:])
            + source[1:-1, 1:-1]
        ) / denominator
        old, new = new, old
    return old.copy()


def solve_reference(
    config: ProblemConfig,
    *,
    tolerance: float = 1.0e-5,
    residual_interval: int = 10,
    max_iters: int = 2_000_000,
) -> SolveResult:
    """Solve with two-grid Jacobi iteration using float32 arithmetic.

    Update equation:
    ``u_new = (u_old + r * sum(neighbours) + dt*f) / (1 + 4*r)``.
    The convergence measure is the maximum absolute update, sampled every M
    iterations, matching the supplied CUDA baseline's current convention.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if residual_interval <= 0 or max_iters <= 0:
        raise ValueError("residual_interval and max_iters must be positive")

    old, source = make_problem(config)
    new = old.copy()
    r = np.float32(config.r)
    denominator = np.float32(1.0 + 4.0 * config.r)
    history: list[tuple[int, float]] = []
    residual = float("inf")
    converged = False
    start = perf_counter()

    for iteration in range(1, max_iters + 1):
        new[1:-1, 1:-1] = (
            old[1:-1, 1:-1]
            + r
            * (
                old[:-2, 1:-1]
                + old[2:, 1:-1]
                + old[1:-1, :-2]
                + old[1:-1, 2:]
            )
            + source[1:-1, 1:-1]
        ) / denominator

        if iteration % residual_interval == 0:
            residual = float(np.max(np.abs(new[1:-1, 1:-1] - old[1:-1, 1:-1])))
            history.append((iteration, residual))

        old, new = new, old
        if iteration % residual_interval == 0 and residual < tolerance:
            converged = True
            break

    elapsed_ms = (perf_counter() - start) * 1000.0
    return SolveResult(
        iterations=iteration,
        time_ms=elapsed_ms,
        final_residual=residual,
        converged=converged,
        residual_history=history,
        final_grid=old.copy(),
    )
