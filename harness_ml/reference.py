"""NumPy reference implementation of the agreed implicit Jacobi update."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .problem import (
    ProblemConfig, make_problem, validate_grid_array, validate_integer, validate_real,
)


@dataclass
class SolveResult:
    iterations: int
    time_ms: float
    final_residual: float
    converged: bool
    residual_history: list[tuple[int, float]]
    final_grid: np.ndarray
    final_true_residual: float = float("inf")
    precision: str = "float32"
    final_residual_iteration: int | None = None


def true_residual(grid, source, r):
    """Evaluate interior max |F(u)-u|; non-finite grids return infinity."""
    u = validate_grid_array(grid, "grid")
    f = validate_grid_array(source, "source")
    if f.shape != u.shape:
        raise ValueError("source shape must match grid shape")
    validate_real(r, "r", positive=True)
    if not np.isfinite(f).all():
        raise ValueError("source must contain only finite values")
    u = u.astype(np.float64, copy=False)
    f = f.astype(np.float64, copy=False)
    r = np.float64(r)
    if not np.isfinite(u).all():
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        denom = 1.0 + 4.0 * r
    if not np.isfinite(denom):
        raise ValueError("r produces a non-finite denominator")
    updated = (
        u[1:-1, 1:-1]
        + r * (
            u[:-2, 1:-1] + u[2:, 1:-1]
            + u[1:-1, :-2] + u[1:-1, 2:]
        )
        + f[1:-1, 1:-1]
    ) / denom
    residual = float(np.max(np.abs(updated - u[1:-1, 1:-1])))
    return residual if np.isfinite(residual) else float("inf")


def grid_after_iterations(config: ProblemConfig, iterations: int) -> np.ndarray:
    """Return the reference grid after an exact iteration count for validation."""

    validate_integer(iterations, "iterations", minimum=0)
    old, source = make_problem(config)
    new = old.copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        r = np.float32(config.r)
        denominator = np.float32(1.0 + 4.0 * config.r)
    if not np.isfinite(r) or r <= 0 or not np.isfinite(denominator):
        raise ValueError("r and denominator must be finite and representable in float32")
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
    tolerance: float = 1.0e-4,
    residual_interval: int = 10,
    max_iters: int = 2_000_000,
    precision: str = "float32",
) -> SolveResult:
    """Solve with two-grid Jacobi iteration at the selected precision.

    Update equation:
    ``u_new = (u_old + r * sum(neighbours) + dt*f) / (1 + 4*r)``.
    The convergence measure is the maximum absolute update, sampled every M
    iterations. Default float32 stopping matches the CUDA baseline's convention;
    it does not guarantee the independent final_true_residual meets tolerance.
    Float64 mode also requires that independent residual before convergence.
    Both modes use the same float32-generated source samples and boundaries.
    final_residual is the last sampled update, associated with
    final_residual_iteration (None if unsampled), and may precede the final grid
    when the cap is between checkpoints. final_true_residual always evaluates
    the completed grid independently.
    """

    validate_real(tolerance, "tolerance", positive=True)
    validate_integer(residual_interval, "residual_interval", minimum=1)
    validate_integer(max_iters, "max_iters", minimum=1)
    if not isinstance(precision, str) or precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")

    old, source = make_problem(config)
    validate_grid_array(old, "initial grid")
    validate_grid_array(source, "source")
    if source.shape != old.shape or old.shape != (config.N, config.N):
        raise ValueError("generated grids must match the problem shape")
    if not np.isfinite(old).all() or not np.isfinite(source).all():
        raise ValueError("generated grids and source must contain only finite values")
    dtype = np.float32 if precision == "float32" else np.float64
    old = old.astype(dtype, copy=False)
    source = source.astype(dtype, copy=False)
    new = old.copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        r = dtype(config.r)
        denominator = dtype(1.0 + 4.0 * config.r)
    if not np.isfinite(r) or r <= 0 or not np.isfinite(denominator):
        raise ValueError("r and denominator must be finite and representable at the selected precision")
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
            if precision == "float32" or true_residual(old, source, config.r) < tolerance:
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
        final_true_residual=true_residual(old, source, config.r),
        precision=precision,
        final_residual_iteration=history[-1][0] if history else None,
    )
