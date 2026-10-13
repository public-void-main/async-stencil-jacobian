"""Command-line wrapper for validating the harness with the CPU reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .problem import ProblemConfig
from .reference import solve_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--source", choices=["point", "gaussian"], required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--strategy", default="sync")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--residual-interval", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=2_000_000)
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--grid-output", type=Path)
    args = parser.parse_args()

    if args.strategy != "sync":
        raise SystemExit("the CPU reference only implements the sync strategy")
    config = ProblemConfig(args.N, args.source, args.r, seed=args.seed)
    try:
        result = solve_reference(
            config,
            tolerance=args.tolerance,
            residual_interval=args.residual_interval,
            max_iters=args.max_iters,
            precision=args.precision,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.grid_output:
        args.grid_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.grid_output, result.final_grid)

    print(
        json.dumps(
            {
                "iterations": result.iterations,
                "time_ms": result.time_ms,
                "final_residual": result.final_residual if np.isfinite(result.final_residual) else None,
                "final_residual_iteration": result.final_residual_iteration,
                "final_true_residual": result.final_true_residual if np.isfinite(result.final_true_residual) else None,
                "precision": result.precision,
                "converged": result.converged,
                "residual_history": [
                    [iteration, residual if np.isfinite(residual) else None]
                    for iteration, residual in result.residual_history
                ],
                "final_grid": str(args.grid_output) if args.grid_output else None,
            }, allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
