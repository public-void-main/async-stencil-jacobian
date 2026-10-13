"""Compare two grids and check their true fixed-point residuals."""

import argparse
import json
from pathlib import Path

import numpy as np

from .problem import ProblemConfig, make_problem, validate_grid_array, validate_real
from .reference import true_residual  # Preserve the existing public import name.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-grid", type=Path, required=True)
    parser.add_argument("--reference-grid", type=Path, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--source", choices=["point", "gaussian"], required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--max-error", type=float, default=1e-4)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        validate_real(args.tolerance, "tolerance", positive=True)
        validate_real(args.max_error, "max_error", nonnegative=True)
        initial, source = make_problem(ProblemConfig(args.N, args.source, args.r))
        gpu = validate_grid_array(np.load(args.gpu_grid), "candidate grid")
        reference = validate_grid_array(np.load(args.reference_grid), "reference grid")
    except ValueError as error:
        parser.error(str(error))

    if gpu.shape != (args.N, args.N):
        raise SystemExit(f"Unexpected GPU grid shape: {gpu.shape}")
    if reference.shape != gpu.shape:
        raise SystemExit(f"Reference shape {reference.shape} != GPU shape {gpu.shape}")

    boundary = np.ones(initial.shape, dtype=bool)
    boundary[1:-1, 1:-1] = False
    gpu_boundary_correct = bool(np.array_equal(gpu[boundary], initial[boundary]))
    reference_boundary_correct = bool(np.array_equal(reference[boundary], initial[boundary]))

    difference = np.abs(gpu.astype(np.float64) - reference.astype(np.float64))
    try:
        gpu_residual = true_residual(gpu, source, args.r)
        ref_residual = true_residual(reference, source, args.r)
    except ValueError as error:
        parser.error(str(error))
    max_error = float(difference.max())
    # Scale before squaring so large finite differences do not overflow RMSE.
    rmse = None
    if np.isfinite(max_error):
        rmse = 0.0 if max_error == 0 else float(
            max_error * np.sqrt(np.mean((difference / max_error) ** 2))
        )

    passed = (
        float(difference.max()) <= args.max_error
        and gpu_residual < args.tolerance
        and ref_residual < args.tolerance
        and gpu_boundary_correct
        and reference_boundary_correct
    )

    print(json.dumps({
        "max_abs_error": max_error if np.isfinite(max_error) else None,
        "rmse": rmse,
        "gpu_true_residual": gpu_residual if np.isfinite(gpu_residual) else None,
        "reference_true_residual": ref_residual if np.isfinite(ref_residual) else None,
        "gpu_boundary_correct": gpu_boundary_correct,
        "reference_boundary_correct": reference_boundary_correct,
        "max_error_threshold": args.max_error,
        "residual_threshold": args.tolerance,
        "passed": passed,
    }, indent=2, allow_nan=False))

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
