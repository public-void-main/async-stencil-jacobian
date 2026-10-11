"""Compare a CUDA-produced NumPy grid with the CPU reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .problem import ProblemConfig
from .reference import grid_after_iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-grid", type=Path, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--source", choices=["point", "gaussian"], required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--max-error", type=float, default=1.0e-4)
    args = parser.parse_args()

    gpu = np.load(args.gpu_grid)
    expected = grid_after_iterations(ProblemConfig(args.N, args.source, args.r), args.iterations)
    if gpu.shape != expected.shape:
        raise SystemExit(f"shape mismatch: GPU {gpu.shape}, CPU {expected.shape}")
    difference = np.abs(gpu.astype(np.float64) - expected.astype(np.float64))
    metrics = {
        "max_abs_error": float(difference.max()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "passed": bool(difference.max() <= args.max_error),
    }
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
