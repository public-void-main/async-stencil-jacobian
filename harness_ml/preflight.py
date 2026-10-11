"""Check solver-manifest paths and placeholders without launching GPU work."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .problem import ProblemConfig, deterministic_seed
from .run_experiments import _check_executable, _load_manifest
from .spec import DEFAULT_SPEC_PATH, load_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    solvers = _load_manifest(args.manifest)
    root = args.manifest.resolve().parent
    config = ProblemConfig(spec.grid_sizes[0], spec.source_types[0], spec.r_values[0])
    values = {
        "python": sys.executable,
        "root": str(root),
        "N": str(config.N),
        "source": config.source,
        "r": f"{config.r:g}",
        "strategy": "sync",
        "D": "0",
        "run_id": "1",
        "seed": str(deterministic_seed(config, 1)),
        "tolerance": f"{spec.tolerance:.9g}",
        "residual_interval": str(spec.residual_interval),
        "max_iters": str(spec.max_iters),
    }
    for strategy in spec.strategies:
        strategy_values = dict(values)
        strategy_values["strategy"] = strategy
        strategy_values["D"] = strategy[1:] if strategy.startswith("D") else "0"
        command = [part.format(**strategy_values) for part in solvers[strategy].command]
        _check_executable(command, root)
        print(f"PASS {strategy}: parser={solvers[strategy].parser}, executable={command[0]}")


if __name__ == "__main__":
    main()
