"""Export byte-exact float32 inputs for CUDA integration and debugging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .problem import ProblemConfig, make_problem


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_problem(config: ProblemConfig, output_dir: Path) -> dict[str, object]:
    grid, source = make_problem(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "initial_grid.f32"
    source_path = output_dir / "dt_times_source.f32"
    grid.tofile(grid_path)
    source.tofile(source_path)
    metadata: dict[str, object] = {
        "N": config.N,
        "source": config.source,
        "r": config.r,
        "dtype": "float32 little-endian",
        "layout": "row-major N x N",
        "initial_grid": grid_path.name,
        "initial_grid_sha256": _digest(grid_path),
        "dt_times_source": source_path.name,
        "dt_times_source_sha256": _digest(source_path),
    }
    (output_dir / "problem.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--source", choices=["point", "gaussian"], required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_problem(
        ProblemConfig(args.N, args.source, args.r), args.output_dir
    ), indent=2))


if __name__ == "__main__":
    main()
