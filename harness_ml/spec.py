"""Load and validate the single source of truth for all experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_SPEC_PATH = Path(__file__).with_name("experiment_spec.json")


@dataclass(frozen=True)
class ExperimentSpec:
    grid_sizes: tuple[int, ...]
    source_types: tuple[str, ...]
    r_values: tuple[float, ...]
    strategies: tuple[str, ...]
    repeats: int
    base_seed: int
    tolerance: float
    residual_interval: int
    max_iters: int
    near_tie_fraction: float
    raw: dict[str, object]

    @property
    def expected_runs(self) -> int:
        return (
            len(self.grid_sizes)
            * len(self.source_types)
            * len(self.r_values)
            * len(self.strategies)
            * self.repeats
        )


def load_spec(path: Path = DEFAULT_SPEC_PATH) -> ExperimentSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "grid_sizes", "source_types", "r_values", "strategies", "repeats",
        "base_seed", "tolerance", "residual_interval", "max_iters", "near_tie_fraction",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"experiment specification is missing: {sorted(missing)}")
    spec = ExperimentSpec(
        grid_sizes=tuple(int(value) for value in raw["grid_sizes"]),
        source_types=tuple(str(value) for value in raw["source_types"]),
        r_values=tuple(float(value) for value in raw["r_values"]),
        strategies=tuple(str(value) for value in raw["strategies"]),
        repeats=int(raw["repeats"]),
        base_seed=int(raw["base_seed"]),
        tolerance=float(raw["tolerance"]),
        residual_interval=int(raw["residual_interval"]),
        max_iters=int(raw["max_iters"]),
        near_tie_fraction=float(raw["near_tie_fraction"]),
        raw=raw,
    )
    if spec.grid_sizes != (512, 1024):
        raise ValueError("project scope requires grid sizes 512 and 1024")
    if set(spec.source_types) != {"point", "gaussian"}:
        raise ValueError("project scope requires point and gaussian sources")
    if set(spec.strategies) != {"sync", "D1", "D2", "D4", "D8"}:
        raise ValueError("project scope requires sync and D in {1,2,4,8}")
    if spec.repeats < 1 or spec.tolerance <= 0 or spec.residual_interval < 1:
        raise ValueError("invalid repeat, tolerance, or residual interval setting")
    return spec
