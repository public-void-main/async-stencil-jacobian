"""Audit measured results before plotting, ML training, or reporting."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_experiments import HISTORY_COLUMNS, RUN_COLUMNS
from .spec import DEFAULT_SPEC_PATH, ExperimentSpec, load_spec


def audit_runs(
    runs_path: Path,
    spec: ExperimentSpec,
    histories_path: Path | None = None,
) -> dict[str, object]:
    runs = pd.read_csv(runs_path)
    if list(runs.columns) != RUN_COLUMNS:
        raise ValueError(f"runs.csv columns must be exactly {RUN_COLUMNS}, got {list(runs.columns)}")

    key_columns = ["N", "source", "r", "strategy", "run_id"]
    if runs.duplicated(key_columns).any():
        duplicates = runs.loc[runs.duplicated(key_columns, keep=False), key_columns]
        raise ValueError(f"duplicate experiment keys found:\n{duplicates.to_string(index=False)}")
    if len(runs) != spec.expected_runs:
        raise ValueError(f"expected {spec.expected_runs} runs, found {len(runs)}")

    expected = set(
        itertools.product(
            spec.grid_sizes,
            spec.source_types,
            spec.r_values,
            spec.strategies,
            range(1, spec.repeats + 1),
        )
    )
    actual = {
        (int(row.N), str(row.source), float(row.r), str(row.strategy), int(row.run_id))
        for row in runs.itertuples(index=False)
    }
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"matrix mismatch: {len(missing)} missing and {len(unexpected)} unexpected keys; "
            f"first missing={next(iter(missing), None)}, first unexpected={next(iter(unexpected), None)}"
        )

    numeric = runs[["iters", "time_ms", "final_residual"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("non-finite numeric result detected")
    if (runs["iters"] <= 0).any() or (runs["time_ms"] <= 0).any():
        raise ValueError("iterations and time_ms must be positive")
    if (runs["final_residual"] < 0).any():
        raise ValueError("final_residual cannot be negative")
    failed = runs[runs["final_residual"] >= spec.tolerance]
    if not failed.empty:
        raise ValueError(f"{len(failed)} runs did not meet tolerance {spec.tolerance:g}")

    timing = runs.groupby(["N", "source", "r", "strategy"])["time_ms"].agg(["mean", "std"])
    coefficient_of_variation = (timing["std"] / timing["mean"]).replace([np.inf, -np.inf], np.nan)
    noisy = coefficient_of_variation[coefficient_of_variation > 0.10]
    warnings: list[str] = []
    observed_curves = 0
    if histories_path is None or not histories_path.exists() or histories_path.stat().st_size == 0:
        warnings.append("no residual histories were recorded; convergence-curve output will be unavailable")
    else:
        histories = pd.read_csv(histories_path)
        if list(histories.columns) != HISTORY_COLUMNS:
            raise ValueError(f"residual history columns must be exactly {HISTORY_COLUMNS}")
        if (histories["iteration"] <= 0).any() or (histories["residual"] < 0).any():
            raise ValueError("residual histories contain invalid iteration or residual values")
        curve_keys = ["N", "source", "r", "strategy", "run_id"]
        observed_curves = int(histories.groupby(curve_keys).ngroups)
        for key, curve in histories.groupby(curve_keys):
            iterations = curve["iteration"].to_numpy()
            if len(np.unique(iterations)) != len(iterations) or not np.all(np.diff(iterations) > 0):
                raise ValueError(f"residual history is not strictly increasing for {key}")
        if observed_curves < spec.expected_runs:
            warnings.append(
                f"residual histories cover {observed_curves}/{spec.expected_runs} runs; "
                "representative curves can be plotted but coverage is incomplete"
            )

    summary: dict[str, object] = {
        "status": "pass",
        "expected_runs": spec.expected_runs,
        "observed_runs": int(len(runs)),
        "configurations": int(len(spec.grid_sizes) * len(spec.source_types) * len(spec.r_values)),
        "strategies": list(spec.strategies),
        "repeats": spec.repeats,
        "max_final_residual": float(runs["final_residual"].max()),
        "noisy_groups_cv_over_10pct": int(len(noisy)),
        "residual_curves_observed": observed_curves,
        "residual_history_complete": observed_curves == spec.expected_runs,
        "warnings": warnings,
        "noisy_groups": [
            {"N": int(index[0]), "source": str(index[1]), "r": float(index[2]),
             "strategy": str(index[3]), "coefficient_of_variation": float(value)}
            for index, value in noisy.items()
        ],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--histories", type=Path, default=Path("results/residual_history.csv"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/audit.json"))
    args = parser.parse_args()
    summary = audit_runs(args.runs, load_spec(args.spec), args.histories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
