"""One-command execution of experiments, plots, ML, and report summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_runs
from .plots import generate_plots
from .reproducibility import write_metadata
from .run_experiments import run_matrix
from .spec import DEFAULT_SPEC_PATH, load_spec
from .summarize import write_summary
from .train_ml import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    runs = args.results_dir / "runs.csv"
    histories = args.results_dir / "residual_history.csv"
    write_metadata(args.spec, args.manifest, args.results_dir / "reproducibility.json")
    run_matrix(
        args.manifest, runs, histories,
        sizes=list(spec.grid_sizes), sources=list(spec.source_types), r_values=list(spec.r_values),
        strategies=list(spec.strategies), repeats=spec.repeats, tolerance=spec.tolerance,
        residual_interval=spec.residual_interval, max_iters=spec.max_iters, rerun=args.rerun,
    )
    audit = audit_runs(runs, spec, histories)
    (args.results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    generate_plots(runs, histories, args.results_dir / "figures")
    train_and_evaluate(runs, args.results_dir / "ml", near_tie_fraction=spec.near_tie_fraction)
    write_summary(runs, args.results_dir / "ml" / "ml_metrics.json", Path("report/vaibhav_sections.md"))


if __name__ == "__main__":
    main()
