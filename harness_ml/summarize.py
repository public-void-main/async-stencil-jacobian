"""Create a report-ready Markdown summary from measured results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def write_summary(runs_path: Path, metrics_path: Path, output: Path) -> None:
    runs = pd.read_csv(runs_path)
    med = runs.groupby(["N", "source", "r", "strategy"], as_index=False).agg(
        median_time_ms=("time_ms", "median"),
        time_q25_ms=("time_ms", lambda values: values.quantile(0.25)),
        time_q75_ms=("time_ms", lambda values: values.quantile(0.75)),
        median_iters=("iters", "median"),
    )
    winners = med.loc[med.groupby(["N", "source", "r"])["median_time_ms"].idxmin()].copy()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    lines = [
        "# Vaibhav: Experimental Harness and ML Selection",
        "",
        "## Reproducibility",
        "",
        f"The audited dataset contains {len(runs)} measured runs. The experiment matrix uses "
        "N in {512, 1024}, point and Gaussian sources, "
        "r in {1, 10, 100}, five synchronization strategies, and three repeats. "
        "Fixed seeds are generated deterministically from the configuration and repeat index and "
        "passed to both CUDA executables. Median time is the primary metric; the plots show the "
        "interquartile range across repeats.",
        "",
        "## Fastest strategy by configuration",
        "",
        winners.to_markdown(index=False),
        "",
        "## ML-guided selection",
        "",
        f"A depth-limited decision tree was evaluated with leave-one-configuration-out "
        f"cross-validation across {metrics['configurations']} configurations. Its strategy-selection "
        f"accuracy was {100 * metrics['loocv_accuracy']:.1f}% and its total-time regret relative to "
        f"the per-configuration oracle was {metrics['ml_regret_pct']:.2f}%.",
        "",
        f"When predictions within the {100 * metrics['near_tie_fraction']:.1f}% timing-noise band "
        f"are counted as equivalent, accuracy was {100 * metrics['loocv_near_optimal_accuracy']:.1f}%.",
        "",
        f"Always selecting synchronous execution had {metrics['always_sync_regret_pct']:.2f}% regret. "
        f"The best fixed strategy was {metrics['best_fixed_strategy']}, with "
        f"{metrics['best_fixed_regret_pct']:.2f}% regret.",
        "",
        "## Limitations",
        "",
        "The classifier is a proof of concept trained on a small, single-GPU dataset. "
        "Its conclusions should not be generalized to other GPUs, boundary conditions, source families, "
        "or parameter ranges without additional measurements. Boundary type is not an ML feature because "
        "it is constant (Dirichlet) in the reduced scope. Near-ties are sensitive to timing noise.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("results/ml/ml_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("report/vaibhav_sections.md"))
    args = parser.parse_args()
    write_summary(args.runs, args.metrics, args.output)


if __name__ == "__main__":
    main()
