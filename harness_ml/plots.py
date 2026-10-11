"""Generate the required performance and convergence figures."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "async-stencil-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_ORDER = ["sync", "D1", "D2", "D4", "D8"]


def _load_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    required = {"N", "source", "r", "strategy", "run_id", "iters", "time_ms", "final_residual"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"runs CSV is missing columns: {sorted(missing)}")
    if runs.duplicated(["N", "source", "r", "strategy", "run_id"]).any():
        raise ValueError("runs CSV contains duplicate experiment keys")
    return runs


def generate_plots(runs_path: Path, histories_path: Path, output_dir: Path) -> None:
    runs = _load_runs(runs_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    med = runs.groupby(["N", "source", "r", "strategy"], as_index=False).agg(
        time_ms=("time_ms", "median"),
        time_q25_ms=("time_ms", lambda values: values.quantile(0.25)),
        time_q75_ms=("time_ms", lambda values: values.quantile(0.75)),
        iters=("iters", "median"),
    )
    med.to_csv(output_dir / "timing_summary.csv", index=False)

    labels = med.assign(config=med.apply(lambda x: f"N={x.N:g}, {x.source}, r={x.r:g}", axis=1))
    pivot = labels.pivot(index="config", columns="strategy", values="time_ms").reindex(columns=STRATEGY_ORDER)
    lower = labels.pivot(index="config", columns="strategy", values="time_q25_ms").reindex_like(pivot)
    upper = labels.pivot(index="config", columns="strategy", values="time_q75_ms").reindex_like(pivot)
    errors = np.stack(
        [(pivot - lower).to_numpy().T, (upper - pivot).to_numpy().T],
        axis=1,
    )
    ax = pivot.plot(kind="bar", figsize=(13, 6), width=0.82, yerr=errors, capsize=2)
    ax.set_ylabel("Median time to convergence (ms)")
    ax.set_xlabel("Problem configuration")
    ax.set_title("Time to convergence by synchronization strategy (median and IQR)")
    ax.legend(title="Strategy", ncol=5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_strategy.png", dpi=180)
    plt.close()

    sync = med[med.strategy == "sync"][["N", "source", "r", "time_ms", "iters"]].rename(
        columns={"time_ms": "sync_time_ms", "iters": "sync_iters"}
    )
    comparison = med.merge(sync, on=["N", "source", "r"], how="left")
    comparison["speedup"] = comparison["sync_time_ms"] / comparison["time_ms"]
    comparison["iteration_ratio"] = comparison["iters"] / comparison["sync_iters"]
    comparison.to_csv(output_dir / "speedup_iteration_ratio.csv", index=False)

    async_rows = comparison[comparison.strategy != "sync"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for strategy in STRATEGY_ORDER[1:]:
        group = async_rows[async_rows.strategy == strategy]
        axes[0].scatter(group["iteration_ratio"], group["speedup"], label=strategy, s=45)
    axes[0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0].axvline(1.0, color="black", linewidth=1, linestyle=":")
    axes[0].set_xlabel("Iteration ratio $I_{async}/I_{sync}$")
    axes[0].set_ylabel("Speedup $T_{sync}/T_{async}$")
    axes[0].set_title("Synchronization trade-off")
    axes[0].legend()

    winners = med.loc[med.groupby(["N", "source", "r"])["time_ms"].idxmin()]
    counts = winners.strategy.value_counts().reindex(STRATEGY_ORDER, fill_value=0)
    axes[1].bar(counts.index, counts.values)
    axes[1].set_ylabel("Number of configurations won")
    axes[1].set_title("Fastest strategy by median time")
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_and_winners.png", dpi=180)
    plt.close()

    if histories_path.exists() and histories_path.stat().st_size:
        histories = pd.read_csv(histories_path)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
        selected = histories[(histories.run_id == histories.run_id.min()) & (histories.r == histories.r.min())]
        for ax, ((n, source), group) in zip(axes.flat, selected.groupby(["N", "source"])):
            for strategy in STRATEGY_ORDER:
                curve = group[group.strategy == strategy]
                if not curve.empty:
                    ax.semilogy(curve.iteration, curve.residual, label=strategy)
            ax.set_title(f"N={n}, {source}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Max update residual")
            ax.grid(alpha=0.25)
            ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "residual_curves.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--histories", type=Path, default=Path("results/residual_history.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    generate_plots(args.runs, args.histories, args.output_dir)


if __name__ == "__main__":
    main()
