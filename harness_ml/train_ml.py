"""Train and evaluate the offline synchronization strategy selector."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = ["N", "source", "r"]


def build_dataset(runs: pd.DataFrame, near_tie_fraction: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = runs.groupby(["N", "source", "r", "strategy"], as_index=False)["time_ms"].median()
    pivot = med.pivot(index=["N", "source", "r"], columns="strategy", values="time_ms").reset_index()
    expected = {"sync", "D1", "D2", "D4", "D8"}
    missing = expected - set(pivot.columns)
    if missing or pivot[list(expected)].isna().any().any():
        raise ValueError(f"dataset is incomplete; missing strategy values: {sorted(missing)}")
    times = pivot[sorted(expected)]
    pivot["label"] = times.idxmin(axis=1)
    ordered = np.sort(times.to_numpy(), axis=1)
    pivot["near_tie"] = ordered[:, 1] <= ordered[:, 0] * (1.0 + near_tie_fraction)
    labels = pivot[[*FEATURES, "label", "near_tie"]].copy()
    return labels, pivot


def _model() -> Pipeline:
    preprocess = ColumnTransformer(
        [("source", OneHotEncoder(handle_unknown="ignore"), ["source"])],
        remainder="passthrough",
    )
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=52)
    return Pipeline([("preprocess", preprocess), ("tree", tree)])


def train_and_evaluate(
    runs_path: Path,
    output_dir: Path,
    *,
    near_tie_fraction: float = 0.02,
) -> dict[str, object]:
    runs = pd.read_csv(runs_path)
    labels, pivot = build_dataset(runs, near_tie_fraction=near_tie_fraction)
    x = labels[FEATURES]
    y = labels["label"]
    predictions: list[str] = []

    # Leave-one-configuration-out evaluation, equivalent to LOOCV here.
    for held_out in range(len(labels)):
        train_mask = np.arange(len(labels)) != held_out
        fold_model = _model()
        fold_model.fit(x.loc[train_mask], y.loc[train_mask])
        predictions.append(str(fold_model.predict(x.iloc[[held_out]])[0]))

    labels["predicted_strategy"] = predictions
    strategy_columns = ["sync", "D1", "D2", "D4", "D8"]
    oracle_times = pivot[strategy_columns].min(axis=1).to_numpy(dtype=float)
    predicted_times = np.array([pivot.loc[i, strategy] for i, strategy in enumerate(predictions)], dtype=float)
    fixed_totals = {strategy: float(pivot[strategy].sum()) for strategy in strategy_columns}
    best_fixed = min(fixed_totals, key=fixed_totals.get)
    oracle_total = float(oracle_times.sum())
    near_optimal = predicted_times <= oracle_times * (1.0 + near_tie_fraction)

    labels["oracle_time_ms"] = oracle_times
    labels["predicted_time_ms"] = predicted_times
    labels["prediction_regret_pct"] = 100.0 * (predicted_times - oracle_times) / oracle_times
    labels["prediction_within_tie_band"] = near_optimal

    metrics: dict[str, object] = {
        "configurations": int(len(labels)),
        "loocv_accuracy": float((labels.label == labels.predicted_strategy).mean()),
        "loocv_near_optimal_accuracy": float(near_optimal.mean()),
        "oracle_total_time_ms": oracle_total,
        "ml_total_time_ms": float(predicted_times.sum()),
        "ml_regret_pct": float(100.0 * (predicted_times.sum() - oracle_total) / oracle_total),
        "always_sync_total_time_ms": fixed_totals["sync"],
        "always_sync_regret_pct": float(100.0 * (fixed_totals["sync"] - oracle_total) / oracle_total),
        "best_fixed_strategy": best_fixed,
        "best_fixed_total_time_ms": fixed_totals[best_fixed],
        "best_fixed_regret_pct": float(100.0 * (fixed_totals[best_fixed] - oracle_total) / oracle_total),
        "near_tie_configurations": int(labels.near_tie.sum()),
        "near_tie_fraction": near_tie_fraction,
    }

    final_model = _model()
    final_model.fit(x, y)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_dir / "strategy_labels_and_predictions.csv", index=False)
    (output_dir / "ml_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "decision_tree.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    transformed_names = final_model.named_steps["preprocess"].get_feature_names_out()
    tree_text = export_text(final_model.named_steps["tree"], feature_names=list(transformed_names))
    (output_dir / "decision_tree.txt").write_text(tree_text, encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ml"))
    parser.add_argument("--near-tie-fraction", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(
        args.runs, args.output_dir, near_tie_fraction=args.near_tie_fraction
    ), indent=2))


if __name__ == "__main__":
    main()
