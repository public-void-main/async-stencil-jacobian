from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness_ml.audit import audit_runs
from harness_ml.export_problem import export_problem
from harness_ml.problem import ProblemConfig, deterministic_seed, make_problem
from harness_ml.plots import generate_plots
from harness_ml.reference import grid_after_iterations, solve_reference
from harness_ml.run_experiments import STRATEGIES, _parse_result, run_matrix
from harness_ml.spec import ExperimentSpec, load_spec
from harness_ml.summarize import write_summary
from harness_ml.train_ml import build_dataset, train_and_evaluate


class ProblemTests(unittest.TestCase):
    def test_project_spec_has_180_runs(self) -> None:
        self.assertEqual(load_spec().expected_runs, 180)

    def test_sources_and_boundaries_are_deterministic(self) -> None:
        for source_name in ("point", "gaussian"):
            config = ProblemConfig(17, source_name, 1)
            grid_a, source_a = make_problem(config)
            grid_b, source_b = make_problem(config)
            np.testing.assert_array_equal(grid_a, grid_b)
            np.testing.assert_array_equal(source_a, source_b)
            np.testing.assert_array_equal(grid_a[0], np.full(17, 100, dtype=np.float32))
            self.assertTrue(np.all(source_a[[0, -1], :] == 0))
            self.assertTrue(np.all(source_a[:, [0, -1]] == 0))

    def test_seed_changes_by_repeat(self) -> None:
        config = ProblemConfig(32, "point", 1)
        self.assertNotEqual(deterministic_seed(config, 1), deterministic_seed(config, 2))

    def test_binary_export_matches_generator(self) -> None:
        config = ProblemConfig(9, "gaussian", 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_problem(config, root)
            grid, source = make_problem(config)
            np.testing.assert_array_equal(np.fromfile(root / "initial_grid.f32", dtype=np.float32), grid.ravel())
            np.testing.assert_array_equal(
                np.fromfile(root / "dt_times_source.f32", dtype=np.float32), source.ravel()
            )


class ReferenceTests(unittest.TestCase):
    def test_exact_iteration_reference_is_repeatable(self) -> None:
        config = ProblemConfig(10, "gaussian", 10)
        np.testing.assert_array_equal(
            grid_after_iterations(config, 25),
            grid_after_iterations(config, 25),
        )

    def test_reference_converges_and_keeps_boundaries(self) -> None:
        result = solve_reference(
            ProblemConfig(12, "point", 1, source_strength=0.1),
            tolerance=1e-3,
            residual_interval=5,
            max_iters=20_000,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.final_residual, 1e-3)
        np.testing.assert_array_equal(result.final_grid[0], np.full(12, 100, dtype=np.float32))
        self.assertTrue(np.all(result.final_grid[-1] == 0))


class MLTests(unittest.TestCase):
    def _runs(self) -> pd.DataFrame:
        rows = []
        strategies = ["sync", "D1", "D2", "D4", "D8"]
        for n in (32, 64):
            for source in ("point", "gaussian"):
                for r in (1, 10, 100):
                    winner = "sync" if r == 1 else ("D2" if source == "point" else "D4")
                    for strategy in strategies:
                        for run_id in (1, 2, 3):
                            base = 10.0 + strategies.index(strategy)
                            time_ms = 5.0 if strategy == winner else base
                            rows.append({
                                "N": n, "source": source, "r": r, "strategy": strategy,
                                "run_id": run_id, "iters": 100 + strategies.index(strategy),
                                "time_ms": time_ms + run_id * 0.01, "final_residual": 1e-6,
                            })
        return pd.DataFrame(rows)

    def test_labels_and_training_outputs(self) -> None:
        runs = self._runs()
        labels, _ = build_dataset(runs)
        self.assertEqual(len(labels), 12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_path = root / "runs.csv"
            runs.to_csv(runs_path, index=False)
            metrics = train_and_evaluate(runs_path, root / "ml")
            self.assertEqual(metrics["configurations"], 12)
            self.assertIn("loocv_near_optimal_accuracy", metrics)
            self.assertTrue((root / "ml" / "decision_tree.pkl").exists())
            self.assertTrue((root / "ml" / "ml_metrics.json").exists())

    def test_plots_and_report_summary(self) -> None:
        runs = self._runs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_path = root / "runs.csv"
            runs.to_csv(runs_path, index=False)
            figures = root / "figures"
            generate_plots(runs_path, root / "no_histories.csv", figures)
            self.assertTrue((figures / "time_vs_strategy.png").exists())
            self.assertTrue((figures / "speedup_and_winners.png").exists())
            self.assertTrue((figures / "speedup_iteration_ratio.csv").exists())
            self.assertTrue((figures / "timing_summary.csv").exists())
            train_and_evaluate(runs_path, root / "ml")
            summary = root / "vaibhav_sections.md"
            write_summary(runs_path, root / "ml" / "ml_metrics.json", summary)
            self.assertIn("proof of concept", summary.read_text(encoding="utf-8"))


class ExperimentRunnerTests(unittest.TestCase):
    def test_text_output_adapter(self) -> None:
        output = """
Iterations            : 1200
Final residual        : 9.2e-6
Convergence status    : TOLERANCE REACHED
GPU time              : 18.5 ms
"""
        result = _parse_result(output, "auto")
        self.assertEqual(result["iterations"], 1200)
        self.assertTrue(result["converged"])

    def test_runner_parses_json_and_resumes(self) -> None:
        result_code = (
            "import json; print(json.dumps({{\"iterations\": 20, \"time_ms\": 1.5, "
            "\"final_residual\": 1e-6, \"converged\": True, "
            "\"residual_history\": [[10, 0.1], [20, 1e-6]]}}))"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "solvers.json"
            manifest.write_text(
                json.dumps({"commands": {strategy: [sys.executable, "-c", result_code] for strategy in STRATEGIES}}),
                encoding="utf-8",
            )
            runs = root / "runs.csv"
            histories = root / "histories.csv"
            arguments = dict(
                sizes=[8], sources=["point"], r_values=[1], strategies=["sync"], repeats=1,
                tolerance=1e-5, residual_interval=10, max_iters=100, rerun=False,
            )
            run_matrix(manifest, runs, histories, **arguments)
            run_matrix(manifest, runs, histories, **arguments)
            self.assertEqual(len(pd.read_csv(runs)), 1)
            self.assertEqual(len(pd.read_csv(histories)), 2)

    def test_complete_dataset_audit(self) -> None:
        runs = MLTests()._runs()
        spec = ExperimentSpec(
            grid_sizes=(32, 64), source_types=("point", "gaussian"), r_values=(1.0, 10.0, 100.0),
            strategies=STRATEGIES, repeats=3, base_seed=52, tolerance=1e-5,
            residual_interval=10, max_iters=1000, near_tie_fraction=0.02, raw={},
        )
        with tempfile.TemporaryDirectory() as directory:
            runs_path = Path(directory) / "runs.csv"
            runs.to_csv(runs_path, index=False)
            summary = audit_runs(runs_path, spec)
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["observed_runs"], 180)


if __name__ == "__main__":
    unittest.main()
