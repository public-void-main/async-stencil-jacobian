"""Regressions for rounded-update convergence versus equation accuracy."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from harness_ml.problem import ProblemConfig, make_problem
from harness_ml.reference import grid_after_iterations, solve_reference, true_residual
from harness_ml.solver_cli import main as solver_main
from harness_ml.validate_converged_grid import main as validator_main


class ReferencePrecisionTests(unittest.TestCase):
    def test_float32_stagnation_is_reported_separately(self):
        config = ProblemConfig(3, "point", 1, source_strength=1e-4)
        result = solve_reference(config, tolerance=1e-8, residual_interval=1)
        explicit = solve_reference(
            config, tolerance=1e-8, residual_interval=1, precision="float32"
        )
        self.assertTrue(result.converged)  # Existing rounded-update semantics.
        self.assertEqual(result.iterations, 12)
        self.assertEqual(result.final_residual, 0.0)
        self.assertGreater(result.final_true_residual, 1e-8)
        self.assertEqual(result.precision, "float32")
        self.assertEqual(result.final_grid.dtype, np.float32)
        np.testing.assert_array_equal(result.final_grid, explicit.final_grid)
        np.testing.assert_array_equal(
            result.final_grid, grid_after_iterations(config, result.iterations)
        )

    def test_float64_reference_meets_independent_criterion(self):
        config = ProblemConfig(3, "point", 1, source_strength=1e-4)
        result = solve_reference(
            config, tolerance=1e-8, residual_interval=1, precision="float64"
        )
        self.assertTrue(result.converged)
        self.assertLess(result.final_residual, 1e-8)
        self.assertLess(result.final_true_residual, 1e-8)
        self.assertEqual(result.final_grid.dtype, np.float64)
        initial, source = make_problem(config)
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        np.testing.assert_array_equal(result.final_grid[mask], initial[mask])
        exact_center = (100.0 + float(source[1, 1])) / 4.0
        self.assertAlmostEqual(result.final_grid[1, 1], exact_center, places=8)

    def test_float64_rejects_failed_independent_check(self):
        with patch("harness_ml.reference.true_residual", return_value=1.0):
            result = solve_reference(
                ProblemConfig(3, "point", 1), tolerance=100,
                residual_interval=1, max_iters=3, precision="float64",
            )
        # An above-tolerance checker must actually reject the candidate.
        with patch("harness_ml.reference.true_residual", return_value=100.0):
            rejected = solve_reference(
                ProblemConfig(3, "point", 1), tolerance=100,
                residual_interval=1, max_iters=3, precision="float64",
            )
        self.assertTrue(result.converged)
        self.assertFalse(rejected.converged)
        self.assertEqual(rejected.iterations, 3)

    def test_precision_argument_validation(self):
        with self.assertRaises(ValueError):
            solve_reference(ProblemConfig(3, "point", 1), precision="float16")

    def validate(self, grid, reference, tolerance, max_error):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("gpu.npy", "reference.npy")]
            np.save(paths[0], grid)
            np.save(paths[1], reference)
            args = [
                "validate", "--gpu-grid", str(paths[0]),
                "--reference-grid", str(paths[1]), "--N", str(grid.shape[0]),
                "--source", "point", "--r", "1",
                "--tolerance", str(tolerance), "--max-error", str(max_error),
            ]
            output = io.StringIO()
            status = 0
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(output):
                try:
                    validator_main()
                except SystemExit as error:
                    status = error.code
            return status, json.loads(output.getvalue())

    def test_identical_stalled_grid_fails_residual_validation(self):
        result = solve_reference(
            ProblemConfig(32, "point", 1), tolerance=1e-6, residual_interval=10
        )
        self.assertTrue(result.converged)
        status, metrics = self.validate(result.final_grid, result.final_grid, 1e-6, 1e-4)
        self.assertEqual(status, 1)
        self.assertEqual(metrics["max_abs_error"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertGreater(metrics["gpu_true_residual"], 1e-6)
        self.assertFalse(metrics["passed"])

    def test_validator_preserves_float64_reference(self):
        # Both grids satisfy the residual threshold, but downcasting would erase
        # this difference and incorrectly pass the tighter grid-error threshold.
        reference, _ = make_problem(ProblemConfig(3, "point", 1))
        reference = reference.astype(np.float64)
        reference[1, 1] = 25.0 + 1e-7
        grid = reference.astype(np.float32)
        status, metrics = self.validate(grid, reference, 1e-4, 1e-8)
        self.assertEqual(status, 1)
        self.assertAlmostEqual(metrics["max_abs_error"], 1e-7, places=12)
        self.assertGreater(metrics["reference_true_residual"], 0)

    def test_float64_reference_passes_validator_without_downcast(self):
        result = solve_reference(
            ProblemConfig(32, "point", 1), tolerance=1e-9,
            residual_interval=10, precision="float64",
        )
        self.assertTrue(result.converged)
        status, metrics = self.validate(result.final_grid, result.final_grid, 1e-9, 1e-4)
        self.assertEqual(status, 0)
        self.assertTrue(metrics["passed"])

    def test_nonfinite_grids_are_rejected_including_corners(self):
        grid = solve_reference(ProblemConfig(3, "point", 1)).final_grid
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                bad = grid.copy()
                bad[0, 0] = value
                self.assertEqual(true_residual(bad, np.zeros_like(bad), 1), float("inf"))
                status, metrics = self.validate(bad, grid, 1e-4, 1e-4)
                self.assertEqual(status, 1)
                self.assertFalse(metrics["passed"])

    def test_cli_reports_precision_and_both_residuals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npy"
            args = [
                "solver", "--N", "3", "--source", "point", "--r", "1",
                "--precision", "float64", "--tolerance", "1e-9",
                "--residual-interval", "1", "--grid-output", str(path),
            ]
            output = io.StringIO()
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(output):
                solver_main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["precision"], "float64")
            self.assertTrue(report["converged"])
            self.assertLess(report["final_residual"], 1e-9)
            self.assertLess(report["final_true_residual"], 1e-9)
            self.assertEqual(np.load(path).dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
