"""Focused CPU regressions for validation, cap reporting and strict JSON."""

import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

from harness_ml.problem import ProblemConfig, make_problem
from harness_ml.reference import grid_after_iterations, solve_reference, true_residual
from harness_ml.solver_cli import main as solver_main
from harness_ml.validate_converged_grid import main as validator_main


class ValidationHardeningTests(unittest.TestCase):
    def valid_grid(self):
        grid, _ = make_problem(ProblemConfig(3, "point", 1))
        grid = grid.astype(np.float64)
        grid[1, 1] = 25.25
        return grid

    def cli(self, main, args, grids=None):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["test", *args]))
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(errors))
            if grids is not None:
                stack.enter_context(patch(
                    "harness_ml.validate_converged_grid.np.load", side_effect=grids
                ))
            status = 0
            try:
                main()
            except SystemExit as error:
                status = error.code
        def reject_constant(token):
            self.fail(f"Non-standard JSON constant: {token}")
        report = json.loads(output.getvalue(), parse_constant=reject_constant) \
                 if output.getvalue() else None
        return status, report, errors.getvalue()

    def validate(self, candidate, reference, extra=()):
        return self.cli(validator_main, [
            "--gpu-grid", "unused", "--reference-grid", "unused",
            "--N", "3", "--source", "point", "--r", "1", *extra,
        ], [candidate, reference])

    def test_wrong_boundary_self_comparison_fails(self):
        wrong = np.zeros((3, 3), dtype=np.float64)
        wrong[1, 1] = 0.25  # Exact solution of the wrong boundary problem.
        status, report, _ = self.validate(wrong, wrong)
        self.assertEqual(status, 1)
        self.assertEqual(report["max_abs_error"], 0)
        self.assertEqual(report["gpu_true_residual"], 0)
        self.assertFalse(report["gpu_boundary_correct"])
        self.assertFalse(report["reference_boundary_correct"])
        self.assertFalse(report["passed"])

    def test_each_boundary_including_top_corners_is_enforced(self):
        good = self.valid_grid()
        for cell in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)):
            for side in (0, 1):
                with self.subTest(cell=cell, side=side):
                    bad = good.copy()
                    bad[cell] += 1e-6
                    grids = [good, good]
                    grids[side] = bad
                    status, report, _ = self.validate(*grids)
                    self.assertEqual(status, 1)
                    field = "gpu_boundary_correct" if side == 0 else "reference_boundary_correct"
                    self.assertFalse(report[field])

    def test_correct_boundaries_and_zero_error_threshold_pass(self):
        good = self.valid_grid()
        status, report, _ = self.validate(good, good, ["--max-error", "0"])
        self.assertEqual(status, 0)
        self.assertTrue(report["gpu_boundary_correct"])
        self.assertTrue(report["reference_boundary_correct"])

    def test_solver_rejects_invalid_numeric_parameters(self):
        config = ProblemConfig(3, "point", 1)
        for value in (np.nan, np.inf, -np.inf, 0, -1, True, 1j):
            with self.subTest(tolerance=value), self.assertRaises(ValueError):
                solve_reference(config, tolerance=value, max_iters=1)
        for name in ("max_iters", "residual_interval"):
            for value in (np.nan, np.inf, 0, -1, 1.5, 2.0, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    solve_reference(config, **{name: value})

    def test_problem_rejects_invalid_configuration_and_source(self):
        for value in (np.nan, np.inf, -np.inf, 0, -1, True, 1j):
            with self.subTest(r=value), self.assertRaises(ValueError):
                make_problem(ProblemConfig(3, "point", value))
        for value in (np.nan, np.inf, -np.inf, 1e40):
            for source in ("point", "gaussian"):
                with self.subTest(strength=value, source=source), self.assertRaises(ValueError):
                    make_problem(ProblemConfig(3, source, 1, source_strength=value))
        for value in (2, 3.0, np.nan, np.inf, True):
            with self.subTest(N=value), self.assertRaises(ValueError):
                make_problem(ProblemConfig(value, "point", 1))
        for config in (ProblemConfig(3, "other", 1), ProblemConfig(3, "point", 1, boundary="other")):
            with self.assertRaises(ValueError):
                make_problem(config)

    def test_nonfinite_sources_rejected_before_solve_and_residual(self):
        initial, source = make_problem(ProblemConfig(3, "point", 1))
        for value in (np.nan, np.inf, -np.inf):
            bad = source.copy()
            bad[1, 1] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    true_residual(initial, bad, 1)
                with patch("harness_ml.reference.make_problem", return_value=(initial, bad)):
                    with self.assertRaises(ValueError):
                        solve_reference(ProblemConfig(3, "point", 1))

    def test_grid_and_source_shapes_and_dtypes_rejected(self):
        good = self.valid_grid()
        invalid = (
            np.zeros(3), np.zeros((3, 3, 1)), np.zeros((3, 4)), np.zeros((2, 2)),
            good.astype(complex), good.astype(int), good.astype(np.float16),
            good.astype(object), good.astype(str),
        )
        for bad in invalid:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                for grid, source in ((bad, good), (good, bad)):
                    with self.assertRaises(ValueError):
                        true_residual(grid, source, 1)
                for grids in ((bad, good), (good, bad)):
                    status, _, _ = self.validate(*grids)
                    self.assertNotEqual(status, 0)
        with self.assertRaises(ValueError):
            true_residual(good, np.zeros((4, 4), dtype=np.float64), 1)

    def test_validator_rejects_invalid_parameters(self):
        good = self.valid_grid()
        for name in ("--tolerance", "--max-error", "--r"):
            values = ("nan", "inf", "-inf", "-1")
            if name != "--max-error":
                values += ("0",)
            for value in values:
                with self.subTest(name=name, value=value):
                    status, report, _ = self.validate(good, good, [f"{name}={value}"])
                    self.assertEqual(status, 2)
                    self.assertIsNone(report)

    def test_unrepresentable_diffusion_parameter_rejected(self):
        with self.assertRaises(ValueError):
            solve_reference(ProblemConfig(3, "point", 1e40), max_iters=1)
        with self.assertRaises(ValueError):
            grid_after_iterations(ProblemConfig(3, "point", 1e40), 1)
        good = self.valid_grid()
        status, report, _ = self.validate(good, good, ["--r", "1e308"])
        self.assertEqual(status, 2)
        self.assertIsNone(report)

    def test_solver_cli_rejects_invalid_parameters(self):
        base = ["--N", "3", "--source", "point", "--r", "1", "--max-iters", "1"]
        for name, value in (("--tolerance", "nan"), ("--tolerance", "inf"),
                            ("--r", "nan"), ("--r", "inf"),
                            ("--max-iters", "0"), ("--residual-interval", "0"),
                            ("--residual-interval", "1.5")):
            with self.subTest(name=name, value=value):
                status, report, errors = self.cli(solver_main, [*base, name, value])
                self.assertEqual(status, 2)
                self.assertIsNone(report)
                self.assertIn("error:", errors)

    def test_nonfinite_grids_produce_strict_failure_json(self):
        good = self.valid_grid()
        for value in (np.nan, np.inf, -np.inf):
            for side in (0, 1):
                with self.subTest(value=value, side=side):
                    bad = good.copy()
                    bad[1, 1] = value
                    grids = [good, good]
                    grids[side] = bad
                    status, report, _ = self.validate(*grids)
                    self.assertEqual(status, 1)
                    self.assertFalse(report["passed"])
                    field = "gpu_true_residual" if side == 0 else "reference_true_residual"
                    self.assertIsNone(report[field])

    def test_natural_caps_and_strict_cli_json(self):
        config = ProblemConfig(3, "point", 1)
        for precision in ("float32", "float64"):
            for cap, sampled in ((1, None), (3, 2), (4, 4)):
                with self.subTest(precision=precision, cap=cap):
                    result = solve_reference(
                        config, precision=precision, tolerance=1e-12,
                        residual_interval=2, max_iters=cap,
                    )
                    self.assertFalse(result.converged)
                    self.assertEqual(result.iterations, cap)
                    self.assertEqual(result.final_residual_iteration, sampled)
                    if sampled is None:
                        self.assertEqual(result.final_residual, float("inf"))
                    else:
                        self.assertEqual(result.final_residual, result.residual_history[-1][1])
                    _, source = make_problem(config)
                    self.assertEqual(result.final_true_residual, true_residual(result.final_grid, source, 1))
                    status, report, _ = self.cli(solver_main, [
                        "--N", "3", "--source", "point", "--r", "1",
                        "--precision", precision, "--tolerance", "1e-12",
                        "--residual-interval", "2", "--max-iters", str(cap),
                    ])
                    self.assertEqual(status, 0)  # Existing CLI status semantics.
                    self.assertFalse(report["converged"])
                    self.assertEqual(report["final_residual_iteration"], sampled)
                    if sampled is None:
                        self.assertIsNone(report["final_residual"])
                    else:
                        self.assertEqual(report["final_residual"], result.final_residual)

    def test_valid_float32_history_and_grid_are_unchanged(self):
        for source in ("point", "gaussian"):
            for r in (1, 100):
                config = ProblemConfig(8, source, r)
                result = solve_reference(config, tolerance=1e-20, residual_interval=7, max_iters=25)
                self.assertFalse(result.converged)
                self.assertEqual(result.iterations, 25)
                self.assertEqual(result.final_residual_iteration, 21)
                np.testing.assert_array_equal(result.final_grid, grid_after_iterations(config, 25))
                expected = []
                for iteration in (7, 14, 21):
                    old = grid_after_iterations(config, iteration - 1)
                    new = grid_after_iterations(config, iteration)
                    expected.append((iteration, float(np.max(np.abs(new[1:-1, 1:-1] - old[1:-1, 1:-1])))))
                self.assertEqual(result.residual_history, expected)


if __name__ == "__main__":
    unittest.main()
