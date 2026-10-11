"""Run the fixed experiment matrix through external CUDA solver commands."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .problem import ProblemConfig, deterministic_seed

RUN_COLUMNS = ["N", "source", "r", "strategy", "run_id", "iters", "time_ms", "final_residual"]
HISTORY_COLUMNS = ["N", "source", "r", "strategy", "run_id", "iteration", "residual"]
STRATEGIES = ("sync", "D1", "D2", "D4", "D8")


@dataclass(frozen=True)
class SolverCommand:
    command: tuple[str, ...]
    parser: str = "auto"
    timeout_seconds: int = 7200


def _load_manifest(path: Path) -> dict[str, SolverCommand]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    commands = raw.get("commands", raw)
    missing = set(STRATEGIES) - set(commands)
    if missing:
        raise ValueError(f"manifest has no commands for: {sorted(missing)}")
    parsed: dict[str, SolverCommand] = {}
    for strategy, entry in commands.items():
        if isinstance(entry, list):
            command, parser_name, timeout = entry, "auto", 7200
        elif isinstance(entry, dict):
            command = entry.get("command")
            parser_name = str(entry.get("parser", "auto"))
            timeout = int(entry.get("timeout_seconds", 7200))
        else:
            raise ValueError(f"manifest entry for {strategy} must be a command list or object")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise ValueError(f"command for {strategy} must be a non-empty JSON list of strings")
        if parser_name not in {"auto", "json", "text"}:
            raise ValueError(f"unsupported parser for {strategy}: {parser_name}")
        if timeout <= 0:
            raise ValueError(f"timeout_seconds for {strategy} must be positive")
        parsed[strategy] = SolverCommand(tuple(command), parser_name, timeout)
    return parsed


def _existing_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["N"], row["source"], row["r"], row["strategy"], row["run_id"])
            for row in csv.DictReader(handle)
        }


def _append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _normalise_json_result(value: dict[str, Any]) -> dict[str, Any] | None:
    aliases = {
        "iterations": value.get("iterations", value.get("iters")),
        "time_ms": value.get("time_ms", value.get("gpu_time_ms")),
        "final_residual": value.get("final_residual", value.get("residual")),
        "converged": value.get("converged"),
        "residual_history": value.get("residual_history", []),
    }
    if aliases["converged"] is None and isinstance(value.get("status"), str):
        aliases["converged"] = value["status"].strip().lower() in {"converged", "tolerance reached"}
    required = ("iterations", "time_ms", "final_residual", "converged")
    return aliases if all(aliases[name] is not None for name in required) else None


def _parse_json_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (normalised := _normalise_json_result(value)) is not None:
            return normalised
    raise ValueError("solver did not print a supported JSON result object on stdout")


def _parse_text_result(stdout: str) -> dict[str, Any]:
    patterns = {
        "iterations": r"(?im)^\s*Iterations\s*:\s*(\d+)\s*$",
        "time_ms": r"(?im)^\s*(?:GPU time|Time)\s*:\s*([0-9.eE+-]+)\s*(?:ms)?\s*$",
        "final_residual": r"(?im)^\s*Final residual\s*:\s*([0-9.eE+-]+)\s*$",
        "status": r"(?im)^\s*Convergence status\s*:\s*(.+?)\s*$",
    }
    matches = {name: re.findall(pattern, stdout) for name, pattern in patterns.items()}
    if any(len(values) != 1 for values in matches.values()):
        counts = {name: len(values) for name, values in matches.items()}
        raise ValueError(
            "text output must contain exactly one result block per process; "
            f"matched fields: {counts}"
        )
    status = matches["status"][0].strip().upper()
    return {
        "iterations": int(matches["iterations"][0]),
        "time_ms": float(matches["time_ms"][0]),
        "final_residual": float(matches["final_residual"][0]),
        "converged": status == "TOLERANCE REACHED",
        "residual_history": [],
    }


def _parse_result(stdout: str, parser_name: str = "auto") -> dict[str, Any]:
    errors: list[str] = []
    if parser_name in {"auto", "json"}:
        try:
            return _parse_json_result(stdout)
        except ValueError as error:
            errors.append(str(error))
    if parser_name in {"auto", "text"}:
        try:
            return _parse_text_result(stdout)
        except ValueError as error:
            errors.append(str(error))
    raise ValueError("; ".join(errors))


def _backup_for_rerun(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}")
    path.replace(backup)
    print(f"backed up {path} to {backup}")


def _check_executable(command: list[str], root: Path) -> None:
    executable = command[0]
    candidate = Path(executable)
    if candidate.is_absolute() and candidate.exists():
        return
    if any(separator in executable for separator in ("/", "\\")):
        if (root / candidate).exists():
            return
        raise FileNotFoundError(f"solver executable does not exist: {root / candidate}")
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"solver executable is not on PATH: {executable}")


def run_matrix(
    manifest: Path,
    output: Path,
    histories: Path,
    *,
    sizes: list[int],
    sources: list[str],
    r_values: list[float],
    strategies: list[str],
    repeats: int,
    tolerance: float,
    residual_interval: int,
    max_iters: int,
    rerun: bool,
) -> None:
    commands = _load_manifest(manifest)
    if rerun:
        _backup_for_rerun(output)
        _backup_for_rerun(histories)
    completed = _existing_keys(output)
    root = manifest.resolve().parent

    for n, source, r_value, strategy, run_id in itertools.product(
        sizes, sources, r_values, strategies, range(1, repeats + 1)
    ):
        key = (str(n), source, f"{r_value:g}", strategy, str(run_id))
        if key in completed:
            print(f"skip existing: {key}")
            continue
        config = ProblemConfig(n, source, r_value)
        values = {
            "python": sys.executable,
            "root": str(root),
            "N": str(n),
            "source": source,
            "r": f"{r_value:g}",
            "strategy": strategy,
            "D": strategy[1:] if strategy.startswith("D") else "0",
            "run_id": str(run_id),
            "seed": str(deterministic_seed(config, run_id)),
            "tolerance": f"{tolerance:.9g}",
            "residual_interval": str(residual_interval),
            "max_iters": str(max_iters),
        }
        solver = commands[strategy]
        command = [part.format(**values) for part in solver.command]
        _check_executable(command, root)
        print("run:", shlex.join(command), flush=True)
        completed_process = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=solver.timeout_seconds,
        )
        if completed_process.returncode != 0:
            raise RuntimeError(
                f"solver failed for {key}\nstdout:\n{completed_process.stdout}\n"
                f"stderr:\n{completed_process.stderr}"
            )
        result = _parse_result(completed_process.stdout, solver.parser)
        if not bool(result["converged"]):
            raise RuntimeError(f"solver did not converge for {key}: {result}")

        base = {"N": n, "source": source, "r": f"{r_value:g}", "strategy": strategy, "run_id": run_id}
        _append_rows(
            output,
            RUN_COLUMNS,
            [{**base, "iters": int(result["iterations"]), "time_ms": float(result["time_ms"]),
              "final_residual": float(result["final_residual"])}],
        )
        history_rows = [
            {**base, "iteration": int(iteration), "residual": float(residual)}
            for iteration, residual in result.get("residual_history", [])
        ]
        _append_rows(histories, HISTORY_COLUMNS, history_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--histories", type=Path, default=Path("results/residual_history.csv"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--sources", nargs="+", default=["point", "gaussian"])
    parser.add_argument("--r-values", type=float, nargs="+", default=[1, 10, 100])
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--residual-interval", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=2_000_000)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run_matrix(
        args.manifest, args.output, args.histories,
        sizes=args.sizes, sources=args.sources, r_values=args.r_values,
        strategies=args.strategies, repeats=args.repeats, tolerance=args.tolerance,
        residual_interval=args.residual_interval, max_iters=args.max_iters, rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
