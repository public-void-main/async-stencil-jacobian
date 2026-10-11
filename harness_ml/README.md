# Vaibhav's reference, experiment and ML harness

This folder implements the complete role-C pipeline without inventing GPU results.

## Scope and ownership

This code stays within Vaibhav's assigned role: CPU reference, deterministic
problem generation, experiment orchestration, dataset checks, plots, and the
offline ML selector. It does not implement or modify either CUDA kernel.

`experiment_spec.json` is the machine-readable source of truth for the equation,
boundaries, sources, experiment matrix, convergence settings and timing scope.
Teammates should review it once before integration so all three components use
identical definitions.

## Solver contract

Each CUDA command is configured in `solvers.json`; teammates do not have to use
fixed argument names. The preferred final stdout line is one JSON object:

```json
{"iterations": 1200, "time_ms": 18.5, "final_residual": 9.2e-6, "converged": true, "residual_history": [[10, 0.8], [20, 0.3]]}
```

For minimal changes to existing CUDA code, an automatic parser also accepts one
labelled text result block. See `INTEGRATION_CONTRACT.md` for the precise handoff.

The CUDA solvers and `reference.py` must implement:

`u_new = (u_old + r * neighbour_sum + dt_times_source) / (1 + 4*r)`

## Setup and validation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r harness_ml\requirements.txt
python -m unittest discover -s harness_ml\tests -v
```

For a small CPU smoke test:

```powershell
python -m harness_ml.solver_cli --N 32 --source point --r 1 --strategy sync --max-iters 5000
```

To validate a CUDA grid saved as NumPy `.npy` data after an exact iteration
count:

```powershell
python -m harness_ml.validate_grid --gpu-grid results\gpu_grid.npy --N 32 --source point --r 1 --iterations 100
```

If teammates need byte-identical inputs while debugging, export the initial grid
and source as row-major float32 files:

```powershell
python -m harness_ml.export_problem --N 512 --source gaussian --r 10 --output-dir results\integration_input
```

## Full experiment

Copy `harness_ml/solvers.example.json` to `solvers.json` and adjust executable
paths if required. Check paths and placeholders without launching GPU work:

```powershell
python -m harness_ml.preflight --manifest solvers.json
```

Then run everything with one command:

```powershell
python -m harness_ml.pipeline --manifest solvers.json
```

The experiment driver resumes safely by skipping keys already present in
`results/runs.csv`. `--rerun` first moves existing CSV files to timestamped
backups, so measurements are never silently destroyed or duplicated.

Before plots or ML are produced, the pipeline audits all 180 expected rows,
duplicate keys, convergence, finite timings and repeat coverage. Outputs include:

- fixed-schema `runs.csv` and separate residual histories
- `audit.json` and reproducibility hashes
- median/IQR timing summaries and convergence/performance figures
- speedup and iteration-ratio tables
- strategy labels, near-tie flags and leave-one-configuration-out predictions
- strict accuracy, tie-aware accuracy, total-time regret and baselines
- a saved decision tree and `report/vaibhav_sections.md`

## Important integration note

The CUDA source supplied in the original archive does not yet accept this
interface and implements plain neighbour averaging rather than the agreed
equation with `r` and source terms. The sync and async owners must align their
executables with this contract before collecting the final dataset.
