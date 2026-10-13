# Deadline-bounded submission study

This directory is a new artifact set. All historical outputs remain untouched.

- `tighter_validation/`: eight float64 references at tolerance 1e-10 and forty
  comparisons with saved GPU grids; all forty comparisons fail grid-error acceptance.
- `prior_representative/`: preserved copy of the input GPU grids and original logs.
- `benchmarks/`: 280 measured runs, N32/N64 × point/Gaussian × r1/r100 × five
  strategies × seven repeats, with solver tolerance 1e-6 and interval 10.
- `analysis/`: median/IQR/min/max tables, four plots and integrity/ML status.
- `python_tests.txt`, `host_test_*.txt`, `test_status.json`: verification evidence.
- `ml_pipeline_demonstration.txt`: synthetic-data unit-test demonstration only.

Every repeated row fails the unchanged maximum grid-error requirement. Timings
are exploratory, not validated speedups. No new performance selector is trained.
Necessary async candidate verification remains included in its existing solve
timer, while external validation is excluded for all strategies. This differs
from Sync's stopping workload and is disclosed in the report.

## Reproduce without overwriting results

Use a fresh directory under `results/`; the runners refuse existing output
subdirectories. Run GPU work only from the functioning Fedora terminal.

```bash
cd ~/async-stencil-jacobian
study_dir="$(mktemp -d "$PWD/results/submission-repeat-XXXXXXXX")"
cp results/submission_20261013T040854Z/probe_problem.cpp "$study_dir/"
cp results/submission_20261013T040854Z/run_tighter_references.py "$study_dir/"
cp results/submission_20261013T040854Z/run_benchmarks.py "$study_dir/"
cp results/submission_20261013T040854Z/analyze_results.py "$study_dir/"
PCAP_SAVED_GPU_DIR="$PWD/results/submission_20261013T040854Z/prior_representative" \
  PYTHONPATH=. .venv/bin/python -B "$study_dir/run_tighter_references.py"
PYTHONPATH=. .venv/bin/python -B "$study_dir/run_benchmarks.py"
PYTHONPATH=. .venv/bin/python -B "$study_dir/analyze_results.py"
```

The source probe needs g++. The numerical reference, analysis and tests use
`harness_ml/requirements.txt`. GPU executables must already exist; build commands
are in the report. New machines need compatible CUDA hardware/toolchain.

Raw solver/validator commands are recorded in each stage's `commands.jsonl`.
Failed accuracy checks are retained and cause rows to be marked exploratory.
CUDA initialization failure stops further GPU execution. No matrices overwrite
the historical `expanded_*` or `fair_*` paths.

`report.md` is the final written report at the repository root; its HTML/PDF export
and submission archive are produced separately after numerical/test verification.
