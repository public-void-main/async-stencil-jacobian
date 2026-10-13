# CUDA integration contract

This is the only handoff required from the synchronous and asynchronous CUDA
owners. It does not prescribe their kernel implementation.

## Shared numerical definition

Both executables must use `experiment_spec.json` as the source of truth:

- float32 state
- fixed Dirichlet boundaries: top edge 100, all other edges 0
- point source at `(floor(N/2), floor(N/2))`, peak `dt*f = 1`
- Gaussian source at the same centre, `sigma = N/16`, peak `dt*f = 1`
- update equation:
  `u_new = (u_old + r*(north+south+west+east) + dt*f)/(1+4*r)`
- maximum absolute update as residual, checked every 10 iterations
- convergence when residual is strictly below `1e-4`

The same boundaries and source term must be retained on every local sweep. `D`
is the number of local shared-memory sweeps between halo refreshes.

## Process interface

The harness starts one process for one `(N, source, r, strategy, run_id)` tuple.
Commands are configured in `solvers.json`; argument names can be changed there
without modifying Python code. Supported placeholders are:

`{N}`, `{source}`, `{r}`, `{strategy}`, `{D}`, `{run_id}`, `{seed}`,
`{tolerance}`, `{residual_interval}`, `{max_iters}`, `{python}`, `{root}`.

Elapsed time must cover GPU solving and residual checks, but exclude process
startup and problem generation. A non-zero exit code means the run failed.

## Total solve time

The primary metric is `time_ms` from CUDA events bracketing the complete solver
loop, including every convergence check required by that solver. Keep these
boundaries and the existing stopping semantics unchanged for Sync and D1/D2/D4/D8.

Included: iteration kernels, residual resets/reductions, scalar residual copies,
and host checkpoint decisions between the events. Async additionally copies
candidate grids to the host and evaluates `true_max_update` before accepting
convergence (and at the iteration cap). This necessary verification stays inside
the interval. CUDA events do not directly time CPU instructions, but the stop
event is submitted after verification, so the interval can include GPU idle time
while the host verifies a candidate.

Sync stops on its sampled sweep update without that independent candidate check.
This asymmetry is intentional under the total solve time contract: necessary
async verification may affect strategy comparisons and ML labels. Do not subtract
it, remove it, move it after timing, or change Sync's stopping rule to force
equivalent work.

Excluded for every strategy: process/context startup, problem generation, device
allocation, initial transfers, warm-up, post-loop final-grid transfer, post-loop
residual evaluation where present, grid output, reference comparisons, and cleanup.
Async's reported final residual is evaluated after the loop; its residual history
contains local-sweep updates. Sync reports its sampled sweep-update residual.

Positive `--iters K` selects fixed mode: exactly K sweeps, no convergence stopping,
and `converged=false`. Candidate verification is absent in fixed mode. Preserve
the existing sampling behavior: Sync checks every M sweeps and at K; normal async
fixed mode checks only at K.

## Expanded rerun override

The `fair_v1` rerun explicitly overrides the default experiment specification:
N64/N256, point/Gaussian, r1/r10/r100, Sync/D1/D2/D4/D8, seven repetitions,
tolerance `1e-6`, residual interval 10, and maximum 2,000,000 iterations.
There are 12 problems, 60 problem-strategy groups, and 420 runs. This does not
change the defaults in `experiment_spec.json`.

Use new outputs only: `results/fair_runs_v1.csv`, `results/fair_histories_v1.csv`,
and `results/ml_fair_v1/`. Refuse to overwrite existing paths. Validate the exact
matrix, unique run keys, seven repetitions per group, convergence, and finite
results before training. Compare held-out ML predictions with always Sync,
always D4, always D8, and the best fixed strategy; report the per-configuration
oracle separately. These comparisons include necessary async verification.

## Preferred output

The final stdout line should be one JSON object:

```json
{"iterations": 1200, "time_ms": 18.5, "final_residual": 9.2e-6, "converged": true, "residual_history": [[10, 0.8], [20, 0.3]]}
```

`residual_history` is optional but strongly recommended. The aliases `iters`,
`gpu_time_ms`, `residual`, and textual `status` are accepted for easier
integration with existing code.

The CPU reference CLI additionally reports `precision`, `final_true_residual`,
and `final_residual_iteration`. Its existing `final_residual` is the last sampled
sweep update, which may precede the final grid at an iteration cap. If no update
checkpoint was sampled, both `final_residual` and `final_residual_iteration` are
JSON `null`. `final_true_residual` evaluates the completed grid in float64;
non-finite residual values serialize as `null`, never bare Infinity or NaN.
The default float32 stopping rule is unchanged; explicit `--precision float64`
also requires the independent residual to pass before declaring convergence.
Non-converged CPU outputs remain unsuitable for the experiment runner.

The converged-grid validator preserves float32/float64 input precision, reports
`gpu_boundary_correct` and `reference_boundary_correct`, and requires both full
Dirichlet boundary rings to match the generated problem. Its existing error and
independent residual thresholds remain unchanged. Non-finite grids fail validation;
unsupported shapes/dtypes, non-finite sources, and invalid parameters are rejected.

## Existing text output fallback

If JSON is inconvenient, the automatic parser accepts exactly one block with:

```text
Iterations         : 1200
Final residual     : 9.2e-6
Convergence status : TOLERANCE REACHED
GPU time           : 18.5 ms
```

An executable that prints results for multiple configurations in one process is
rejected because the harness cannot safely associate timing and residual history
with the requested experiment tuple.

## Integration sequence

1. Validate a small fixed-iteration GPU grid with `validate_grid.py`.
2. Run one configuration for all five strategies into a temporary CSV.
3. Confirm D1-with-barrier matches the synchronous final grid.
4. Run the full pipeline only after these checks pass.
