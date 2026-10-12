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

## Preferred output

The final stdout line should be one JSON object:

```json
{"iterations": 1200, "time_ms": 18.5, "final_residual": 9.2e-6, "converged": true, "residual_history": [[10, 0.8], [20, 0.3]]}
```

`residual_history` is optional but strongly recommended. The aliases `iters`,
`gpu_time_ms`, `residual`, and textual `status` are accepted for easier
integration with existing code.

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
