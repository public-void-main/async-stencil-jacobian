# Results CSV schema (agree on this on Day 1; do not change later)

File: `results/runs.csv`

| column | meaning |
|---|---|
| N | grid size (512 or 1024) |
| source | point / gaussian |
| r | alpha*dt/h^2 (1, 10, 100) |
| strategy | sync / D1 / D2 / D4 / D8 |
| run_id | repeat index (1..3) |
| iters | iterations to reach tolerance |
| time_ms | wall-clock time to convergence (CUDA events) |
| final_residual | residual at stop |

Residual histories are stored separately in `results/residual_history.csv` with
columns `N, source, r, strategy, run_id, iteration, residual`. This preserves
the agreed `runs.csv` schema while supporting convergence plots.
