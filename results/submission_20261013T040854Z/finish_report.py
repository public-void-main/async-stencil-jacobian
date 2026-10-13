"""Insert measured tables and conclusions into this study's report draft."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
report=REPO/'report.md'
text=report.read_text()
text='\n'.join(line for line in text.splitlines() if not line.startswith('> Working report:'))+'\n'
tight=pd.read_csv(ROOT/'tighter_validation/validation.csv')
summary=pd.read_csv(ROOT/'analysis/benchmark_summary.csv')
median=pd.read_csv(ROOT/'analysis/median_time_table.csv')
baselines=json.loads((ROOT/'analysis/exploratory_baselines.json').read_text())
integrity=json.loads((ROOT/'analysis/integrity.json').read_text())
tests=json.loads((ROOT/'test_status.json').read_text())
relative=ROOT.relative_to(REPO).as_posix()
fig=relative+'/analysis/figures/'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def section(heading,next_heading,body):
    global text
    start=text.index(heading);end=text.index(next_heading,start)
    text=text[:start]+heading+'\n\n'+body.strip()+'\n\n'+text[end:]

diagnosis=[]
for key,g in tight.groupby(['N','source','r']):
    first=g.iloc[0]
    diagnosis.append([*key,int(first.reference_iterations),f'{first.reference_residual:.3e}',
                      f'{g.gpu_true_residual.max():.3e}',f'{g.max_abs_error.max():.6g}',
                      f'{g.rmse.max():.6g}',f'{int(g.passed.sum())}/5'])
section('## 7. Numerical diagnosis and measured results','## 8. Repeated performance results',
'''All eight tighter float64 references converged with an independent residual below 1e−10. The 40 existing GPU grids were reused without solver reruns. C++ problem-generation probes using the current `async/common.h` matched the NumPy initial grid and source samples for every configuration: zero different source cells and maximum source difference zero. r and the denominator were respectively 1/5 or 100/401. Both candidate and reference boundaries passed. Thus the observed N256 Gaussian/r1 discrepancy is not explained by mismatched sources, boundary conventions or parameter values.

The table reports worst values across the five strategies for each configuration. All forty fail the unchanged maximum-error threshold; the five N256 Gaussian/r1 grids also fail the independent residual threshold. These are validation failures, not crashes or iteration-cap non-convergence.

'''+table(['N','Source','r','CPU sweeps','CPU residual','GPU residual (max)','Grid error (max)','RMSE (max)','Accepted'],diagnosis)+'''

For N256 Gaussian/r1, the tighter comparison gives a maximum error about 0.192152 and RMSE about 0.088946; the independently evaluated GPU residual remains 1.1276e−4 despite its reported update of zero. The rounded-map fixed point is therefore insufficient under the agreed accuracy contract. Sync and D1 fail too, so delayed halos alone are not the explanation. No clear, small source or kernel correctness fix was supported by the checks; changing precision or the mathematical stopping criterion would be a larger method/contract change. The default solvers were preserved.

For the synchronous linear map, its slowest mode has eigenvalue `(1+4r*cos(pi/(N−1)))/(1+4r)`. The gap from one shrinks approximately as N−2. A small update can coexist with a much larger remaining low-frequency solution error, and accumulated float32 rounding can leave a biased fixed point. Tightening the CPU reference from 1e−6 to 1e−10 makes comparisons more reliable but does not make the GPU pass. The unchanged source amplitude also makes Gaussian/low-r grids larger in magnitude, increasing representational spacing.

![Independent numerical validation against tight references]('''+fig+'''precision_validation.png)

Figure 1. Maximum solution error and independent residual for Sync on the eight diagnostic configurations. Every strategy was checked separately; the complete forty-row table is saved alongside the figure. Dashed lines are the unchanged 1e−4 thresholds.

![N256 Gaussian precision error]('''+fig+'''gaussian_error_map.png)

Figure 2. Saved N256 Gaussian/r1 GPU grid and its absolute error against the tighter float64 reference. Correct fixed boundaries do not guarantee interior accuracy.''')
timing_rows=[]
for row in median.itertuples(index=False):
    timing_rows.append([row.N,row.source,row.r,*[f'{getattr(row,s):.3f}' for s in ['sync','D1','D2','D4','D8']]])
section('## 8. Repeated performance results','## 9. ML strategy-selection component',
'''The compact matrix contains N32/N64, point/Gaussian sources, r1/r100, five strategies and seven repetitions: **280 runs, forty problem-strategy groups, eight distinct problems**. This smaller deadline-bounded matrix is not the prior N64/N256 420-run expanded experiment and not the default N512/N1024 specification. Repetitions interleave the five strategies within each problem; their order is fixed, not randomized. Each process creates its own context and warm-up outside its solve timer. External reference validation follows the process and is outside every solve timer. Desktop compositor activity and unlocked laptop clocks remain sources of variability.

Integrity checks confirmed all 280 keys, exactly seven repetitions per group, run IDs 1–7, positive finite runtimes, valid iteration counts, 280 correctly aligned residual curves ending at the final sweep, 280 convergence flags and 280 correct candidate boundaries. **Zero rows pass numerical acceptance. All timing results below are exploratory; no validated speedup is claimed.**

Median total required solve time, in milliseconds:

'''+table(['N','Source','r','Sync','D1','D2','D4','D8'],timing_rows)+'''

![Exploratory median solve times and spread]('''+fig+'''timing_median_iqr.png)

Figure 3. Measured medians with interquartile ranges from seven repetitions. These timings include each solver's required convergence work; the independent accuracy failures prevent validated speedup claims.

![Rounded residual histories]('''+fig+'''residual_histories.png)

Figure 4. Actual first-repeat rounded-update histories for point/r1. Zero updates are plotted at a display floor of 1e−8, not reported as an independent residual. Iteration count increases with delayed halo depth in these cases, while reduced launch frequency can lower observed solve time.

The observed best strategy is not universally D8: D4 has the lowest median for N32 point/r1 and N64 Gaussian/r1; D2 has the lowest median for N64 point/r100. These are descriptions of rejected numerical runs, not an accuracy-preserving selection rule. D1 should not be expected to outperform Sync from reducing global launches, since both launch one sweep at a time; its shared-memory path and candidate-check workload differ.

The full forty-group summary records median, Q25/Q75, minimum/maximum, median/minimum/maximum iterations, independent residual, grid error, RMSE, convergence count and acceptance count. Appendix A lists median/IQR and iteration counts; exact repeat-level values remain in the raw CSV and JSON logs.''')
ml_header='## 9. ML strategy-selection component'
ml_start=text.index(ml_header)
ml_end=text.index('## 10. Limitations and conclusions',ml_start)
ml_original=text[ml_start:ml_end]
ml_extra='''### 9.1 Evaluation gate and limited demonstration

There are **zero trustworthy complete configurations** in the final matrix, so no new measured-performance classifier is trained and no ML accuracy, regret or ML speedup is reported. Training on the rejected rows would learn how quickly strategies reach a rounded stopping condition, rather than how quickly they deliver an accepted solution. Historical trained artifacts remain preserved and are excluded from the submission's predictive conclusions.

The existing ML pipeline's synthetic-data functional test passed. It exercises label generation, decision-tree training, leave-one-configuration-out evaluation and artifact writing, but is only a software demonstration. Synthetic test timings are not project measurements and are not used in the results figures.

For transparency, the following are sums of the eight **exploratory per-problem median runtimes**, not ML evaluation totals, not sums of all 280 runs, and not accepted-solution times:

'''
totals=baselines['fixed_strategy_totals_ms']
ml_rows=[['Always Sync',f"{totals['sync']:.3f}",'Exploratory'],
         ['Always D4',f"{totals['D4']:.3f}",'Exploratory'],
         ['Always D8',f"{totals['D8']:.3f}",'Exploratory'],
         ['Best fixed ('+baselines['best_fixed_strategy']+')',f"{min(totals.values()):.3f}",'Exploratory'],
         ['Per-configuration oracle',f"{baselines['per_configuration_oracle_total_ms']:.3f}",'Non-deployable exploratory lower envelope'],
         ['ML selector','Not evaluated','No accurate training/evaluation configurations']]
text=text[:ml_start]+ml_original+ml_extra+table(['Policy','Sum of medians (ms)','Interpretation'],ml_rows)+'\n\n'+text[ml_end:]
section('## 10. Limitations and conclusions','## References',
'''The implementation and harness are reproducible and regression-tested, but numerical acceptance is unresolved. The tighter comparison rejected all forty representative GPU results, and the repeated small-grid matrix rejected all 280 rows. A small rounded update, even exactly zero, cannot certify the steady equation or solution error. This is the principal result of the study, not a successful accurate-solver speedup experiment.

Other limitations are different required convergence-check work across Sync/async, one laptop GPU, small-grid underutilization, fixed strategy order, unlocked clocks, desktop activity, and a limited pre-runtime feature space. There are no newly measured Nsight achieved-occupancy or bandwidth numbers for this final code, and no claim of large-grid scalability or unrestricted asynchronous execution. The constant source samples and ratio r should not be confused with a full time-dependent physical simulation.

The useful findings are: checkpoint-aligned D-sweep scheduling is testable without a GPU; separate global buffers remove the earlier in-place race; higher local depth can reduce observed solve time while increasing sweep counts; and float32 stagnation can dominate numerical acceptance. The decision-tree workflow is implemented and functionally demonstrated, but the accuracy gate correctly prevents training a performance model on rejected solutions.

Before a future validated performance study, define the physical/error requirement explicitly, retain a sufficiently accurate independent reference, and decide whether higher precision or a different independently enforced stopping rule is required. Do not silently relax the current thresholds or retroactively classify these timings as validated. A residual criterion and a solution-error criterion must both be justified for the intended problem scale.

## 11. Reproduction, tests and submission files

All final artifacts are in `'''+relative+'''/`. `prior_representative/` preserves the input grids/logs copied from the earlier temporary validation. `tighter_validation/` contains eight tight references, forty validator outputs, source probes and commands. `benchmarks/` contains the 280-row enriched CSV, 280 histories, grids, solver/validator logs, exact commands and metadata. `analysis/` contains tables, four figures, acceptance/integrity status and the explicit reason no ML model was trained.

The Python suite completed **35 tests successfully**. The host residual-scheduling C++ test built with `-std=c++17 -Wall -Wextra -Werror` and passed. `git diff --check` passed. Test logs, compiler output, the synthetic ML functional-demo log and source/binary hashes are saved with the study. GPU implementations, numerical equations, thresholds, timing boundaries and historical datasets were not changed for this final study. Phase 1 took about 6.5 minutes and repeated collection about 2.3 minutes; no full expanded benchmark was launched.

Minimal commands to reproduce the core numerical checks, using fresh outputs:

```bash
cd ~/async-stencil-jacobian
run_dir="$(mktemp -d /tmp/pcap-check-XXXXXXXX)"
.venv/bin/python -B -m harness_ml.solver_cli --N 64 --source point --r 1 \\
  --precision float64 --tolerance 1e-10 --residual-interval 10 \\
  --max-iters 2000000 --grid-output "$run_dir/reference.npy"
./build/async_jacobi --N 64 --source point --r 1 --D 4 \\
  --tolerance 1e-6 --residual-interval 10 --max-iters 2000000 \\
  --dump-npy "$run_dir/gpu.npy" --json
.venv/bin/python -B -m harness_ml.validate_converged_grid \\
  --gpu-grid "$run_dir/gpu.npy" --reference-grid "$run_dir/reference.npy" \\
  --N 64 --source point --r 1 --max-error 1e-4 --tolerance 1e-4
```

The GPU command must run in the working Fedora terminal session; CUDA was unavailable inside the Codex sandbox. Expect the validator to reject the documented N64 case. Whole-study rerun instructions with fresh directories are provided in the study README; they deliberately retain failed validation results rather than suppressing them.

### Appendix A. Repeated group detail

All rows below are exploratory. Every group has seven converged runs and zero numerically accepted runs. IQR is Q75−Q25, not a confidence interval. Full minimum/maximum and repeat-level data are in `analysis/benchmark_summary.csv` and `benchmarks/runs.csv`.

'''+table(['N','Source','r','Strategy','Median ms','IQR ms','Median sweeps','Max residual','Max grid error'],
          [[x.N,x.source,x.r,x.strategy,f'{x.median_ms:.3f}',f'{x.q75_ms-x.q25_ms:.3f}',
            int(x.median_iters),f'{x.max_true_residual:.3e}',f'{x.max_grid_error:.3e}']
           for x in summary.sort_values(['N','source','r','strategy']).itertuples()]))
text=text.replace('The final tables below use saved measurements only.',
                  'The final study collected 280 repeated measurements, but none met all accuracy checks; consequently no new performance-based ML model was trained. All tables and plots use saved measurements only.')
text=text.replace('Source/binary hashes and toolchain records accompany the results.',
                  'Source/binary hashes and toolchain records accompany the results. Measured NVIDIA driver version is 610.57.04; nvcc is CUDA 13.3, V13.3.73; NVIDIA-SMI reports 8151 MiB device memory.')
text=text.replace('The final conclusion will state which cases pass, which remain exploratory, and what additional work is required before any validated performance or predictive claim.','')
text='\n'.join(line.rstrip() for line in text.splitlines())+'\n'
report.write_text(text)
print('Completed report:',report)
