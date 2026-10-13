"""Bounded CPU diagnosis using saved GPU grids; no GPU solver execution."""
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from harness_ml.problem import ProblemConfig, make_problem
from harness_ml.reference import solve_reference, true_residual

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = Path(os.environ.get('PCAP_SAVED_GPU_DIR', '/tmp/jacobi-representative-hardened-T44keZEd/outputs'))
OUT = ROOT / 'tighter_validation'
OUT.mkdir(exist_ok=False)
shutil.copytree(OLD, ROOT / 'prior_representative')
commands = (OUT / 'commands.jsonl').open('x')
rows = []
source_rows = []
start = time.monotonic()

def command(args, label):
    commands.write(json.dumps({'label': label, 'command': args}) + '\n')
    commands.flush()
    with (OUT / (label + '.stdout')).open('xb') as out, (OUT / (label + '.stderr')).open('xb') as err:
        p = subprocess.run(args, cwd=REPO, stdout=out, stderr=err, check=False)
    return p.returncode

probe = ROOT / 'probe_problem'
assert command(['g++', '-O2', '-ffp-contract=off', '-std=c++17', '-I', str(REPO),
                str(ROOT / 'probe_problem.cpp'), '-o', str(probe)], 'build_source_probe') == 0
for n, source, r in itertools.product((64, 256), ('point', 'gaussian'), (1, 100)):
    if time.monotonic() - start > 3600:
        raise SystemExit('Phase 1 time limit reached; no more references started')
    label = f'N{n}_{source}_r{r}'
    print('TIGHT REFERENCE', label, flush=True)
    config = ProblemConfig(n, source, r)
    initial, src = make_problem(config)
    assert command([str(probe), str(n), source, str(r)], label + '_cpp_problem') == 0
    cpp = np.fromfile(OUT / (label + '_cpp_problem.stdout'), dtype=np.float32)
    assert cpp.size == 2 * n * n
    cpp_initial, cpp_source = cpp[:n*n].reshape(n,n), cpp[n*n:].reshape(n,n)
    source_rows.append({'N':n,'source':source,'r':r,
                        'initial_bitwise_equal':bool(np.array_equal(initial, cpp_initial)),
                        'source_bitwise_equal':bool(np.array_equal(src, cpp_source)),
                        'source_different_cells':int(np.count_nonzero(src != cpp_source)),
                        'source_max_abs_difference':float(np.max(np.abs(src.astype(float)-cpp_source.astype(float))))})
    result = solve_reference(config, precision='float64', tolerance=1e-10,
                             residual_interval=10, max_iters=2_000_000)
    ref = OUT / (label + '_reference.npy')
    with ref.open('xb') as handle:
        np.save(handle, result.final_grid)
    with (OUT / (label + '_reference.json')).open('x') as handle:
        json.dump({'N':n,'source':source,'r':r,'precision':'float64','tolerance':1e-10,
                   'residual_interval':10,'max_iters':2_000_000,'iterations':result.iterations,
                   'converged':result.converged,'final_residual':result.final_residual,
                   'final_true_residual':result.final_true_residual,
                   'residual_history':result.residual_history}, handle)
    for strategy in ('sync','D1','D2','D4','D8'):
        grid = OLD / (label + '_' + strategy + '.npy')
        vlabel = label + '_' + strategy + '_validation'
        code = command([sys.executable,'-B','-m','harness_ml.validate_converged_grid',
                        '--gpu-grid',str(grid),'--reference-grid',str(ref),
                        '--N',str(n),'--source',source,'--r',str(r),
                        '--max-error','1e-4','--tolerance','1e-4'], vlabel)
        if code not in (0,1):
            raise RuntimeError('Validator execution failure: ' + vlabel)
        metrics=json.loads((OUT/(vlabel+'.stdout')).read_text())
        gpu=np.load(grid)
        new=gpu.copy()
        s=((gpu[:-2,1:-1]+gpu[2:,1:-1])+gpu[1:-1,:-2])+gpu[1:-1,2:]
        new[1:-1,1:-1]=((gpu[1:-1,1:-1]+np.float32(r)*s)+src[1:-1,1:-1])/np.float32(1+4*r)
        row={'N':n,'source':source,'r':r,'strategy':strategy,
             'reference_iterations':result.iterations,'reference_converged':result.converged,
             'reference_residual':result.final_true_residual,
             'rounded_next_update':float(np.max(np.abs(new-gpu))),**metrics}
        rows.append(row)
        print(strategy,'max_error',metrics['max_abs_error'],'residual',metrics['gpu_true_residual'],
              'pass',metrics['passed'],flush=True)
    print('REFERENCE COMPLETE', label, result.iterations, result.final_true_residual, flush=True)
for filename, data in [('validation.csv',rows),('source_check.csv',source_rows)]:
    with (OUT/filename).open('x',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
with (OUT/'done.json').open('x') as handle:
    json.dump({'references':8,'gpu_comparisons':len(rows),'passed':sum(x['passed'] for x in rows),
               'elapsed_seconds':time.monotonic()-start},handle,indent=2)
commands.close()
print('PHASE 1 COMPLETE',flush=True)
