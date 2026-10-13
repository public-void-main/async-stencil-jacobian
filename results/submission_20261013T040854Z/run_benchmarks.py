"""Compact repeat measurements; run from normal Fedora terminal, not sandbox."""
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from harness_ml.problem import ProblemConfig, make_problem
from harness_ml.reference import solve_reference

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
OUT=ROOT/'benchmarks'
OUT.mkdir(exist_ok=False)
(OUT/'raw').mkdir()
(OUT/'references').mkdir()
STRATEGIES=('sync','D1','D2','D4','D8')
CASES=tuple(itertools.product((32,64),('point','gaussian'),(1,100)))
REPEATS=7
start=time.monotonic()
commands=(OUT/'commands.jsonl').open('x')
session=(OUT/'session.log').open('x')
rows=[]

def log(message):
    print(message,flush=True);session.write(message+'\n');session.flush()

def run(args,label):
    commands.write(json.dumps({'label':label,'command':args})+'\n');commands.flush()
    log('COMMAND '+label+': '+' '.join(args))
    with (OUT/'raw'/(label+'.stdout')).open('x') as stdout, (OUT/'raw'/(label+'.stderr')).open('x') as stderr:
        p=subprocess.run(args,cwd=REPO,stdout=stdout,stderr=stderr,check=False)
    return p.returncode

def result(label,whole=False):
    text=(OUT/'raw'/(label+'.stdout')).read_text()
    return json.loads(text if whole else text.splitlines()[-1])

paths=['sync_jacobi.cu','async/async_solver.cu','async/async_kernel.cuh','async/common.h',
       'async/residual_schedule.h','build/sync_jacobi','build/async_jacobi',
       'harness_ml/reference.py','harness_ml/problem.py','harness_ml/validate_converged_grid.py']
metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
          'numpy':np.__version__,'platform':platform.platform(),'sizes':[32,64],
          'sources':['point','gaussian'],'r_values':[1,100],'strategies':STRATEGIES,
          'repeats':REPEATS,'solver_tolerance':1e-6,'reference_tolerance':1e-10,
          'residual_interval':10,'max_iters':2_000_000,'grid_error_threshold':1e-4,
          'independent_residual_threshold':1e-4,
          'timing':'Unmodified CUDA-event total required solve cost; necessary async candidate verification included; external validation excluded',
          'classification':'Numerically rejected cases are exploratory only',
          'sha256':{p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in paths}}
with (OUT/'metadata.json').open('x') as handle:json.dump(metadata,handle,indent=2)
run(['nvidia-smi'],'hardware_before')
run(['nvcc','--version'],'nvcc_version')
references={}
source_checks=[]
for n,source,r in CASES:
    label=f'N{n}_{source}_r{r}'
    ref=OUT/'references'/(label+'.npy')
    if n==64:
        prior=ROOT/'tighter_validation'/(label+'_reference.npy')
        shutil.copyfile(prior,ref)
        shutil.copyfile(ROOT/'tighter_validation'/(label+'_reference.json'),OUT/'references'/(label+'.json'))
    else:
        log('REFERENCE '+label)
        a=solve_reference(ProblemConfig(n,source,r),precision='float64',tolerance=1e-10,
                          residual_interval=10,max_iters=2_000_000)
        if not a.converged or a.final_true_residual >=1e-10:
            raise RuntimeError('Tight CPU reference failed: '+label)
        with ref.open('xb') as handle:np.save(handle,a.final_grid)
        with (OUT/'references'/(label+'.json')).open('x') as handle:
            json.dump({'N':n,'source':source,'r':r,'precision':'float64','iterations':a.iterations,
                       'converged':a.converged,'final_residual':a.final_residual,
                       'final_true_residual':a.final_true_residual,'residual_history':a.residual_history},handle)
    references[(n,source,r)]=ref
    plabel=label+'_cpp_source'
    if run([str(ROOT/'probe_problem'),str(n),source,str(r)],plabel)!=0:
        raise RuntimeError('C++ problem probe failed: '+label)
    data=np.fromfile(OUT/'raw'/(plabel+'.stdout'),dtype=np.float32)
    initial,src=make_problem(ProblemConfig(n,source,r))
    if data.size!=2*n*n:raise RuntimeError('Bad problem probe size: '+label)
    u0,cppsrc=data[:n*n].reshape(n,n),data[n*n:].reshape(n,n)
    source_checks.append({'N':n,'source':source,'r':r,
                          'initial_bitwise_equal':bool(np.array_equal(initial.view(np.uint32),u0.view(np.uint32))),
                          'source_bitwise_equal':bool(np.array_equal(src.view(np.uint32),cppsrc.view(np.uint32)))})
    if not source_checks[-1]['initial_bitwise_equal'] or not source_checks[-1]['source_bitwise_equal']:
        raise RuntimeError('Source sample mismatch: '+label)
with (OUT/'source_checks.json').open('x') as handle:json.dump(source_checks,handle,indent=2)

fields=['N','source','r','strategy','run_id','iters','time_ms','final_residual','converged',
        'gpu_true_residual','reference_true_residual','max_abs_error','rmse',
        'gpu_boundary_correct','reference_boundary_correct','numerically_accepted','exit_code','validator_exit_code','status']
csv_handle=(OUT/'runs.csv').open('x',newline='');writer=csv.DictWriter(csv_handle,fieldnames=fields)
writer.writeheader();csv_handle.flush()
history_handle=(OUT/'residual_history.csv').open('x',newline='')
hist=csv.DictWriter(history_handle,fieldnames=['N','source','r','strategy','run_id','iteration','residual'])
hist.writeheader();history_handle.flush()
stop=False
for n,source,r in CASES:
    for repeat in range(1,REPEATS+1):
        for strategy in STRATEGIES:
            if stop or time.monotonic()-start>7200:
                stop=True;break
            label=f'N{n}_{source}_r{r}_{strategy}_run{repeat}'
            grid=OUT/'raw'/(label+'.npy')
            executable='build/sync_jacobi' if strategy=='sync' else 'build/async_jacobi'
            args=[str(REPO/executable),'--N',str(n),'--source',source,'--r',str(r),
                  '--tolerance','1e-6','--residual-interval','10','--max-iters','2000000',
                  '--dump-npy',str(grid),'--json']
            if strategy!='sync':args+=['--D',strategy[1:]]
            code=run(args,label)
            row={k:None for k in fields}
            base={'N':n,'source':source,'r':r,'strategy':strategy,'run_id':repeat}
            row.update(base,exit_code=code,status='EXECUTION_FAILURE',numerically_accepted=False)
            if code:
                error=(OUT/'raw'/(label+'.stderr')).read_text()
                if any(x in error for x in ('cudaErrorNoDevice','cudaErrorInsufficientDriver','cudaErrorInitializationError')):
                    stop=True;log('STOP CUDA: '+error)
            else:
                a=result(label)
                row.update(iters=a['iterations'],time_ms=a['time_ms'],final_residual=a['final_residual'],converged=a['converged'])
                hist.writerows({**base,'iteration':i,'residual':v} for i,v in a['residual_history'])
                history_handle.flush()
                vc=run([sys.executable,'-B','-m','harness_ml.validate_converged_grid',
                        '--gpu-grid',str(grid),'--reference-grid',str(references[(n,source,r)]),
                        '--N',str(n),'--source',source,'--r',str(r),
                        '--max-error','1e-4','--tolerance','1e-4'],label+'_validation')
                row['validator_exit_code']=vc
                if vc in (0,1):
                    metrics=result(label+'_validation',whole=True)
                    for key in ('gpu_true_residual','reference_true_residual','max_abs_error','rmse','gpu_boundary_correct','reference_boundary_correct'):
                        row[key]=metrics[key]
                    row['numerically_accepted']=bool(metrics['passed'] and a['converged'] and 0<=a['final_residual']<1e-6 and 0<a['iterations']<=2_000_000)
                    row['status']='ACCEPTED' if row['numerically_accepted'] else 'EXPLORATORY_NUMERICAL_FAILURE'
                else:row['status']='VALIDATOR_EXECUTION_FAILURE'
            rows.append(row);writer.writerow(row);csv_handle.flush()
            log(label+' '+row['status'])
        if stop:break
    if stop:break
run(['nvidia-smi'],'hardware_after')
summary={'expected_rows':280,'observed_rows':len(rows),'numerically_accepted_rows':sum(x['numerically_accepted'] for x in rows),
         'complete':len(rows)==280,'elapsed_seconds':time.monotonic()-start}
with (OUT/'done.json').open('x') as handle:json.dump(summary,handle,indent=2)
log('DONE '+json.dumps(summary))
csv_handle.close();history_handle.close();commands.close();session.close()
sys.exit(0 if summary['complete'] else 1)
