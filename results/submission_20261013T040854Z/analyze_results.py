"""Generate tables/figures and gate ML using measured, validated rows only."""
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/async-stencil-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis'
OUT.mkdir(exist_ok=False)
FIG=OUT/'figures'
FIG.mkdir()
STRATEGIES=['sync','D1','D2','D4','D8']
runs=pd.read_csv(ROOT/'benchmarks/runs.csv')
keys=['N','source','r','strategy','run_id']
expected=set(itertools.product((32,64),('point','gaussian'),(1,100),STRATEGIES,range(1,8)))
actual=set(runs[keys].itertuples(index=False,name=None))
assert len(runs)==280 and actual==expected and not runs.duplicated(keys).any()
assert np.isfinite(runs[['iters','time_ms','final_residual','gpu_true_residual','reference_true_residual','max_abs_error','rmse']].to_numpy(float)).all()
assert (runs.time_ms>0).all() and (runs.iters>0).all() and (runs.iters<=2_000_000).all()
histories=pd.read_csv(ROOT/'benchmarks/residual_history.csv')
assert histories.groupby(keys).ngroups==280
assert np.isfinite(histories[['iteration','residual']].to_numpy(float)).all()
run_map=runs.set_index(keys)
for key,g in histories.groupby(keys):
    a=g.iteration.to_numpy()
    assert np.array_equal(a,np.arange(10,int(run_map.loc[key,'iters'])+1,10))
accepted=(runs.converged & (runs.final_residual>=0) & (runs.final_residual<1e-6)
          & (runs.gpu_true_residual<1e-4) & (runs.reference_true_residual<1e-4)
          & (runs.max_abs_error<=1e-4) & runs.gpu_boundary_correct & runs.reference_boundary_correct)
assert (accepted==runs.numerically_accepted).all()
groupcols=['N','source','r','strategy']
summary=runs.groupby(groupcols,as_index=False).agg(
    median_ms=('time_ms','median'),q25_ms=('time_ms',lambda x:x.quantile(.25)),
    q75_ms=('time_ms',lambda x:x.quantile(.75)),min_ms=('time_ms','min'),max_ms=('time_ms','max'),
    median_iters=('iters','median'),min_iters=('iters','min'),max_iters=('iters','max'),
    convergence_count=('converged','sum'),accepted_count=('numerically_accepted','sum'),
    max_true_residual=('gpu_true_residual','max'),max_grid_error=('max_abs_error','max'),
    max_rmse=('rmse','max'),boundaries_correct=('gpu_boundary_correct','all'))
sync=summary[summary.strategy=='sync'][['N','source','r','median_ms','median_iters']].rename(
    columns={'median_ms':'sync_median_ms','median_iters':'sync_median_iters'})
summary=summary.merge(sync,on=['N','source','r'])
summary['observed_time_ratio_sync_over_strategy']=summary.sync_median_ms/summary.median_ms
summary['iteration_ratio']=summary.median_iters/summary.sync_median_iters
summary['classification']=np.where(summary.accepted_count==7,'ACCEPTED','EXPLORATORY')
summary.to_csv(OUT/'benchmark_summary.csv',index=False)
pivot=summary.pivot(index=['N','source','r'],columns='strategy',values='median_ms').reindex(columns=STRATEGIES)
pivot.to_csv(OUT/'median_time_table.csv')
validity=runs.groupby(['N','source','r']).numerically_accepted.all()
valid_cases=[list(key) for key,value in validity.items() if value]
accepted_runs=runs[accepted]
accepted_runs.to_csv(OUT/'accepted_rows.csv',index=False)
complete_valid=runs.merge(pd.DataFrame(valid_cases,columns=['N','source','r']),on=['N','source','r']) if valid_cases else runs.iloc[:0]
ml_status={'valid_configurations':len(valid_cases),'valid_cases':valid_cases,
           'method':'Use existing tree only on complete accurate configurations; LOOCV by problem',
           'performance_claim':'No prediction or validated speedup claim from rejected rows'}
if len(valid_cases)>=4:
    from harness_ml.train_ml import train_and_evaluate
    mlruns=complete_valid[['N','source','r','strategy','run_id','iters','time_ms','final_residual']]
    mlruns.to_csv(OUT/'ml_accepted_runs.csv',index=False)
    metrics=train_and_evaluate(OUT/'ml_accepted_runs.csv',OUT/'ml')
    ml_status.update(status='LIMITED_MEASURED_EVALUATION',metrics=metrics)
else:
    ml_status.update(status='NOT_TRAINED_INSUFFICIENT_ACCURATE_CONFIGURATIONS',
                     explanation='Existing synthetic unit tests demonstrate the pipeline, not predictive performance.')
with (OUT/'ml_status.json').open('x') as handle:json.dump(ml_status,handle,indent=2)
med=summary.groupby('strategy').median_ms.sum().reindex(STRATEGIES)
oracle=float(pivot.min(axis=1).sum())
baselines={'scope':'Observed exploratory timings; not validated solver-performance comparisons',
           'fixed_strategy_totals_ms':med.to_dict(),'best_fixed_strategy':str(med.idxmin()),
           'per_configuration_oracle_total_ms':oracle,
           'oracle_labels':pivot.idxmin(axis=1).reset_index(name='strategy').to_dict(orient='records')}
with (OUT/'exploratory_baselines.json').open('x') as handle:json.dump(baselines,handle,indent=2)
fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=False)
for ax,((n,source),g) in zip(axes.flat,summary.groupby(['N','source'])):
    for strategy in STRATEGIES:
        a=g[g.strategy==strategy].sort_values('r')
        ax.errorbar(a.r,a.median_ms,yerr=np.array([a.median_ms-a.q25_ms,a.q75_ms-a.median_ms]),
                    marker='o',capsize=3,label=strategy)
    ax.set_xscale('log');ax.set_xticks([1,100],labels=['1','100'])
    ax.set_title(f'N={n}, {source}');ax.set_xlabel('r');ax.set_ylabel('Total solve time (ms), median/IQR')
    ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.suptitle('Observed timings; rejected cases remain exploratory')
fig.tight_layout();fig.savefig(FIG/'timing_median_iqr.png',dpi=170);plt.close(fig)
tight=pd.read_csv(ROOT/'tighter_validation/validation.csv')
tightsync=tight[tight.strategy=='sync']
fig,axes=plt.subplots(1,2,figsize=(11,4))
labels=[f"{x.N}/{x.source}/r{x.r}" for x in tightsync.itertuples()]
axes[0].bar(labels,tightsync.max_abs_error);axes[0].axhline(1e-4,color='red',ls='--',label='error threshold')
axes[1].bar(labels,tightsync.gpu_true_residual);axes[1].axhline(1e-4,color='red',ls='--',label='residual threshold')
for ax,title in zip(axes,['Maximum error vs tight CPU reference','Independent float64 update residual']):
    ax.set_yscale('log');ax.set_title(title);ax.tick_params(axis='x',rotation=65);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(FIG/'precision_validation.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,n in zip(axes,(32,64)):
    h=histories[(histories.N==n)&(histories.source=='point')&(histories.r==1)&(histories.run_id==1)]
    for strategy in STRATEGIES:
        a=h[h.strategy==strategy]
        ax.semilogy(a.iteration,np.maximum(a.residual,1e-8),label=strategy)
    ax.set_title(f'N={n}, point, r=1');ax.set_xlabel('Sweeps');ax.set_ylabel('Sampled rounded update')
    ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.suptitle('Rounded-update histories; zero values shown at 1e-8 display floor')
fig.tight_layout();fig.savefig(FIG/'residual_histories.png',dpi=170);plt.close(fig)
cpu=np.load(ROOT/'tighter_validation/N256_gaussian_r1_reference.npy')
gpu=np.load(ROOT/'prior_representative/N256_gaussian_r1_sync.npy')
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,data,title in zip(axes,[gpu,np.abs(gpu-cpu)],['Saved GPU grid','Absolute error vs tight float64 reference']):
    im=ax.imshow(data,origin='upper');ax.set_title(title);fig.colorbar(im,ax=ax)
fig.tight_layout();fig.savefig(FIG/'gaussian_error_map.png',dpi=170);plt.close(fig)
audit={'status':'COMPLETE','rows':len(runs),'groups':summary.shape[0],'repetitions_per_group':7,
       'history_curves':280,'solver_converged_rows':int(runs.converged.sum()),
       'boundary_correct_rows':int(runs.gpu_boundary_correct.sum()),'accepted_rows':int(accepted.sum()),
       'complete_valid_configurations':len(valid_cases),'invalid_rows_retained':int((~accepted).sum())}
with (OUT/'integrity.json').open('x') as handle:json.dump(audit,handle,indent=2)
print(json.dumps(audit,indent=2))
print('Median time table (ms):\n'+pivot.to_string())
print('ML status: '+ml_status['status'])
