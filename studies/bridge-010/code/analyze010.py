"""Frozen Study 5 analysis, run only after all twelve conditions complete."""
import json
import numpy as np
from common010 import *


class Bootstrap:
    def __init__(self, rows):
        self.base = Crossed(rows, draws=DRAWS, seed=ANALYSIS_SEED)
        self.draws = load(HERE/'data/draws/bootstrap.npz')
        np.testing.assert_array_equal(self.draws['blocks'],self.base.tt)
        np.testing.assert_array_equal(self.draws['triples'],self.base.pp)
        self.repweights=[]
        for draws in self.draws['replicates']:
            self.repweights.append(np.stack([(draws==k).mean(1) for k in range(REPLICATES)],1))

    def resample(self, values, control=None, mask=None):
        values=np.asarray(values,float)
        if values.ndim==1:
            grid=self.base.grid(values,mask)
            return float(grid.mean()), self.base.resample(grid)
        assert values.shape==(REPLICATES,len(self.base.rows)) and control is not None
        grids=[self.base.grid(x,mask) for x in values]
        samples=np.stack([self.base.resample(g) for g in grids],1)
        return float(np.mean(grids)),np.sum(samples*self.repweights[control],axis=1)


def interval(point, draws, multiplicity=1):
    if not np.isfinite(point) or not np.isfinite(draws).all():
        return dict(valid=False,mean=None,ci95=None,adjusted_ci=None,
                    reason='Unresolved endpoint or zero denominator; no observations omitted')
    tail=.05/(2*multiplicity)
    return dict(valid=True,mean=float(point),ci95=np.quantile(draws,[.025,.975]).tolist(),
                adjusted_ci=np.quantile(draws,[tail,1-tail]).tolist(),multiplicity=multiplicity)


def distribution(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all(): return dict(valid=False)
    return dict(valid=True,minimum=float(x.min()),maximum=float(x.max()),
                quantiles=dict(zip(['p05','p25','p50','p75','p95'],
                    np.quantile(x,[.05,.25,.5,.75,.95]).tolist())))


def main():
    from runner007 import digest, save_arrays, now
    target=HERE/'results/analysis'
    assert not target.exists(),'Analysis is immutable; verify rather than overwrite'
    completion=read(HERE/'results/inference-complete.json'); assert completion['passed']
    rows=read(HERE/'data/rows.json'); donors=validate_rows(rows)
    baseline={n:load(BASE/f'results/test/{n}.npz') for n in ('clean','corrupt','self','wording')}
    r=baseline['clean']['raw'][...,0].astype(float); d=r[:,donors]
    scale=np.asarray(read(BASE/'data/selection.json')['reference']['scale'])[:,None]
    self_raw=baseline['self']['raw'][...,0].astype(float)
    word_raw=baseline['wording']['raw'][...,0].astype(float)
    gs=paired(self_raw,r,d,scale); gw=paired(word_raw,r,d,scale)
    denominator=np.where(gw['resolvable'],gw['pair_norm'],np.nan)
    word=dict(F=np.linalg.norm((word_raw-self_raw)/scale,axis=-1)/denominator,
              A=np.abs(gw['score']-gs['score']))
    boot=Bootstrap(rows); masks=cell_masks(rows)
    report=dict(utc=now(),study=5,scope='Exploratory, inspected Study 2 bank; no pooling',
                n=len(rows),replicates=REPLICATES,controls={},wording={},
                bootstrap_draws=DRAWS,seed=ANALYSIS_SEED,multiplicity=MULTIPLICITY,
                unresolvable=int((~gw['resolvable']).sum()),
                uncertainty='Crossed context/name bootstrap plus global replicate resampling per control; conditional on fitted assay and fixed source bank. Ratios are ratios of means, not means of ratios. Adjusted intervals cover 12 geometric and 12 behavioural paired contrasts; ratio and difference are two expressions of each geometric contrast.')
    arrays={f'word_{k}':v for k,v in word.items()}
    for ri,ref in enumerate(REFS):
        report['wording'][ref]={m:interval(*boot.resample(v[ri])) for m,v in word.items()}
    report['baseline_behaviour']={}
    for n in baseline:
        b=baseline[n]['behavior'].astype(float)
        report['baseline_behaviour'][n]={key:interval(*boot.resample(value)) for key,value in
            [('recovery',b[:,0]-baseline['corrupt']['behavior'][:,0]),('top1',b[:,3]),('pairwise',b[:,2])]}
    input_hashes={}
    for ci,control in enumerate(CONTROLS):
        rec=[];bs=[]; geometry={key:[] for key in ('F','A','signed_axis','score','q_recipient')}
        for k in range(REPLICATES):
            path=HERE/f'results/test/{control}-r{k}.npz'
            assert digest(path)==completion['outputs'][f'{control}-r{k}']['sha256']
            input_hashes[str(path.relative_to(ROOT))]=digest(path)
            z=load(path); raw=z['raw'][...,0].astype(float);g=paired(raw,r,d,scale)
            geometry['F'].append(np.linalg.norm((raw-self_raw)/scale,axis=-1)/denominator)
            geometry['A'].append(np.abs(g['score']-gs['score']))
            geometry['signed_axis'].append(g['score']-gs['score'])
            geometry['score'].append(g['score']);geometry['q_recipient'].append(g['q_recipient'])
            bs.append(z['behavior'].astype(float))
        geometry={k:np.stack(v) for k,v in geometry.items()};bs=np.stack(bs)
        arrays.update({control+'_'+key:value for key,value in geometry.items()})
        arrays[control+'_behavior']=bs
        entry=dict(geometry={},behaviour={},cells={},replicates=[])
        for ri,ref in enumerate(REFS):
            item={}
            for metric in ('F','A'):
                v=geometry[metric][:,ri]
                nullpoint,nullboot=boot.resample(v,ci)
                wp,wb=boot.resample(word[metric][ri])
                with np.errstate(invalid='ignore',divide='ignore'):
                    ratio=wp/nullpoint;rb=wb/nullboot
                item[metric]=dict(null=interval(nullpoint,nullboot),
                     word_over_null=interval(ratio,rb,MULTIPLICITY),
                     null_minus_word=interval(nullpoint-wp,nullboot-wb,MULTIPLICITY),
                     configuration_distribution=distribution(v.mean(0)),
                     all_draw_distribution=distribution(v))
            for key in ('signed_axis','score','q_recipient'):
                item[key]=interval(*boot.resample(geometry[key][:,ri],ci))
            entry['geometry'][ref]=item
        for metric,values in [('recovery',bs[:,:,0]-baseline['corrupt']['behavior'][None,:,0]),
                              ('top1',bs[:,:,3]),('pairwise',bs[:,:,2])]:
            p,b=boot.resample(values,ci)
            entry['behaviour'][metric]=dict(null=interval(p,b))
            if metric!='pairwise':
                for base in ('self','wording'):
                    bv=(baseline[base]['behavior'][:,0].astype(float)-baseline['corrupt']['behavior'][:,0]) if metric=='recovery' else baseline[base]['behavior'][:,3].astype(float)
                    bp,bb=boot.resample(bv)
                    entry['behaviour'][metric]['minus_'+base]=interval(p-bp,b-bb,MULTIPLICITY)
        entry['disruptions']=dict(
            definition='Retained control cases with lower logit gap than self; accuracy loss requires self correct and control wrong.',
            gap_below_self_per_replicate=(bs[:,:,0]<baseline['self']['behavior'][None,:,0]).sum(1).tolist(),
            lost_top1_from_self_per_replicate=((baseline['self']['behavior'][None,:,3]==1)&(bs[:,:,3]==0)).sum(1).tolist(),
            recovery_below_zero_per_replicate=(bs[:,:,0]<baseline['corrupt']['behavior'][None,:,0]).sum(1).tolist())
        for cell,mask in masks.items():
            entry['cells'][cell]={ref:{metric:interval(*boot.resample(geometry[metric][:,ri],ci,mask))
                                  for metric in ('F','A','signed_axis','q_recipient')}
                                  for ri,ref in enumerate(REFS)}
            entry['cells'][cell]['behaviour']={metric:interval(*boot.resample(v,ci,mask)) for metric,v in
                      [('recovery',bs[:,:,0]-baseline['corrupt']['behavior'][None,:,0]),('top1',bs[:,:,3])]}
        for k in range(REPLICATES):
            entry['replicates'].append(dict(replicate=k,
                geometry={ref:{metric:float(geometry[metric][k,ri].mean()) for metric in ('F','A','signed_axis','q_recipient')}
                            for ri,ref in enumerate(REFS)},
                recovery=float((bs[k,:,0]-baseline['corrupt']['behavior'][:,0]).mean()),top1=float(bs[k,:,3].mean())))
        report['controls'][control]=entry
    report['input_hashes']=input_hashes
    report['freeze_sha256']=digest(FREEZE)
    save_arrays(target/'per-example.npz',**arrays)
    report['per_example_sha256']=digest(target/'per-example.npz')
    write(target/'summary.json',report)
    print(json.dumps(dict(status='analysed',summary_sha256=digest(target/'summary.json'))),flush=True)


if __name__=='__main__': main()
