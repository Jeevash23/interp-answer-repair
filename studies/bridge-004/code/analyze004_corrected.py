"""Frozen primary 24-quantity family, with labeled pointwise sensitivities."""
import argparse
import numpy as np
from common004 import *
from geometry004 import paired,self_test
from transport_bridge.statistics import geometry

def main(model,output=None):
    verify('input-freeze.json');verify(model+'-test-freeze.json');verify('analysis-correction-freeze.json');self_test()
    sel=read(HERE/'data'/f'{model}-selection.json');rows=rows_for(model,'test',sel['families']);directory=HERE/'results'/model/'test';record=read(directory/'run.json')
    assert record['status']=='completed';assert record['record_ids']==[r['id'] for r in rows]
    for p,h in record['hashes'].items():assert digest(directory/p)==h
    out=HERE/output if output else HERE/'results'/model/'analysis';out.mkdir(parents=True,exist_ok=False)
    data={p[:-4]:load_arrays(directory/p) for p in record['hashes'] if p!='natural.npz'};nat=load_arrays(directory/'natural.npz')
    tol=.002 if model.startswith('tiny') else .0005
    for k in ['clean','corrupt']:np.testing.assert_allclose(data[k]['behavior'],nat[k+'_behavior'],atol=tol,rtol=0)
    np.testing.assert_allclose(data['zero']['behavior'],data['corrupt']['behavior'],atol=tol,rtol=0)
    masks={f'w{w}_o{o}':np.array([r['wording']==w and r['order']==o for r in rows]) for w in range(2) for o in range(2)}
    fit=sel['reference'];scale=np.asarray(fit['scale'])[:,None];clean=data['clean'];corrupt=data['corrupt'];coords=axis_coordinates(clean['raw'],rows,fit,'wording');targets=np.array([r['wording'] for r in rows]);metrics={}
    summary=dict(model=model,n=len(rows),families=sel['families'],source_heads=sel['source_heads'],primary_multiplicity=24,discovery_qualified=sel['discovery_qualified'],natural={},conditions={},primary={},sensitivities={},endpoint_checks={},factorial={})
    for k,mask in masks.items():
        c=clean['behavior'][mask];b=corrupt['behavior'][mask]
        summary['natural'][k]=dict(pairwise=float(c[:,2].mean()),top1=float(c[:,3].mean()),corrupt_pairwise=float(b[:,2].mean()),corrupt_top1=float(b[:,3].mean()),wording_classification=((coords[:,mask]>0)==targets[mask]).mean(1).tolist())
    summary['behavior_valid']=all(v['pairwise']>=.95 and v['top1']>=.85 for v in summary['natural'].values())
    summary['test_separated']=all(min(v['wording_classification'])>=.8 for v in summary['natural'].values())
    for factor,v in data.items():
        if factor in ['clean','corrupt']:continue
        b=v['behavior'];gain=b[:,0]-corrupt['behavior'][:,0];mult=24 if factor=='wording' else 1
        c=dict(recovery=interval(gain,rows),probability_recovery=interval(b[:,1]-corrupt['behavior'][:,1],rows),pairwise=float(b[:,2].mean()),top1=float(b[:,3].mean()),cells={k:dict(recovery=interval(gain,rows,mask,mult),pairwise=float(b[mask,2].mean()),top1=float(b[mask,3].mean())) for k,mask in masks.items()})
        if 'raw' in v:
            f='order' if factor=='order' else 'wording';donors=donor_indices(rows,f)
            e,r,d=v['raw'][...,0],clean['raw'][...,0],clean['raw'][...,0][:,donors]
            g=paired(e,r,d,scale);c['unresolvable_pairs']=int((~g['resolvable']).sum());c['geometry']={}
            # Any undefined pair prevents primary qualification; all examples remain in raw arrays.
            for name,arr in g.items():metrics[factor+'__'+name]=arr
            for bi,baseline in enumerate(['zero','discovery_mean']):
                c['geometry'][baseline]={}
                for k,mask in masks.items():
                    cell={}
                    for name in ['score','q_recipient','q_donor','margin','orthogonal']:
                        cell[name]=interval(g[name][bi],rows,mask,mult if name=='margin' else 1) if np.isfinite(g[name][bi,mask]).all() else None
                    c['geometry'][baseline][k]=cell
            if factor=='wording':
                summary['primary']=dict(recovery={k:x['recovery'] for k,x in c['cells'].items()},margin={b:{k:x['margin'] for k,x in cells.items()} for b,cells in c['geometry'].items()})
                original=geometry(v['raw'],clean['raw'],clean['raw'][:,donors],rows,fit,'wording');unscaled=paired(e,r,d,np.ones_like(scale))
                for name,arr in [('centroid_normalized_margin',original['recipient_distance']-original['donor_distance']),('unscaled_paired_margin',unscaled['margin']),('standardized_cosine_margin',g['cosine_margin'])]:
                    metrics[name]=arr;summary['sensitivities'][name]={b:{k:interval(arr[bi],rows,mask) if np.isfinite(arr[bi,mask]).all() else None for k,mask in masks.items()} for bi,b in enumerate(['zero','discovery_mean'])}
                for label,endpoint,expected in [('recipient',r,{'score':0,'q_recipient':0,'q_donor':1,'margin':-1,'orthogonal':0}),('donor',d,{'score':1,'q_recipient':1,'q_donor':0,'margin':1,'orthogonal':0})]:
                    check=paired(endpoint,r,d,scale);summary['endpoint_checks'][label]={name:float(np.nanmax(np.abs(check[name]-val))) for name,val in expected.items()}
                for bi,baseline in enumerate(['zero','discovery_mean']):
                    c['geometry'][baseline]['pooled_distribution']={name:dict(mean=float(np.nanmean(g[name][bi])),quantiles=np.nanquantile(g[name][bi],[0,.05,.25,.5,.75,.95,1]).tolist()) for name in ['score','q_recipient','q_donor','margin','orthogonal']}
        summary['conditions'][factor]=c
    a,r,o,b=[data[k]['behavior'][:,0] for k in ['self','role','order','order_role']]
    for name,x in dict(order=(o+b-a-r)/2,role=(r+b-a-o)/2,interaction=b-o-r+a).items():summary['factorial'][name]={k:interval(x,rows,mask) for k,mask in masks.items()}
    rand=np.mean([data['random'+str(i)]['behavior'][:,0] for i in range(3)],axis=0);summary['wording_over_random']=interval(data['wording']['behavior'][:,0]-rand,rows)
    recovery=all(v['adjusted_ci'][0]>0 for v in summary['primary']['recovery'].values());proximity=all(v is not None and v['adjusted_ci'][1]<0 for cells in summary['primary']['margin'].values() for v in cells.values())
    summary.update(all_primary_recoveries_positive=recovery,all_primary_margins_recipient=proximity,qualified=bool(sel['discovery_qualified'] and summary['behavior_valid'] and summary['test_separated'] and summary['conditions']['wording']['unresolvable_pairs']==0))
    summary['repair_with_recipient_proximity_supported']=bool(summary['qualified'] and recovery and proximity)
    write(out/'summary.json',summary)
    # NaNs are reserved for explicitly flagged unresolvable geometry; never removed silently.
    np.savez_compressed(out/'per-example.npz',**metrics)
    print('ANALYSIS',model,{k:summary[k] for k in ['qualified','all_primary_recoveries_positive','all_primary_margins_recipient','repair_with_recipient_proximity_supported']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);p.add_argument('--output');a=p.parse_args();main(a.model,a.output)
