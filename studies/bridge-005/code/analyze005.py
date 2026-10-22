"""Frozen inference for one selected pair and a twelve-quantity primary family."""
import argparse
from common005 import *

CONDITIONS=['clean','corrupt','wording','self','zero','order','role','order_role']+['residual'+str(i) for i in range(3)]+['projected'+str(i) for i in range(3)]
def main(output=None):
    verify('input-freeze.json');verify('selection-freeze.json');verify('test-freeze.json');self_test()
    out=HERE/output if output else HERE/'results/analysis'
    assert not out.exists(),out
    sel=read(HERE/'data/selection.json');rows=rows_for('test',sel['families']);path=HERE/'results/test';rec=read(path/'run.json')
    assert rec['status']=='completed' and rec['record_ids']==[r['id'] for r in rows]
    for name,sha in rec['hashes'].items():assert digest(path/name)==sha,name
    data={k:load_arrays(path/(k+'.npz')) for k in CONDITIONS};clean=data['clean'];corrupt=data['corrupt'];fit=sel['reference'];scale=np.asarray(fit['scale'])[:,None]
    for name in ['clean','corrupt']:np.testing.assert_allclose(data[name]['behavior'],load_arrays(path/('natural-'+name+'.npz'))['behavior'],rtol=0,atol=TOL)
    np.testing.assert_allclose(data['zero']['behavior'],corrupt['behavior'],rtol=0,atol=TOL)
    for k in ['residual','projected']:
        for i in range(3):np.testing.assert_allclose(data[k+str(i)]['head_norms'],data['wording']['head_norms'],atol=.0001,rtol=0)
    masks={f'w{w}_o{o}':np.array([r['wording']==w and r['order']==o for r in rows]) for w in range(2) for o in range(2)}
    clean_cells=behavior_cells(clean['behavior'],rows);sep=separation(clean['raw'],rows,fit);donors=donor_indices(rows,'wording')
    r=clean['raw'][...,0];d=r[:,donors];metrics={}
    result=dict(model=MODEL,n=len(rows),families=sel['families'],source_heads=sel['source_heads'],primary_multiplicity=12,selection_worst=sel['selection_worst'],independent_qualification=read(HERE/'results/qualification-summary.json'),clean_behavior=clean_cells,clean_separation=sep,test_behavior_pass=behavior_pass(clean_cells),test_separation_pass=min_separation(sep)>=.8,conditions={},primary={},sensitivities={},baseline_movement={},endpoint_checks={})
    result['corrupt_top1']=float(corrupt['behavior'][:,3].mean());result['clean_top1']=float(clean['behavior'][:,3].mean())
    for factor in CONDITIONS:
        if factor in ['clean','corrupt']:continue
        v=data[factor];gain=v['behavior'][:,0]-corrupt['behavior'][:,0];metrics[factor+'__recovery']=gain
        result['conditions'][factor]=dict(recovery=intervals(gain,rows),probability_recovery=intervals(v['behavior'][:,1]-corrupt['behavior'][:,1],rows),top1=float(v['behavior'][:,3].mean()),pairwise=float(v['behavior'][:,2].mean()),cells={k:dict(recovery=intervals(gain,rows,mask,12 if factor=='wording' else 1),top1=float(v['behavior'][mask,3].mean()),pairwise=float(v['behavior'][mask,2].mean())) for k,mask in masks.items()})
        if 'raw' not in v:continue
        g=paired(v['raw'][...,0],r,d,scale)
        for k,arr in g.items():metrics[factor+'__'+k]=arr
        result['conditions'][factor]['unresolvable_pairs']=int((~g['resolvable']).sum())
        result['conditions'][factor]['geometry']={}
        for bi,b in enumerate(['zero','fit_mean']):
            result['conditions'][factor]['geometry'][b]={}
            for k in ['score','q_recipient','q_donor','margin','orthogonal']:
                z=g[k][bi];finite=z[np.isfinite(z)]
                result['conditions'][factor]['geometry'][b][k]=dict(mean=float(finite.mean()) if len(finite) else None,finite_n=len(finite),quantiles=np.quantile(finite,[0,.05,.25,.5,.75,.95,1]).tolist() if len(finite) else None,cells={cell:intervals(z,rows,mask,12 if factor=='wording' and k=='margin' else 1) if np.isfinite(z[mask]).all() else None for cell,mask in masks.items()})
        if factor!='wording':continue
        result['primary']=dict(recovery={k:x['recovery'] for k,x in result['conditions'][factor]['cells'].items()},margin={b:result['conditions'][factor]['geometry'][b]['margin']['cells'] for b in ['zero','fit_mean']})
        old=geometry(v['raw'],clean['raw'],clean['raw'][:,donors],rows,fit,'wording');unscaled=paired(v['raw'][...,0],r,d,np.ones_like(scale))
        for name,arr in [('unscaled_paired_margin',unscaled['margin']),('standardized_cosine_margin',g['cosine_margin']),('centroid_normalized_margin',old['recipient_distance']-old['donor_distance'])]:
            metrics[name]=arr
            result['sensitivities'][name]={b:{k:intervals(arr[bi],rows,mask) if np.isfinite(arr[bi,mask]).all() else None for k,mask in masks.items()} for bi,b in enumerate(['zero','fit_mean'])}
        baseline=paired(corrupt['raw'][...,0],r,d,scale)
        for bi,b in enumerate(['zero','fit_mean']):
            result['baseline_movement'][b]={}
            for k in ['score','margin','q_donor','q_recipient']:
                change=g[k][bi]-baseline[k][bi];metrics[b+'__'+k+'_change']=change
                result['baseline_movement'][b][k]=intervals(change,rows) if np.isfinite(change).all() else None
        for label,endpoint,expected in [('recipient',r,dict(score=0,q_recipient=0,q_donor=1,margin=-1,orthogonal=0)),('donor',d,dict(score=1,q_recipient=1,q_donor=0,margin=1,orthogonal=0))]:
            ep=paired(endpoint,r,d,scale)
            result['endpoint_checks'][label]={k:float(np.nanmax(np.abs(ep[k]-v))) for k,v in expected.items()}
    for kind in ['residual','projected']:
        avg=np.mean([data[kind+str(i)]['behavior'][:,0] for i in range(3)],axis=0)
        result['wording_over_'+kind]=intervals(data['wording']['behavior'][:,0]-avg,rows)
    result['qualified']=bool(result['test_behavior_pass'] and result['test_separation_pass'] and result['independent_qualification']['qualified'] and sel['status']=='selected' and result['conditions']['wording']['unresolvable_pairs']==0)
    result['all_primary_recoveries_positive']=all(v['adjusted_ci'][0]>0 for v in result['primary']['recovery'].values())
    result['all_primary_margins_recipient']=all(v is not None and v['adjusted_ci'][1]<0 for cells in result['primary']['margin'].values() for v in cells.values())
    result['full_conjunction_supported']=bool(result['qualified'] and result['all_primary_recoveries_positive'] and result['all_primary_margins_recipient'])
    out.mkdir(parents=True);write(out/'summary.json',result);np.savez_compressed(out/'per-example.npz',**metrics)
    print('FINAL ANALYSIS',{k:result[k] for k in ['qualified','all_primary_recoveries_positive','all_primary_margins_recipient','full_conjunction_supported']},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();main(a.output)
