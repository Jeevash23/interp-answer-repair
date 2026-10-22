"""Saved-output replay and independent arithmetic checks, preserving all prior records."""
import argparse,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
from common004 import *

def prior():
    manifest=read(ROOT/'output/combined-artifact-manifest-v3.json');checked=0
    for name,sha in manifest['files'].items():assert digest(ROOT.parent/name)==sha,name;checked+=1
    # Draft v1 is a separate immutable artifact layer from the older research ZIP.
    for name,sha in read(ROOT/'paper/draft-v1/evidence-source-hashes.json').items():assert digest(ROOT.parent/name)==sha,name
    v1=read(ROOT/'paper/draft-v1/visual-verification.json')
    for ext in ['pdf','docx']:assert digest(ROOT/'paper/draft-v1'/('manuscript.'+ext))==v1[ext+'_sha256']
    return checked

def replay(model):
    import analyze004_corrected as analyze004
    for name in ['input-freeze.json',model+'-discovery-freeze.json',model+'-test-freeze.json']:verify(name)
    sel=read(HERE/'data'/f'{model}-selection.json');rows=rows_for(model,'test',sel['families']);directory=HERE/'results'/model;record=read(directory/'test/run.json');assert record['status']=='completed'
    for r in rows:
        assert r['token_ids'][r['s2']]==r['subject_id']
        assert r['corrupt_tokens'][r['s2']]==r['alt_id']
        assert r['token_ids'][r['io_position']]==r['io_id']
        assert r['token_ids'][r['subject_position']]==r['subject_id']
        assert [i for i,(a,b) in enumerate(zip(r['token_ids'],r['corrupt_tokens'])) if a!=b]==[r['s2']]
    data={p[:-4]:load_arrays(directory/'test'/p) for p in record['hashes']}
    for p,sha in record['hashes'].items():assert digest(directory/'test'/p)==sha
    for name,arrays in data.items():
        for key,arr in arrays.items():assert np.isfinite(arr).all(),(name,key)
        if 'raw' in arrays:assert arrays['raw'].shape==(2,384,176 if model=='gpt2-medium' else 192,2)
    donors=donor_indices(rows,'wording');sc=np.asarray(sel['reference']['scale'])[:,None]
    e,r,d=[x.astype(np.float64) for x in (data['wording']['raw'][...,0],data['clean']['raw'][...,0],data['clean']['raw'][...,0][:,donors])]
    er=(e-r)/sc;ed=(e-d)/sc;rd=(d-r)/sc;norm=np.sqrt((rd*rd).sum(-1));direct=(np.sqrt((er*er).sum(-1))-np.sqrt((ed*ed).sum(-1)))/norm
    saved=load_arrays(directory/'analysis/per-example.npz');np.testing.assert_allclose(saved['wording__margin'],direct,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(saved['wording__q_donor']**2-saved['wording__q_recipient']**2,1-2*saved['wording__score'],rtol=1e-12,atol=1e-12)
    summary=read(directory/'analysis/summary.json');assert summary['primary_multiplicity']==24
    for k,x in summary['primary']['recovery'].items():
        w,o=int(k[1]),int(k[4]);mask=np.array([r['wording']==w and r['order']==o for r in rows]);assert x==interval(data['wording']['behavior'][:,0]-data['corrupt']['behavior'][:,0],rows,mask,24)
    dest=f'results/{model}/reproduced';assert not (HERE/dest).exists();analyze004.main(model,dest)
    assert read(HERE/dest/'summary.json')==summary
    rerun=load_arrays(HERE/dest/'per-example.npz');assert set(rerun)==set(saved)
    for k in saved:np.testing.assert_equal(saved[k],rerun[k])
    report=dict(status='passed',utc=now(),raw_bundles=len(record['hashes']),replayed_metric_arrays=len(saved),direct_distance_calculation=True,paired_identity=True,selected_pair_semantic_positions=True,exact_summary_replay=True,exact_interval_replay=True,prior_research_files_unchanged=prior(),input_freeze=digest(HERE/'data/input-freeze.json'),test_freeze=digest(HERE/'data'/f'{model}-test-freeze.json'))
    write(directory/'verification.json',report);print('AUDIT',model,report,flush=True)

def clamps():
    import analyze_clamps004
    verify('input-freeze.json');verify('gpt2-medium-test-freeze.json');directory=HERE/'results/medium-cancellation';rec=read(directory/'run.json');assert rec['status']=='completed'
    for p,sha in rec['hashes'].items():assert digest(directory/p)==sha
    metrics=load_arrays(directory/'analysis/per-example.npz');checks={}
    for f in ['wording','order']:
        data=load_arrays(directory/(f+'.npz'));base=data['base'][None,:,0];edited=data['edited'][None,:,0];noised=data['clamped'][...,0];denoised=data['denoised'][...,0]
        identity=float(np.max(np.abs(data['clamped_base']-data['base'][None])));assert identity<.0005
        np.testing.assert_allclose(metrics[f+'__attenuation'],edited-noised,atol=.0005,rtol=0)
        np.testing.assert_equal(metrics[f+'__PIE'],denoised-base)
        np.testing.assert_allclose(metrics[f+'__INT'],edited-denoised-noised+base,atol=.0005,rtol=0)
        checks[f]=dict(identity_max=identity,four_regime_identity=True,n=384,clamps=19)
    dest='results/medium-cancellation/reproduced';assert not (HERE/dest).exists();analyze_clamps004.main(dest)
    assert read(HERE/dest/'summary.json')==read(directory/'analysis/summary.json')
    rerun=load_arrays(HERE/dest/'per-example.npz')
    for k in metrics:np.testing.assert_equal(metrics[k],rerun[k])
    report=dict(status='passed',utc=now(),checks=checks,exact_summary_replay=True,replayed_metric_arrays=len(metrics),prior_research_files_unchanged=prior());write(directory/'verification.json',report);print('CLAMP AUDIT',report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS+['clamps']);a=p.parse_args();clamps() if a.model=='clamps' else replay(a.model)
