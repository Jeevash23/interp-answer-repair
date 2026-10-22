"""Independent saved-evidence checks for Appendix H.5, with explicit edge cases."""
import json,sys,tomllib,platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'studies/bridge-005/code'))
from common005 import *
from transport_bridge.engine import outcomes
from activation_transfer.paths import recorded_path


def main():
    out=ROOT/'studies/bridge-006/results';report=dict(utc=now(),checks={})
    checks=report['checks']
    # Fit definitions independently, including the placement of the floor.
    families=read(HERE/'results/screen.json')['eligible'];rr=rows_for('fit',families)
    ids=[i for i,r in enumerate(rr) if r['family'] in [6,7]]
    rr=[dict(rr[i],wording=int(rr[i]['family']==7)) for i in ids]
    raw=load_arrays(HERE/'results/fit/clean.npz')['raw'][:,ids]
    x=raw[...,0].astype(float);labels=np.array([2*r['wording']+r['order'] for r in rr])
    means=np.stack([x[:,labels==k].mean(1) for k in range(4)],1)
    residual=x-means[:,labels]
    rms=np.sqrt((residual**2).mean(1));scale=np.maximum(rms,.1);centers=means/scale[:,None]
    fit=read(HERE/'data/selection.json')['reference']
    np.testing.assert_array_equal(scale,np.array(fit['scale']))
    np.testing.assert_array_equal(centers,np.array(fit['centers']))
    checks['fit_scale_and_centers_exact']=True
    qr=rows_for('qualification',[6,7]);qa=load_arrays(HERE/'results/qualification/clean.npz')
    qs=read(HERE/'results/qualification-summary.json')
    qb=behavior_cells(qa['behavior'],qr);qsep=separation(qa['raw'],qr,fit)
    assert qs['behavior']==qb and qs['separation']==qsep
    assert qs['qualified']==bool(behavior_pass(qb) and min_separation(qsep)>=.8)
    checks['independent_qualification_recomputed']=True
    rr=rows_for('test',[6,7]);clean=load_arrays(HERE/'results/test/clean.npz');edited=load_arrays(HERE/'results/test/wording.npz')
    coords=axis_coordinates(clean['raw'],rr,fit,'wording')
    x=clean['raw'][...,0].astype(float)/scale[:,None]
    nearest=np.empty((2,len(rr)),bool)
    min_axis=np.inf
    for i,r in enumerate(rr):
        a=centers[:,r['order']];b=centers[:,2+r['order']]
        min_axis=min(min_axis,float(np.linalg.norm(b-a,axis=1).min()))
        nearest[:,i]=((x[:,i]-b)**2).sum(1)<((x[:,i]-a)**2).sum(1)
    np.testing.assert_array_equal(coords>0,nearest)
    checks['classification_nearest_scaled_centroid_exact']=True
    report['classifier']=dict(tie_rule='score > 0 is family 1; exact midpoint goes to family 0',
                             min_centroid_axis_norm=min_axis,nonfinite_scores=int((~np.isfinite(coords)).sum()),
                             floor='max(sqrt(mean(within-cell residual squared)), 0.1); equal weight per fitting example',
                             floor_coordinate_counts=(rms<.1).sum(1).tolist())
    # Test a midpoint and a degenerate axis without imposing an invented policy.
    z=np.zeros((2,4,1,2)); tr=[dict(wording=w,order=o) for w in range(2) for o in range(2)]
    f=dict(scale=[[1.],[1.]],centers=[[[-1.],[-1.],[1.],[1.]]]*2)
    assert (axis_coordinates(z,tr,f,'wording')==0).all()
    with np.errstate(invalid='ignore',divide='ignore'):
        dg=separation(z,tr,dict(scale=[[1.],[1.]],centers=[[[0.]]*4]*2))
    assert min_separation(dg)==0
    report['classifier']['degenerate_axis_handling']='NaN score compares false; both family-1 cells have zero classification, so the all-cell gate fails. No observed axes are degenerate.'
    checks['classification_midpoint_and_degenerate_gate']=True
    # Exactly zero, subthreshold nonzero, and threshold paired norms.
    recipient=np.zeros((3,2));donor=np.array([[0.,0.],[.999e-6,0.],[1e-6,0.]])
    g=paired(recipient,recipient,donor,np.ones(2))
    np.testing.assert_array_equal(g['resolvable'],[False,False,True])
    assert np.isnan(g['score'][:2]).all() and g['score'][2]==0
    checks['paired_small_norm_and_nan_retention']=True
    r=clean['raw'][...,0];d=r[:,donor_indices(rr,'wording')]
    g=paired(edited['raw'][...,0],r,d,scale[:,None])
    assert g['resolvable'].all()
    report['pair_norm_min']=g['pair_norm'].min(1).tolist()
    report['pair_norm_max']=g['pair_norm'].max(1).tolist()
    assert np.max(abs(g['q_donor']**2-g['q_recipient']**2-(1-2*g['score'])))<1e-11
    np.testing.assert_array_equal(g['margin']<0,g['score']<.5)
    checks['per_example_distance_projection_identity']=True
    # Counterexample to the invalid aggregate equivalence.
    s=np.array([-1.,1.1]);o=np.array([100.,0.]);m=np.sqrt(s*s+o*o)-np.sqrt((s-1)**2+o*o)
    assert s.mean()<.5 and m.mean()>0
    checks['mean_projection_not_mean_unsquared_margin']=True
    # Accuracy ties: pairwise is strict; argmax keeps the first token index.
    fake=[dict(io_id=0,subject_id=1,alt_id=2),dict(io_id=1,subject_id=0,alt_id=2)]
    y=outcomes(torch.tensor([[[2.,2.,0.]],[[2.,2.,0.]]]),fake).numpy()
    np.testing.assert_array_equal(y[:,2],[0,0]);np.testing.assert_array_equal(y[:,3],[1,0])
    report['behavior_ties']='Pairwise requires strictly greater logit. Top-1 uses first-index argmax, including ties; not all maxima count as correct.'
    checks['behavior_ties_explicit']=True
    # Exact involution explains the opposite control means.
    t=donor_indices(rr,'order_role')
    assert all(rr[i]['corrupt_tokens']==rr[j]['corrupt_tokens'] and rr[i]['io_id']==rr[j]['subject_id'] for i,j in enumerate(t))
    errors={}
    for a,b in [('order','role'),('order_role','self')]:
        aa=load_arrays(HERE/f'results/test/{a}.npz')['behavior'][:,0]
        bb=load_arrays(HERE/f'results/test/{b}.npz')['behavior'][t,0]
        errors[a+'_'+b]=float(abs(aa+bb).max());assert errors[a+'_'+b]<.0005
    checks['opposite_control_mapping_verified']=True
    report['opposite_control_max_errors']=errors
    # Recover the Medium discovery ranking using all saved individual effects.
    base=ROOT/'studies/transport'
    observations=[json.loads(x) for x in (base/'results/experiment-002/localization/head-effects-per-example.jsonl').read_text().splitlines()]
    selection=read(base/'results/experiment-002/localization/selected-heads.json');union=set();rankings={}
    for condition in ['canonical','controlled']:
        obs=[o for o in observations if o['condition']==condition and o['site']=='s2'];assert len(obs)==80
        scores=np.array([o['effects'] for o in obs]).mean(0)
        order=sorted(range(384),key=lambda i:(-scores[i],i))
        heads=[[i//16,i%16] for i in order[:8]]
        assert heads==selection['conditions'][condition]['s2']['selected_heads']
        union.update(map(tuple,heads));rankings[condition]=dict(heads=heads,cutoff_gap=float(scores[order[7]]-scores[order[8]]))
    assert sorted(union)==list(map(tuple,read(ROOT/'configs/study.json')['models']['gpt2-medium']['source_heads']))
    report['medium_source_selection']=dict(examples=80,candidate_heads=384,per_condition_top=8,union=sorted(union),rankings=rankings,
        freeze_utc=read(base/'data/experiment-002-implementation-freeze.json')['frozen_at_utc'])
    checks['historical_medium_ranking_and_union_exact']=True
    manifest=[json.loads(x) for x in (base/'data/experiment-001-manifest.jsonl').read_text().splitlines()]
    assert digest(base/'data/experiment-001-manifest.jsonl')==read(base/'data/experiment-001-freeze.json')['manifest_sha256']
    report['transport_example']=manifest[0]
    checks['historical_transport_manifest_hash']=True
    report['group_control_heads']=read(ROOT/'studies/bridge-003/data/medium-selection.json')['groups']
    # Installed environment versus active resolution, not historical version guesses.
    lock=tomllib.loads((ROOT/'uv.lock').read_text())
    locked={p['name']:p['version'] for p in lock['package'] if 'version' in p}
    report['environment']=dict(python=platform.python_version(),cuda_available=torch.cuda.is_available(),packages={})
    for name in ['torch','transformer-lens','transformers','numpy','scipy']:
        installed=version(name); expected=locked[name]
        assert installed.split('+')[0]==expected.split('+')[0],(name,installed,expected)
        report['environment']['packages'][name]=dict(installed=installed,locked=expected)
    report['environment']['lock_sha256']=digest(ROOT/'uv.lock')
    checks['active_environment_matches_lock']=True
    # Check all Medium/TinyLlama checkpoint bytes, including actual weight files.
    verified=[]
    cp=read(ROOT/'studies/bridge-004/data/checkpoints.json')
    for name,rec in cp.items():
        p=recorded_path(ROOT/'studies/bridge-004'/name)
        assert p.stat().st_size==rec['bytes'] and digest(p)==rec['sha256'],p
        verified.append(dict(path=str(p.resolve().relative_to(ROOT)),**rec))
    report['verified_checkpoint_files']=verified
    checks['checkpoint_weights_and_tokenizers_hashes']=True
    # All frozen references still verify; files were restored, never skipped.
    for name in ['input-freeze.json','selection-freeze.json','test-freeze.json']:verify(name)
    checks['original_main_freezes_verify']=True
    times={k:read(HERE/p)[field] for k,p,field in [('selection','data/selection-freeze.json','utc'),('test_freeze','data/test-freeze.json','utc'),('test_start','results/test/run.json','started')]}
    assert times['selection']<times['test_freeze']<times['test_start']
    report['main_local_freeze_times']=times;checks['local_freezes_precede_test']=True
    report['passed']=all(checks.values());write(out/'methods-audit.json',report)
    print('METHODS AUDIT',len(checks),'checks passed')


if __name__=='__main__':main()
