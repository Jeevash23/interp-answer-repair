"""Ordered clean selection, independent qualification and one held-out experiment."""
import argparse,itertools
import numpy as np
import torch
from common005 import *

def clean_outputs(e,rows):
    out={}
    for ids in batches(rows):
        rr=[rows[i] for i in ids];behavior,cache=e.forward(rr);_,z,h=e.extract(rr,cache)
        for key,value in [('behavior',behavior.cpu().numpy()),('z',z.numpy()),('head_results',h.numpy())]:
            if key not in out:out[key]=np.empty((len(rows),)+value.shape[1:],value.dtype)
            out[key][ids]=value
    return out

def behavior_only(e,rows,d):
    out=np.empty((len(rows),6),np.float32)
    for ids in batches(rows):
        rr=[dict(rows[i],token_ids=rows[i]['corrupt_tokens']) for i in ids]
        out[ids]=e.forward(rr,hooks(e,d[ids],rr))[0].cpu().numpy()
    return dict(behavior=out,delta_norm=d.flatten(1).norm(dim=-1).numpy(),head_norms=d.norm(dim=-1).numpy())

def chunked_signatures(bundle,name,e,rows,bank,d=None):
    raw=[];behavior=[]
    for start in range(0,len(rows),128):
        rr=rows[start:start+128];dd=d[start:start+128] if d is not None else None
        part=bundle.get(f'{name}-chunk{start//128:03d}',lambda rr=rr,dd=dd:signature(e,rr,bank,dd))
        raw.append(part['raw']);behavior.append(part['behavior'])
    return dict(raw=np.concatenate(raw,axis=1),behavior=np.concatenate(behavior))

def random_delta(e,d,rows,kind,draw):
    result=[]
    for hi,(l,h) in enumerate(e.source):
        w=e.weights[l,h].cpu();vectors=[]
        for r in rows:
            label=f'{SEED}:{kind}:{draw}:{r["id"]}:{l}:{h}'
            seed=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'little')%(2**63-1)
            vectors.append(torch.randn(w.shape[0] if kind=='projected' else w.shape[1],generator=torch.Generator().manual_seed(seed)))
        v=torch.stack(vectors)
        if kind=='projected':v=v@w
        v*=d[:,hi].norm(dim=-1,keepdim=True)/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        result.append(v)
    return torch.stack(result,1)

def technical():
    verify('input-freeze.json');target=HERE/'results/technical.json'
    if target.exists():assert read(target)['status']=='passed';return
    rows=rows_for('calibration',list(range(10)));rr=[rows[i] for i in next(batches(rows,2))]
    with torch.inference_mode():
        e=Engine003(MODEL);result=e.validate(rr)
    assert result['max_error']<TOL
    write(target,dict(status='passed',utc=now(),checks=result,record_ids=[r['id'] for r in rr]))
    print('TECHNICAL PASSED',result['max_error'],flush=True)

def screen():
    verify('input-freeze.json');assert read(HERE/'results/technical.json')['status']=='passed'
    target=HERE/'results/screen.json'
    if target.exists():return
    rows=rows_for('screen',list(range(10)));b=np.empty((len(rows),6),np.float32)
    with torch.inference_mode():
        e=Engine003(MODEL)
        for ids in batches(rows):b[ids]=e.forward([rows[i] for i in ids])[0].cpu().numpy()
    cells=behavior_cells(b,rows);eligible=[int(f) for f,c in cells.items() if behavior_pass({f:c})]
    pairs=[p for p in read(HERE/'data/design.json')['alignment_eligible_pairs'] if all(f in eligible for f in p)]
    save_arrays(HERE/'results/screen.npz',behavior=b)
    write(target,dict(status='completed',utc=now(),eligible=eligible,pairs=pairs,cells=cells,raw_sha256=digest(HERE/'results/screen.npz')))
    freeze('screen-freeze.json',[HERE/'data/input-freeze.json',target,HERE/'results/screen.npz',HERE/'results/technical.json'])
    print('SCREEN',dict(eligible=eligible,pairs=len(pairs)),flush=True)

def development(split):
    assert split in ['fit','selection'];verify('input-freeze.json');verify('screen-freeze.json')
    families=read(HERE/'results/screen.json')['eligible'];rows=rows_for(split,families)
    bundle=Bundle(HERE/'results'/split,rows,'screen-freeze.json')
    if bundle.record['status']=='completed':return
    with torch.inference_mode():
        e=Engine003(MODEL);natural=bundle.get('natural',lambda:clean_outputs(e,rows))
        bank=torch.tensor(natural['z'].mean(0) if split=='fit' else load_arrays(HERE/'results/fit/bank.npz')['bank'])
        if split=='fit':bundle.get('bank',lambda:dict(bank=bank.numpy()))
        bundle.get('clean',lambda:chunked_signatures(bundle,'clean',e,rows,bank))
    bundle.done()

def select():
    verify('input-freeze.json');verify('screen-freeze.json');target=HERE/'data/selection.json'
    if target.exists():return
    screen=read(HERE/'results/screen.json');families=screen['eligible']
    train=rows_for('fit',families);valid=rows_for('selection',families)
    a=load_arrays(HERE/'results/fit/clean.npz');b=load_arrays(HERE/'results/selection/clean.npz');scores={}
    for fa,fb in screen['pairs']:
        ids=np.array([i for i,r in enumerate(train) if r['family'] in [fa,fb]]);jds=np.array([i for i,r in enumerate(valid) if r['family'] in [fa,fb]])
        rr=[dict(train[i],wording=int(train[i]['family']==fb)) for i in ids];vv=[dict(valid[i],wording=int(valid[i]['family']==fb)) for i in jds]
        fit=fit_reference(a['raw'][:,ids],rr);train_sep=separation(a['raw'][:,ids],rr,fit);valid_sep=separation(b['raw'][:,jds],vv,fit)
        tc=behavior_cells(a['behavior'][ids],rr);vc=behavior_cells(b['behavior'][jds],vv)
        worst=min_separation(valid_sep);average=float(np.mean(list(valid_sep.values())))
        scores[f'{fa},{fb}']=dict(families=[fa,fb],reference=fit,fit_behavior=tc,selection_behavior=vc,fit_separation=train_sep,selection_separation=valid_sep,worst=worst,average=average,eligible=bool(behavior_pass(tc) and behavior_pass(vc) and min_separation(train_sep)>=.8 and worst>=.9))
    good=sorted((v for v in scores.values() if v['eligible']),key=lambda v:(-v['worst'],-v['average'],v['families']))
    selected=good[0] if good else None
    write(target,dict(utc=now(),status='selected' if selected else 'no_qualified_pair',scores=scores,families=selected['families'] if selected else None,reference=selected['reference'] if selected else None,source_heads=read(ROOT/'configs/study.json')['models'][MODEL]['source_heads'],selection_worst=selected['worst'] if selected else None))
    paths=[HERE/'data/input-freeze.json',HERE/'data/screen-freeze.json',target]
    for stage in ['fit','selection']:paths+=list((HERE/'results'/stage).glob('*.json'))+list((HERE/'results'/stage).glob('*.npz'))
    freeze('selection-freeze.json',paths)
    print('SELECTION',dict(families=selected['families'] if selected else None,worst=selected['worst'] if selected else None,qualified_pairs=len(good),best_worst=max(v['worst'] for v in scores.values())),flush=True)

def selected_replay():
    verify('selection-freeze.json');target=HERE/'results/selected-pair-replay.json'
    if target.exists():assert read(target)['status']=='passed';return
    sel=read(HERE/'data/selection.json');rows=rows_for('calibration',sel['families']);allrows=rows_for('calibration',list(range(10)));mapping=donor_indices(rows,'wording')
    with torch.inference_mode():
        e=Engine003(MODEL);nat=clean_outputs(e,allrows);bank=torch.tensor(nat['z'].mean(0))
        expected=np.empty((2,len(rows),176,2),np.float32);replayed=np.empty_like(expected);eb=np.empty((len(rows),6),np.float32);rb=np.empty_like(eb)
        for ids in batches(rows):
            dr=[rows[i] for i in ids];rr=[dict(rows[mapping[i]],token_ids=rows[mapping[i]]['corrupt_tokens']) for i in ids]
            assert all(len(a['token_ids'])==len(b['token_ids']) and a['s2']==b['s2'] for a,b in zip(dr,rr))
            _,cache=e.forward(dr);a,ab=e.signatures(dr,bank);b,bb=e.signatures(rr,bank,boundary=cache[f'blocks.{e.site}.hook_resid_pre'].cpu())
            expected[:,ids]=a;replayed[:,ids]=b;eb[ids]=ab;rb[ids]=bb
    err=float(np.max(np.abs(expected-replayed)));berr=float(np.max(np.abs(eb-rb)));counts=np.count_nonzero(np.abs(expected)>1e-5,axis=(1,2,3)).tolist()
    assert max(err,berr)<TOL and all(x>0 for x in counts)
    save_arrays(HERE/'results/selected-pair-replay.npz',expected_raw=expected,replayed_raw=replayed,expected_behavior=eb,replayed_behavior=rb,bank=bank.numpy())
    write(target,dict(status='passed',utc=now(),n=len(rows),families=sel['families'],directions='both',bank_n=len(allrows),nonzero_by_reference=counts,signature_error=err,behavior_error=berr,tolerance=TOL,raw_sha256=digest(HERE/'results/selected-pair-replay.npz')))
    print('SELECTED PAIR REPLAY',err,berr,flush=True)

def qualification():
    verify('input-freeze.json');verify('selection-freeze.json');sel=read(HERE/'data/selection.json');rows=rows_for('qualification',sel['families'])
    target=HERE/'results/qualification-summary.json'
    if target.exists():return
    bundle=Bundle(HERE/'results/qualification',rows,'selection-freeze.json')
    with torch.inference_mode():
        e=Engine003(MODEL);bank=torch.tensor(load_arrays(HERE/'results/fit/bank.npz')['bank'])
        clean=bundle.get('clean',lambda:chunked_signatures(bundle,'clean',e,rows,bank))
    bundle.done();cells=behavior_cells(clean['behavior'],rows);sep=separation(clean['raw'],rows,sel['reference'])
    passed=behavior_pass(cells) and min_separation(sep)>=.8
    write(target,dict(status='completed',utc=now(),qualified=passed,behavior=cells,separation=sep,worst_separation=min_separation(sep),families=sel['families'],n=len(rows)))
    print('INDEPENDENT QUALIFICATION',passed,min_separation(sep),flush=True)

def test_freeze():
    if (HERE/'data/test-freeze.json').exists():verify('test-freeze.json');return
    verify('selection-freeze.json');assert read(HERE/'results/qualification-summary.json')['qualified']
    assert read(HERE/'results/selected-pair-replay.json')['status']=='passed'
    freeze('test-freeze.json',[HERE/'data/input-freeze.json',HERE/'data/selection-freeze.json',HERE/'data/selection.json',HERE/'results/qualification-summary.json',HERE/'results/selected-pair-replay.json',HERE/'results/selected-pair-replay.npz']+list((HERE/'results/qualification').glob('*')))

def test():
    verify('input-freeze.json');verify('selection-freeze.json');verify('test-freeze.json');sel=read(HERE/'data/selection.json');rows=rows_for('test',sel['families'])
    bundle=Bundle(HERE/'results/test',rows,'test-freeze.json')
    if bundle.record['status']=='completed':return
    with torch.inference_mode():
        e=Engine003(MODEL);bank=torch.tensor(load_arrays(HERE/'results/fit/bank.npz')['bank'])
        clean=bundle.get('natural-clean',lambda:clean_outputs(e,rows))
        corrupt=bundle.get('natural-corrupt',lambda:clean_outputs(e,[dict(r,token_ids=r['corrupt_tokens']) for r in rows]))
        nat={'clean_head_results':clean['head_results'],'corrupt_head_results':corrupt['head_results']}
        for factor in ['clean','corrupt','wording','self','zero','order','role','order_role']+['residual'+str(i) for i in range(3)]+['projected'+str(i) for i in range(3)]:
            def measure(f=factor):
                if f=='clean':return chunked_signatures(bundle,f,e,rows,bank)
                if f=='corrupt':return chunked_signatures(bundle,f,e,[dict(r,token_ids=r['corrupt_tokens']) for r in rows],bank)
                base_factor='wording' if f=='zero' or f.startswith(('residual','projected')) else f
                d=delta(nat,rows,base_factor)
                if f=='zero':d=d*0
                if f.startswith(('residual','projected')):
                    original=d.norm(dim=-1);d=random_delta(e,d,rows,f[:-1],int(f[-1]));assert float((d.norm(dim=-1)-original).abs().max())<.0001
                if f in ['wording','self']:
                    out=chunked_signatures(bundle,f,e,rows,bank,d);out['head_norms']=d.norm(dim=-1).numpy();return out
                return behavior_only(e,rows,d)
            bundle.get(factor,measure)
    bundle.done()

def computational_replay():
    verify('input-freeze.json');verify('test-freeze.json');target=HERE/'results/computational-replay.json'
    if target.exists():assert read(target)['status']=='passed';return
    sel=read(HERE/'data/selection.json');rows=rows_for('test',sel['families']);path=HERE/'results/test';a=load_arrays(path/'natural-clean.npz');b=load_arrays(path/'natural-corrupt.npz')
    nat={'clean_head_results':a['head_results'],'corrupt_head_results':b['head_results']};d=delta(nat,rows,'wording');expected=load_arrays(path/'wording.npz');bank=torch.tensor(load_arrays(HERE/'results/fit/bank.npz')['bank'])
    with torch.inference_mode():
        e=Engine003(MODEL);repeated=behavior_only(e,rows,d);base=behavior_only(e,rows,d*0)
        errors={'wording_behavior':float(np.max(np.abs(repeated['behavior']-expected['behavior']))),'corrupt_behavior':float(np.max(np.abs(base['behavior']-b['behavior'])))}
        # First two examples in each family/order cell, selected by manifest order.
        ids=np.array([i for w in range(2) for o in range(2) for i in [j for j,r in enumerate(rows) if r['wording']==w and r['order']==o][:2]])
        e.config=dict(e.config,head_chunk=1)
        small=signature(e,[rows[i] for i in ids],bank,d[ids]);errors['singleton_signature']=float(np.max(np.abs(small['raw']-expected['raw'][:,ids])))
        assert max(errors.values())<TOL,errors
    save_arrays(HERE/'results/computational-replay.npz',wording_behavior=repeated['behavior'],corrupt_behavior=base['behavior'],singleton_raw=small['raw'],singleton_behavior=small['behavior'],singleton_indices=ids)
    write(target,dict(status='passed',utc=now(),errors=errors,n=len(rows),singleton_n=len(ids),tolerance=TOL,raw_sha256=digest(HERE/'results/computational-replay.npz')))
    print('COMPUTATIONAL REPLAY',errors,flush=True)

def pipeline():
    technical();screen();s=read(HERE/'results/screen.json')
    if not s['pairs']:
        write(HERE/'results/status.json',dict(status='completed',utc=now(),outcome='no_screen_pair',test_measured=False));return
    development('fit');development('selection');select()
    if read(HERE/'data/selection.json')['status']!='selected':
        write(HERE/'results/status.json',dict(status='completed',utc=now(),outcome='no_selection_pair',test_measured=False));return
    selected_replay();qualification()
    if not read(HERE/'results/qualification-summary.json')['qualified']:
        write(HERE/'results/status.json',dict(status='completed',utc=now(),outcome='independent_qualification_failed',test_measured=False));return
    test_freeze();test();computational_replay()
    from analyze005 import main
    main()
    write(HERE/'results/status.json',dict(status='completed',utc=now(),outcome='final_test_completed',test_measured=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['technical','screen','fit','selection','select','selected_replay','qualification','test_freeze','test','computational_replay','pipeline']);a=p.parse_args()
    development(a.stage) if a.stage in ['fit','selection'] else globals()[a.stage]()
