"""Versioned, resumable selection and fresh held-out measurements."""
import argparse,itertools,time
import numpy as np
import torch
from common004 import *

class Bundle:
    def __init__(self,path,rows,freeze_name):
        self.path=path;path.mkdir(parents=True,exist_ok=True);self.rp=path/'run.json'
        self.record=read(self.rp) if self.rp.exists() else dict(started=now(),status='running',hashes={},record_ids=[r['id'] for r in rows],freeze=digest(HERE/'data'/freeze_name))
        assert self.record['record_ids']==[r['id'] for r in rows]
        assert self.record['freeze']==digest(HERE/'data'/freeze_name)
        for p,h in self.record['hashes'].items():assert digest(path/p)==h,p
        write(self.rp,self.record)
    def get(self,name,fn):
        p=self.path/(name+'.npz')
        if p.name in self.record['hashes']:return load_arrays(p)
        assert not p.exists(),p
        data=fn();save_arrays(p,**data);self.record['hashes'][p.name]=digest(p);write(self.rp,self.record)
        print('SAVED',str(p.relative_to(HERE)),now(),flush=True);return data
    def done(self):self.record.update(status='completed',finished=now());write(self.rp,self.record)

def screen(model):
    verify('input-freeze.json');out=HERE/'results'/model;out.mkdir(parents=True,exist_ok=True)
    target=out/'screen.json';assert not target.exists()
    if model=='gpt2-medium':
        write(target,dict(utc=now(),eligible=[1,2],retained=[1,2],fixed_prior_pair=True));return
    rows=rows_for(model,'screen',list(range(8)));b=np.empty((len(rows),6),np.float32)
    with torch.inference_mode():
        engine=engine_for(model)
        for ix in batches(rows):b[ix]=engine.forward([rows[i] for i in ix])[0].cpu().numpy()
    scores={};eligible=[]
    for f in range(8):
        cells={}
        for o in range(2):
            mask=np.array([r['family']==f and r['order']==o for r in rows]);v=b[mask]
            cells[str(o)]=dict(pairwise=float(v[:,2].mean()),top1=float(v[:,3].mean()),logit=float(v[:,0].mean()),n=len(v))
        good=all(x['pairwise']>=.95 and x['top1']>=.85 for x in cells.values());scores[str(f)]=dict(cells=cells,eligible=good)
        if good:eligible.append(f)
    retained=sorted(set(eligible[:4]+[1,2]));save_arrays(out/'screen.npz',behavior=b)
    write(target,dict(utc=now(),eligible=eligible,retained=retained,scores=scores,raw_sha256=digest(out/'screen.npz')))
    print('SCREEN',model,eligible,retained,flush=True)

def discovery(model):
    verify('input-freeze.json');verify(model+'-discovery-freeze.json')
    families=read(HERE/'results'/model/'screen.json')['retained'];rows=rows_for(model,'discovery',families)
    bundle=Bundle(HERE/'results'/model/'discovery',rows,model+'-discovery-freeze.json')
    if bundle.record['status']=='completed':return
    with torch.inference_mode():
        e=engine_for(model);nat=bundle.get('natural',lambda:natural(e,rows));bank=torch.tensor(nat['clean_z'].mean(0))
        bundle.get('signature',lambda:dict(**signature(e,rows,bank),bank=bank.numpy()))
    bundle.done()

def score_pairs(model):
    screen=read(HERE/'results'/model/'screen.json');rows=rows_for(model,'discovery',screen['retained'])
    raw=load_arrays(HERE/'results'/model/'discovery/signature.npz')['raw'];scores={}
    pairs=sorted(set(itertools.combinations(screen['eligible'],2))|{(1,2)})
    for a,b in pairs:
        ids=np.array([i for i,r in enumerate(rows) if r['family'] in [a,b]])
        if len(ids)!=128:continue
        rr=[dict(rows[i],wording=int(rows[i]['family']==b)) for i in ids];x=raw[:,ids];pred=np.empty(x.shape[:2],bool)
        for block in range(4):
            train=np.array([r['template']!=block for r in rr]);test=~train
            ref=fit_reference(x[:,train],[r for r,k in zip(rr,train) if k])
            pred[:,test]=axis_coordinates(x[:,test],[r for r,k in zip(rr,test) if k],ref,'wording')>0
        cells={}
        for w in range(2):
            for o in range(2):
                mask=np.array([r['wording']==w and r['order']==o for r in rr]);cells[f'w{w}_o{o}']=(pred[:,mask]==bool(w)).mean(1).tolist()
        worst=min(v for a0 in cells.values() for v in a0)
        scores[f'{a},{b}']=dict(families=[a,b],cells=cells,worst=worst,qualified=worst>=.8 and a in screen['eligible'] and b in screen['eligible'])
    return rows,raw,scores

def select_pair(model):
    verify('input-freeze.json');verify(model+'-discovery-freeze.json')
    target=HERE/'data'/f'{model}-selection.json';assert not target.exists()
    rows,raw,scores=score_pairs(model)
    good=sorted([k for k,s in scores.items() if s['qualified']],key=lambda k:(-scores[k]['worst'],k))
    key='1,2' if model=='gpt2-medium' or scores['1,2']['qualified'] else (good[0] if good else '1,2')
    families=scores[key]['families'];ids=np.array([i for i,r in enumerate(rows) if r['family'] in families]);rr=[dict(rows[i],wording=families.index(rows[i]['family'])) for i in ids]
    write(target,dict(utc=now(),families=families,discovery_qualified=scores[key]['qualified'],scores=scores,reference=fit_reference(raw[:,ids],rr),source_heads=read(ROOT/'configs/study.json')['models'][model]['source_heads'] if model=='gpt2-medium' else None))
    print('PAIR',model,families,'qualified',scores[key]['qualified'],flush=True)

def localize(model):
    assert model=='tinyllama-1.1b-base';verify('input-freeze.json');verify(model+'-discovery-freeze.json')
    sel=read(HERE/'data'/f'{model}-selection.json');assert sel['source_heads'] is None
    rows=rows_for(model,'discovery',sel['families']);bundle=Bundle(HERE/'results'/model/'localization',rows,model+'-discovery-freeze.json')
    with torch.inference_mode():
        e=engine_for(model)
        # Per-batch files make the 512-head scan resumable without selective omission.
        pieces=[];order=[]
        for bi,ids in enumerate(batches(rows)):
            def measure(ids=ids):
                rr=[rows[i] for i in ids];cc=[dict(r,token_ids=r['corrupt_tokens']) for r in rr]
                _,clean=e.forward(rr);base,corrupt=e.forward(cc);n=len(rr)
                pos=torch.tensor([r['s2'] for r in rr],device=e.device);effects=np.empty((n,512),np.float32)
                for layer in range(16):
                    for h0 in range(0,32,4):
                        heads=list(range(h0,h0+4));expanded=corrupt[f'blocks.{layer}.hook_resid_pre'].repeat(4,1,1)
                        ix=torch.arange(n,device=e.device);repl=torch.stack([clean[f'blocks.{layer}.attn.hook_z'][ix,pos,h] for h in heads]).reshape(4*n,64)
                        positions=pos.repeat(4);which=torch.tensor(heads,device=e.device).repeat_interleave(n)
                        def patch(v,hook,repl=repl,positions=positions,which=which):
                            out=v.clone();out[torch.arange(len(out),device=out.device),positions,which]=repl;return out
                        b=outcomes(e.suffix(expanded,layer,[(f'blocks.{layer}.attn.hook_z',patch)]),cc*4).reshape(4,n,6)
                        effects[:,layer*32+h0:layer*32+h0+4]=(b[:,:,0]-base[None,:,0]).T.cpu().numpy()
                return dict(effects=effects,indices=ids)
            data=bundle.get(f'batch{bi:03d}',measure);pieces.append(data['effects']);order.extend(data['indices'])
        all_effects=np.concatenate(pieces)[np.argsort(order)];save_arrays(bundle.path/'all-effects.npz',effects=all_effects)
        bundle.record['hashes']['all-effects.npz']=digest(bundle.path/'all-effects.npz');bundle.done()
    rankings={};chosen=set()
    for f in sel['families']:
        means=all_effects[np.array([r['family']==f for r in rows])].mean(0);rank=sorted(range(512),key=lambda i:(-float(means[i]),i))
        rankings[str(f)]=[dict(layer=i//32,head=i%32,mean=float(means[i])) for i in rank];chosen|={(i//32,i%32) for i in rank[:6]}
    sel.update(source_heads=sorted(chosen),source_rankings=rankings,localization_sha256=digest(bundle.path/'all-effects.npz'),source_selected_utc=now());write(HERE/'data'/f'{model}-selection.json',sel)
    print('SOURCE',model,sel['source_heads'],flush=True)

def behavior_only(e,rows,d):
    out=np.empty((len(rows),6),np.float32)
    for ids in batches(rows):
        rr=[dict(rows[i],token_ids=rows[i]['corrupt_tokens']) for i in ids];out[ids]=e.forward(rr,hooks(e,d[ids],rr))[0].cpu().numpy()
    return {'behavior':out,'delta_norm':d.flatten(1).norm(dim=-1).numpy()}

def test(model):
    verify('input-freeze.json');verify(model+'-test-freeze.json')
    sel=read(HERE/'data'/f'{model}-selection.json');rows=rows_for(model,'test',sel['families']);bundle=Bundle(HERE/'results'/model/'test',rows,model+'-test-freeze.json')
    if bundle.record['status']=='completed':return
    with torch.inference_mode():
        e=engine_for(model,sel['source_heads']);nat=bundle.get('natural',lambda:natural(e,rows));bank=torch.tensor(load_arrays(HERE/'results'/model/'discovery/signature.npz')['bank'])
        for f in ['clean','corrupt','zero','self','wording','order','role','order_role','random0','random1','random2']:
            def measure(f=f):
                if f=='clean':return signature(e,rows,bank)
                if f=='corrupt':return signature(e,[dict(r,token_ids=r['corrupt_tokens']) for r in rows],bank)
                d=delta(nat,rows,'wording' if f.startswith('random') or f=='zero' else f)
                if f.startswith('random'):d=random_matched(d,[r['id'] for r in rows],'bridge004-wording',int(f[-1]))
                if f=='zero':d=d*0
                if f in ['role','order_role','zero']:return behavior_only(e,rows,d)
                result=signature(e,rows,bank,d);result['delta_norm']=d.flatten(1).norm(dim=-1).numpy();return result
            bundle.get(f,measure)
    bundle.done()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen','discovery','select_pair','localize','test']);p.add_argument('--model',required=True,choices=MODELS);a=p.parse_args();globals()[a.stage](a.model)
