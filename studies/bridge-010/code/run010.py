"""Study 5 calibration and inference. Any failed calibration blocks the full run."""
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from common010 import *
from runner007 import (make_engine,save_arrays,load_arrays,digest,now,verify_freeze,
                       measure_signatures,batches)
from common005 import hooks
from cpu_engine import Engine003


def save_fixed(path, **arrays):
    if path.exists():
        previous=load_arrays(path)
        assert previous.keys()==arrays.keys()
        for k in arrays: np.testing.assert_array_equal(previous[k],arrays[k])
    else: save_arrays(path,**arrays)


def injections(e):
    """Materialise all draws before scientific inference; inspect every head."""
    ds,dw=base_deltas(); rows=read(HERE/'data/rows.json')
    maps=load(HERE/'data/draws/mappings.npz')['indices']
    records={}
    for control in CONTROLS:
        for k in range(REPLICATES):
            path=HERE/f'results/deltas/{control}-r{k}.npz'
            output=np.empty_like(ds); diagnostics=[]
            z=load(HERE/f'data/draws/random-r{k}.npz')['z'] if control=='random' else None
            for start in range(0,len(rows),128):
                ix=np.arange(start,min(start+128,len(rows))); j=maps[k,ix]
                s=ds[ix].astype(float);w=dw[ix].astype(float)
                if control=='random':
                    v=np.stack([z[ix,hi].astype(float) @ e.weights[l,h].cpu().numpy().astype(float)
                                for hi,(l,h) in enumerate(e.source)],1)
                elif control=='mismatched': v=dw[j].astype(float)-ds[j]
                else: v=ds[j].astype(float)-s
                out,detail=geometry_match(s,w,v)
                diagnostics.append(check_geometry(s,w,out));output[ix]=out
                assert diagnostics[-1]['passed'],(control,k,start,diagnostics[-1])
            save_fixed(path,delta=output)
            summary=dict(passed=all(x['passed'] for x in diagnostics),
                         n=sum(x['n'] for x in diagnostics),sha256=digest(path))
            for name in ('norm','cross','perturbation'):
                summary[name]=dict(violations=sum(x[name]['violations'] for x in diagnostics),
                     max_absolute_error=max(x[name]['max_absolute_error'] for x in diagnostics),
                     max_tolerance_fraction=max(x[name]['max_tolerance_fraction'] for x in diagnostics))
            records[f'{control}-r{k}']=summary
            print(now(),'geometry checked',control,k,flush=True)
            del output,z
    save_fixed(HERE/'results/deltas/source-weights.npz',
               weights=np.stack([e.weights[l,h].cpu().numpy() for l,h in e.source]),
               source_heads=np.asarray(e.source))
    return records


@torch.inference_mode()
def calibration(e):
    target=HERE/'results/calibration.json'
    if target.exists():
        old=read(target)
        assert old['passed'],'Calibration failed earlier: author review required before any restart'
        assert old['freeze_sha256']==digest(FREEZE)
        for name,item in old['geometry'].items():
            assert digest(HERE/f'results/deltas/{name}.npz')==item['sha256']
        return
    record=dict(started=now(),freeze_sha256=digest(FREEZE),passed=False,
                tolerance=TOL,geometry_tolerance=dict(absolute=GEOM_ATOL,relative=GEOM_RTOL),checks={})
    try:
        record['geometry']=injections(e)
        record['weights_sha256']=digest(HERE/'results/deltas/source-weights.npz')
        rows=read(HERE/'data/calibration-rows.json');donors=validate_rows(rows)
        cr=[dict(r,token_ids=r['corrupt_tokens']) for r in rows]
        bank=torch.tensor(load(BASE/'results/fit/bank.npz')['bank'])
        heads={}
        for name,rr in [('clean',rows),('corrupt',cr)]:
            b,cache=e.forward(rr);_,_,h=e.extract(rr,cache);heads[name]=h.numpy()
        ds=heads['clean']-heads['corrupt']; dw=ds[donors]
        zero,_=geometry_match(ds,ds,np.zeros_like(ds))
        own,_=geometry_match(ds,dw,dw.astype(float)-ds)
        record['checks']['own_direction_geometry']=check_geometry(ds,dw,own)
        assert record['checks']['own_direction_geometry']['passed']
        outputs={}
        for name,d in [('self',ds),('zero',zero),('wording',dw),('own_direction',own)]:
            raw,b=e.signatures(cr,bank,hooks(e,torch.tensor(d),cr))
            outputs[name]=dict(raw=raw,behavior=b)
            save_fixed(HERE/f'results/calibration/{name}.npz',raw=raw,behavior=b,delta=d)
        for new,old in [('zero','self'),('own_direction','wording')]:
            for metric in ('raw','behavior'):
                error=float(np.max(np.abs(outputs[new][metric]-outputs[old][metric])))
                record['checks'][new+'_'+metric+'_error']=error
                assert error <= TOL,(new,metric,error)
        # Full-sequence reference confirms the inherited causal-prefix shortcut.
        raw,b=Engine003.signatures(e,cr,bank,hooks(e,torch.tensor(own),cr))
        save_fixed(HERE/'results/calibration/full-sequence.npz',raw=raw,behavior=b)
        for metric,value in [('raw',raw),('behavior',b)]:
            error=float(np.max(abs(value-outputs['own_direction'][metric])))
            record['checks']['full_sequence_'+metric+'_error']=error
            assert error <= TOL,(metric,error)
        record.update(passed=True,n=len(rows),ids=[r['id'] for r in rows],
                      checked_injections=sum(v['n'] for v in record['geometry'].values()))
    except Exception as error:
        record.update(error=str(error),traceback=traceback.format_exc(),finished=now())
        write(target,record)
        write(HERE/'results/status.json',dict(status='calibration_failed',full_experiment_started=False,
                     reason=str(error),author_review_required=True,utc=now()))
        raise
    record['finished']=now();write(target,record)
    print(now(),'ALL CALIBRATION CHECKS PASSED',flush=True)


@torch.inference_mode()
def inference(e):
    calibrated=read(HERE/'results/calibration.json');assert calibrated['passed']
    assert calibrated['freeze_sha256']==digest(FREEZE)
    rows=read(HERE/'data/rows.json');bank=load(BASE/'results/fit/bank.npz')['bank']
    outputs={};start=time.perf_counter()
    for control in CONTROLS:
        for k in range(REPLICATES):
            name=f'{control}-r{k}';dp=HERE/f'results/deltas/{name}.npz'
            assert digest(dp)==calibrated['geometry'][name]['sha256']
            d=load(dp)['delta'];op=HERE/f'results/test/{name}.npz'
            write(HERE/'results/status.json',dict(status='inference',condition=name,
                    completed=len(outputs),conditions=12,utc=now()))
            measure_signatures(e,rows,bank,op,FREEZE,d)
            outputs[name]=dict(sha256=digest(op),delta_sha256=digest(dp))
            del d;gc.collect()
    target=HERE/'results/inference-complete.json'
    if not target.exists():
        write(target,dict(passed=True,utc=now(),outputs=outputs,n=len(rows),
                         freeze_sha256=digest(FREEZE),invocation_seconds=time.perf_counter()-start))
    else: assert read(target)['outputs']==outputs


@torch.inference_mode()
def replay(e):
    assert read(HERE/'results/inference-complete.json')['passed']
    path=HERE/'results/singleton-replay.json'
    if path.exists():assert read(path)['passed'];return
    # This stage is launched in a fresh process/engine and uses one head variant
    # per reference rather than the original eight-head measurement chunks.
    e.config['head_chunk']=1
    rows=read(HERE/'data/rows.json');ii=read(HERE/'data/design.json')['singleton_indices']
    bank=torch.tensor(load(BASE/'results/fit/bank.npz')['bank']);checks={}
    for control in CONTROLS:
        for k in range(REPLICATES):
            name=f'{control}-r{k}'; d=load(HERE/f'results/deltas/{name}.npz')['delta'][ii]
            rr=[dict(rows[i],token_ids=rows[i]['corrupt_tokens']) for i in ii]
            raw=np.empty((2,len(ii),176,2),np.float32); b=np.empty((len(ii),6),np.float32)
            for ids in batches(rr,1):
                one=[rr[i] for i in ids]
                x,y=e.signatures(one,bank,hooks(e,torch.tensor(d[ids]),one))
                raw[:,ids]=x;b[ids]=y
            old=load(HERE/f'results/test/{name}.npz')
            errors=dict(raw=float(np.max(abs(raw-old['raw'][:,ii]))),
                        behavior=float(np.max(abs(b-old['behavior'][ii]))))
            save_fixed(HERE/f'results/singletons/{name}.npz',raw=raw,behavior=b,indices=np.asarray(ii))
            checks[name]=errors
            if max(errors.values())>TOL:
                write(path,dict(passed=False,checks=checks,tolerance=TOL,utc=now()))
                raise AssertionError((name,errors))
            print(now(),'singleton replay',name,errors,flush=True)
    write(path,dict(passed=True,checks=checks,tolerance=TOL,utc=now(),n=len(ii),
                   fresh_engine=True,head_chunk=1,prompt_batch=1,freeze_sha256=digest(FREEZE)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['calibration','inference','replay'])
    args=parser.parse_args();verify_freeze(FREEZE)
    e=make_engine(4)
    assert e.source==[tuple(h) for h in read(HERE/'data/design.json')['source_heads']]
    globals()[args.stage](e)


if __name__=='__main__': main()
