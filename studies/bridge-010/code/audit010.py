"""Independent reconstruction from saved raw observations and explicit bootstrap indices.

Does not call the production geometry, metrics or resampling implementations.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];BASE=ROOT/'studies/bridge-007'


def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def explicit_bootstrap(values,rows,draws,control=None):
    """Literal sampled block × triple × replicate observations, in small chunks."""
    values=np.asarray(values,float)
    if values.ndim==1:values=values[None,:]
    blocks=sorted({r['template'] for r in rows});triples=sorted({r['pair'] for r in rows})
    grid=np.empty((len(values),len(blocks),len(triples)))
    for bi,b in enumerate(blocks):
        for ti,t in enumerate(triples):
            ii=[i for i,r in enumerate(rows) if r['template']==b and r['pair']==t]
            grid[:,bi,ti]=np.mean(values[:,ii],axis=1)
    sampled=np.empty(len(draws['blocks']))
    for start in range(0,len(sampled),256):
        stop=min(start+256,len(sampled));b=draws['blocks'][start:stop];t=draws['triples'][start:stop]
        k=np.zeros((stop-start,1),int) if control is None else draws['replicates'][control,start:stop]
        sampled[start:stop]=grid[k[:,:,None,None],b[:,None,:,None],t[:,None,None,:]].mean((1,2,3))
    return values.mean(),sampled


def main():
    target=HERE/'results/verification.json';assert not target.exists()
    design=read(HERE/'data/design.json');freeze=read(HERE/'data/input-freeze.json')
    for name,expected in freeze['files'].items():assert sha(ROOT/name)==expected,name
    rows=read(HERE/'data/rows.json');n=len(rows)
    keys=[tuple(r[k] for k in ('template','pair','role','wording','order')) for r in rows]
    lookup={key:i for i,key in enumerate(keys)}
    donors=np.array([lookup[(b,t,role,1-w,o)] for b,t,role,w,o in keys])
    maps=load(HERE/'data/draws/mappings.npz')['indices']
    for mapping in maps:
        assert sorted(mapping.tolist())==list(range(n))
        for i,j in enumerate(mapping):
            assert keys[i][0]!=keys[j][0] and keys[i][1]!=keys[j][1] and keys[i][2:]==keys[j][2:]
    with np.load(BASE/'results/test/natural-clean.npz') as z:ds=z['head_results']
    with np.load(BASE/'results/test/natural-corrupt.npz') as z:ds-=z['head_results']
    dw=ds[donors]
    weights=load(HERE/'results/deltas/source-weights.npz')['weights']
    cal=read(HERE/'results/calibration.json');assert cal['passed']
    assert sha(HERE/'results/deltas/source-weights.npz')==cal['weights_sha256']
    baselines={name:load(BASE/f'results/test/{name}.npz') for name in ('clean','self','wording','corrupt')}
    r=baselines['clean']['raw'][...,0].astype(float);scale=np.array(read(BASE/'data/selection.json')['reference']['scale'])[:,None,:]
    axis=(r[:,donors]-r)/scale;den=np.sqrt(np.sum(axis*axis,axis=-1))
    assert np.all(den>=1e-6)
    s=baselines['self']['raw'][...,0].astype(float)
    worddiff=(baselines['wording']['raw'][...,0].astype(float)-s)/scale
    word={'F':np.sqrt(np.sum(worddiff**2,axis=-1))/den,
          'A':np.abs(np.sum(worddiff*axis,axis=-1)/den**2)}
    saved=load(HERE/'results/analysis/per-example.npz');summary=read(HERE/'results/analysis/summary.json')
    draws=load(HERE/'data/draws/bootstrap.npz');errors=[];interval_checks=0;chunk_count=0
    for metric in word:np.testing.assert_allclose(word[metric],saved['word_'+metric],atol=1e-12,rtol=0)
    def verify_interval(record,point,boot):
        nonlocal interval_checks
        tail=.05/(2*record['multiplicity'])
        expected=np.r_[point,np.quantile(boot,[.025,.975,tail,1-tail])]
        actual=np.r_[record['mean'],record['ci95'],record['adjusted_ci']]
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
        interval_checks+=1
    for ci,control in enumerate(design['controls']):
        f=[];a=[];beh=[]
        for k in range(design['replicates']):
            name=f'{control}-r{k}';path=HERE/f'results/test/{name}.npz';z=load(path)
            record=read(HERE/f'results/test/{name}-record.json')
            assert record['status']=='completed' and record['ids']==[r['id'] for r in rows]
            assert sha(path)==record['aggregate_sha256']
            for chunk,expected in record['hashes'].items():
                cp=path.parent/chunk;assert sha(cp)==expected
                c=load(cp);ii=c['_resume_indices']
                assert str(c['_resume_input_key'].item())==record['input_key']
                np.testing.assert_array_equal(c['raw'],z['raw'][:,ii])
                np.testing.assert_array_equal(c['behavior'],z['behavior'][ii]);chunk_count+=1
            actual=load(HERE/f'results/deltas/{name}.npz')['delta']
            gaussian=load(HERE/f'data/draws/random-r{k}.npz')['z'] if control=='random' else None
            for first in range(0,n,128):
                ii=np.arange(first,min(first+128,n)); ss=ds[ii].astype(float);ww=dw[ii].astype(float)
                sh=ss/np.sqrt(np.sum(ss*ss,axis=-1,keepdims=True));ep=ww-ss
                parallel=np.sum(ep*sh,axis=-1,keepdims=True)
                magnitude=np.sqrt(np.maximum(0,np.sum(ep*ep,axis=-1,keepdims=True)-parallel**2))
                j=maps[k,ii]
                if control=='random':vv=np.einsum('nhz,hzd->nhd',gaussian[ii].astype(float),weights.astype(float))
                elif control=='mismatched':vv=dw[j].astype(float)-ds[j]
                else:vv=ds[j].astype(float)-ss
                vv=vv-np.sum(vv*sh,axis=-1,keepdims=True)*sh
                length=np.sqrt(np.sum(vv*vv,axis=-1,keepdims=True))
                assert np.all(length>1e-12)
                expected=(ss+parallel*sh+magnitude*vv/length).astype(np.float32)
                error=float(np.max(np.abs(actual[ii]-expected)));errors.append(error)
                np.testing.assert_allclose(actual[ii],expected,atol=2e-6,rtol=1e-6)
                for aa,bb in [(np.linalg.norm(actual[ii].astype(float),axis=-1),np.linalg.norm(ww,axis=-1)),
                              ((ss*actual[ii]).sum(-1),(ss*ww).sum(-1))]:
                    assert np.all(np.abs(aa-bb)<=1e-6+1e-5*np.abs(bb))
            diff=(z['raw'][...,0].astype(float)-s)/scale
            f.append(np.sqrt(np.sum(diff**2,axis=-1))/den)
            a.append(np.abs(np.sum(diff*axis,axis=-1)/den**2));beh.append(z['behavior'].astype(float))
        metrics={'F':np.stack(f),'A':np.stack(a)};beh=np.stack(beh)
        for metric,values in metrics.items():
            np.testing.assert_allclose(values,saved[control+'_'+metric],atol=1e-12,rtol=0)
            for ri,ref in enumerate(('zero','fit_mean')):
                npnt,nb=explicit_bootstrap(values[:,ri],rows,draws,ci)
                wp,wb=explicit_bootstrap(word[metric][ri],rows,draws)
                entry=summary['controls'][control]['geometry'][ref][metric]
                verify_interval(entry['null'],npnt,nb)
                verify_interval(entry['null_minus_word'],npnt-wp,nb-wb)
                verify_interval(entry['word_over_null'],wp/npnt,wb/nb)
        for metric,values in [('recovery',beh[:,:,0]-baselines['corrupt']['behavior'][None,:,0]),('top1',beh[:,:,3])]:
            p,b=explicit_bootstrap(values,rows,draws,ci)
            entry=summary['controls'][control]['behaviour'][metric]
            verify_interval(entry['null'],p,b)
            for condition in ('self','wording'):
                v=baselines[condition]['behavior'].astype(float)
                bv=v[:,0]-baselines['corrupt']['behavior'][:,0] if metric=='recovery' else v[:,3]
                bp,bb=explicit_bootstrap(bv,rows,draws)
                verify_interval(entry['minus_'+condition],p-bp,b-bb)
        print('Independently reconstructed',control,flush=True)
    replay=read(HERE/'results/singleton-replay.json');assert replay['passed']
    result=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),
                freeze_files=len(freeze['files']),chunks_checked=chunk_count,
                intervals_checked=interval_checks,realised_head_injections=n*12*4*3,
                independent_injection_max_error=max(errors),
                analysis_summary_sha256=sha(HERE/'results/analysis/summary.json'),
                singleton_replay_sha256=sha(HERE/'results/singleton-replay.json'),
                freeze_sha256=sha(HERE/'data/input-freeze.json'))
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
