"""Post hoc movement and floor sensitivity; never modifies Bridge 005."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'studies/bridge-005/code'))
from common005 import *
OUT=ROOT/'studies/bridge-006/results'


def crossed(values, rows, mask=None, seed=2026100106, multiplicity=1):
    x=np.asarray(values,np.float64)
    if mask is None:mask=np.ones(len(rows),bool)
    ts=sorted({r['template'] for r,k in zip(rows,mask) if k})
    ps=sorted({r['pair'] for r,k in zip(rows,mask) if k})
    grid=np.array([[x[mask & np.array([r['template']==t and r['pair']==p for r in rows])].mean() for p in ps] for t in ts])
    assert np.isfinite(grid).all()
    rng=np.random.default_rng(seed);tt=rng.integers(len(ts),size=(50000,len(ts)));pp=rng.integers(len(ps),size=(50000,len(ps)))
    # Independent implementation: frequencies weight each crossed cell.
    tc=np.stack([(tt==i).sum(1) for i in range(len(ts))],1)
    pc=np.stack([(pp==i).sum(1) for i in range(len(ps))],1)
    boot=np.einsum('bi,ij,bj->b',tc,grid,pc,optimize=False)/(len(ts)*len(ps))
    tail=.05/(2*multiplicity)
    return dict(mean=float(grid.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                adjusted_ci=np.quantile(boot,[tail,1-tail]).tolist(),n=int(mask.sum()),
                blocks=len(ts),triples=len(ps),draws=50000,seed=seed,multiplicity=multiplicity)


def movement(e,c,r,d,scale,rows):
    g=paired(e,r,d,scale);b=paired(c,r,d,scale)
    v=(d-r)/scale;h=(e-c)/scale;n=np.linalg.norm(v,axis=-1)
    ds=np.sum(h*v,axis=-1)/n**2
    perp=h-ds[...,None]*v
    perp_norm=np.linalg.norm(perp,axis=-1)/n
    full=np.linalg.norm(h,axis=-1)/n
    frac=np.sum(perp**2,axis=-1)/np.maximum(np.sum(h**2,axis=-1),1e-24)
    np.testing.assert_allclose(ds,g['score']-b['score'],atol=1e-12)
    np.testing.assert_allclose(full**2,ds**2+perp_norm**2,atol=1e-12)
    identity=float(np.max(abs(g['q_donor']**2-g['q_recipient']**2-(1-2*g['score']))))
    assert identity<1e-11
    arrays=dict(delta_S=ds,orthogonal_displacement=perp_norm,displacement_norm=full,
                orthogonal_energy_fraction=frac,delta_O=g['orthogonal']-b['orthogonal'])
    result=dict(identity_max_error=identity,references={})
    for bi,name in enumerate(['zero','fit_mean']):
        summaries={k:crossed(a[bi],rows) for k,a in arrays.items()}
        for k in ['score','orthogonal','q_recipient','q_donor','margin']:
            summaries['corrupt_'+k]=crossed(b[k][bi],rows)
            summaries['edited_'+k]=crossed(g[k][bi],rows)
        summaries['fraction_displacement_more_orthogonal_than_axial']=float(np.mean(perp_norm[bi]>abs(ds[bi])))
        summaries['fraction_edited_closer_recipient']=float(np.mean(g['margin'][bi]<0))
        result['references'][name]=summaries
    return result,arrays


def main():
    verify('input-freeze.json');verify('test-freeze.json')
    sel=read(HERE/'data/selection.json');rows=rows_for('test',[6,7]);p=HERE/'results/test'
    clean=load_arrays(p/'clean.npz');corrupt=load_arrays(p/'corrupt.npz');edited=load_arrays(p/'wording.npz')
    r=clean['raw'][...,0];d=r[:,donor_indices(rows,'wording')];e=edited['raw'][...,0];c=corrupt['raw'][...,0]
    result,arrays=movement(e,c,r,d,np.array(sel['reference']['scale'])[:,None],rows)
    result['scope']='Post hoc corrupted-to-edited movement; original F6/F7 primary analysis unchanged.'
    result['utc']=now()
    fitrows=rows_for('fit',read(HERE/'results/screen.json')['eligible'])
    ids=np.array([i for i,r in enumerate(fitrows) if r['family'] in [6,7]])
    fitrows=[dict(fitrows[i],wording=int(fitrows[i]['family']==7)) for i in ids]
    x=load_arrays(HERE/'results/fit/clean.npz')['raw'][:,ids][...,0].astype(np.float64)
    assert x.shape==(2,len(fitrows),176)
    labels=np.array([2*r['wording']+r['order'] for r in fitrows])
    means=np.stack([x[:,labels==cell].mean(1) for cell in range(4)],1)
    rms=np.sqrt(((x-means[:,labels])**2).mean(1))
    np.testing.assert_array_equal(np.maximum(rms,.1),np.array(sel['reference']['scale']))
    result['floor_sensitivity']={}
    for floor in [.01,.05,.1,.2,.5,1.]:
        g=paired(e,r,d,np.maximum(rms,floor)[:,None]);cells={}
        for bi,bn in enumerate(['zero','fit_mean']):
            for w in range(2):
                for o in range(2):
                    mask=np.array([r['wording']==w and r['order']==o for r in rows])
                    cells[f'{bn}_w{w}_o{o}']=crossed(g['margin'][bi],rows,mask)
        result['floor_sensitivity'][str(floor)]=dict(cells=cells,all_pointwise_recipient=all(z['ci95'][1]<0 for z in cells.values()),floored_coordinates=(rms<floor).sum(1).tolist())
    # Reproduce original adjusted bounds with independent count-weighted draws.
    original=read(HERE/'results/analysis/summary.json');checks={};scale=np.array(sel['reference']['scale'])[:,None]
    g=paired(e,r,d,scale);gain=edited['behavior'][:,0]-corrupt['behavior'][:,0]
    for w in range(2):
        for o in range(2):
            k=f'w{w}_o{o}';mask=np.array([r['wording']==w and r['order']==o for r in rows])
            for name,z,saved in [('recovery',gain,original['primary']['recovery'][k])]+[(bn,g['margin'][bi],original['primary']['margin'][bn][k]) for bi,bn in enumerate(['zero','fit_mean'])]:
                new=crossed(z,rows,mask,seed=2026090905,multiplicity=12)
                err=max(abs(np.array(new['adjusted_ci'])-saved['adjusted_ci']))
                assert err<1e-10
                checks[f'{name}_{k}']=float(err)
    result['original_primary_independent_bootstrap_errors']=checks
    write(OUT/'f6f7-movement-and-sensitivity.json',result)
    save_arrays(OUT/'f6f7-movement-per-example.npz',**arrays)
    print({b:{k:round(v['mean'],4) for k,v in a.items() if isinstance(v,dict)} for b,a in result['references'].items()})


if __name__=='__main__':main()
