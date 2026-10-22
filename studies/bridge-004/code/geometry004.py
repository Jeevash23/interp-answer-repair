"""Prospective paired endpoints: proximity is distinct from donor matching."""
import numpy as np

def paired(edited,recipient,donor,scale):
    e,r,d=[np.asarray(x,dtype=np.float64) for x in (edited,recipient,donor)]
    s=np.asarray(scale,dtype=np.float64)
    u=(e-r)/s;v=(d-r)/s;n=np.linalg.norm(v,axis=-1);valid=n>=1e-6
    safe=np.where(valid,n,np.nan);score=np.sum(u*v,axis=-1)/(safe**2)
    qr=np.linalg.norm(u,axis=-1)/safe;qd=np.linalg.norm(u-v,axis=-1)/safe
    orth=np.linalg.norm(u-score[...,None]*v,axis=-1)/safe
    def cos(a,b):
        den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        return np.sum(a*b,axis=-1)/np.where(den>=1e-12,den,np.nan)
    return {'score':score,'q_recipient':qr,'q_donor':qd,'margin':qr-qd,'orthogonal':orth,'pair_norm':n,'resolvable':valid,'cosine_margin':cos(e/s,d/s)-cos(e/s,r/s)}

def self_test():
    r=np.array([[1.,2.,3.],[2.,-1.,0.]])
    v=np.array([[2.,0.,0.],[0.,3.,0.]])
    d=r+v;s=np.ones_like(r);checks=[]
    for t in [0.,.25,.5,1.,1.5]:
        g=paired(r+t*v,r,d,s)
        for k,expected in [('score',t),('q_recipient',abs(t)),('q_donor',abs(t-1)),('margin',abs(t)-abs(t-1)),('orthogonal',0.)]:
            np.testing.assert_allclose(g[k],expected,atol=1e-12);checks.append(f't={t}:{k}')
    e=r+.3*v+np.array([[0.,0.,2.],[0.,0.,3.]])
    g=paired(e,r,d,s);np.testing.assert_allclose(g['orthogonal'],1.)
    np.testing.assert_allclose(g['q_donor']**2-g['q_recipient']**2,1-2*g['score'],atol=1e-12)
    assert not paired(r,r,r,s)['resolvable'].any()
    # Nonuniform scaling, batching and independent direct distance calculations.
    rng=np.random.default_rng(42);a,b,c=rng.normal(size=(3,2,10,7));scale=np.exp(rng.normal(size=(2,1,7)))
    g=paired(a,b,c,scale)
    np.testing.assert_allclose(g['margin'],(np.linalg.norm((a-b)/scale,axis=-1)-np.linalg.norm((a-c)/scale,axis=-1))/np.linalg.norm((c-b)/scale,axis=-1))
    np.testing.assert_allclose(g['q_donor']**2-g['q_recipient']**2,1-2*g['score'],atol=1e-12)
    return {'passed':True,'endpoint_checks':checks,'orthogonal_check':True,'degenerate_flag':True,'distance_identity':True,'nonuniform_scale_direct_distance':True}

if __name__=='__main__':
    import json
    print(json.dumps(self_test(),indent=2))
