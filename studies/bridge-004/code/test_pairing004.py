"""Regression for donor indexing across the reference and example dimensions."""
import numpy as np
from common004 import *
from geometry004 import paired,self_test

def main():
    rng=np.random.default_rng(4);rows=rows_for('gpt2-medium','test',[1,2]);donors=donor_indices(rows,'wording')
    raw=rng.normal(size=(2,len(rows),7,2));matched=raw[...,0][:,donors]
    expected=np.stack([np.stack([raw[b,j,:,0] for j in donors]) for b in range(2)])
    np.testing.assert_equal(matched,expected);assert matched.shape==(2,384,7)
    for factor in ['wording','order']:
        ix=donor_indices(rows,factor);d=raw[...,0][:,ix];r=raw[...,0];e=r+.3*(d-r)
        g=paired(e,r,d,np.ones((2,1,7)));np.testing.assert_allclose(g['score'],.3,atol=1e-12);np.testing.assert_allclose(g['margin'],-.4,atol=1e-12)
    result=self_test();result.update(explicit_donor_pairing=True,reference_axis_preserved=True,n=384,utc=now());write(HERE/'results/pairing-regression.json',result);print('PAIRING REGRESSION PASSED')
if __name__=='__main__':main()
