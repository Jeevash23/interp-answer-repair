"""Frozen geometry for the approved exploratory Study 5. No inference on import."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = ROOT / 'studies/bridge-007'
sys.path.insert(0, str(BASE / 'code'))
from analyze007 import read, write, load, rows_for, validate_rows, cell_masks, paired, Crossed

CONTROLS = ('random', 'mismatched', 'natural')
REFS = ('zero', 'fit_mean')
REPLICATES = 4
DRAWS = 50_000
SEED = 2026100501
ANALYSIS_SEED = 2026100502
TOL = 5e-4
GEOM_ATOL = 1e-6
GEOM_RTOL = 1e-5
DEGENERATE = 1e-12
# All paired geometric contrasts (12), plus recovery and top-1 contrasts
# against self and wording (12): one conservative family of 24 contrasts.
MULTIPLICITY = 24
FREEZE = HERE / 'data/input-freeze.json'


def base_deltas():
    # Subtract in the original float32 representation before float64 geometry.
    with np.load(BASE / 'results/test/natural-clean.npz') as z:
        clean = z['head_results']
    with np.load(BASE / 'results/test/natural-corrupt.npz') as z:
        ds = clean - z['head_results']
    rows = read(HERE / 'data/rows.json')
    return ds, ds[validate_rows(rows)]


def geometry_match(self_delta, wording_delta, direction):
    """Keep the parallel component; turn only the perpendicular component.

    Undefined directions invalidate the entire replicate; no redraw or omission.
    Calculation is float64 and the realised joint injection is float32.
    """
    s, w, v = (np.asarray(x, dtype=np.float64) for x in
               (self_delta, wording_delta, direction))
    sn = np.linalg.norm(s, axis=-1)
    if np.any(sn <= DEGENERATE):
        raise ValueError('Undefined self direction; no cases omitted')
    sh = s / sn[..., None]
    eps = w - s
    a2 = np.sum(eps * eps, axis=-1)
    alpha = np.sum(sh * eps, axis=-1)
    beta2 = a2 - alpha * alpha
    if np.any(beta2 < -1e-10 * np.maximum(a2, 1)):
        raise ValueError('Negative squared perpendicular magnitude')
    beta = np.sqrt(np.maximum(beta2, 0))
    perp = v - np.sum(v * sh, axis=-1)[..., None] * sh
    pn = np.linalg.norm(perp, axis=-1)
    active = beta > DEGENERATE
    if np.any(active & (pn <= DEGENERATE)):
        raise ValueError('Undefined perpendicular direction; no redraw or omission')
    unit = np.divide(perp, pn[..., None], out=np.zeros_like(perp),
                     where=(pn > DEGENERATE)[..., None])
    eta = alpha[..., None] * sh + beta[..., None] * unit
    injection = (s + eta).astype(np.float32)
    return injection, dict(alpha=alpha, beta=beta,
                           perturbation_target=np.sqrt(a2), direction_norm=pn)


def check_geometry(s, w, injection):
    s, w, x = [np.asarray(z, dtype=np.float64) for z in (s, w, injection)]
    targets = dict(norm=np.linalg.norm(w, axis=-1),
                   cross=np.sum(s*w, axis=-1),
                   perturbation=np.linalg.norm(w-s, axis=-1))
    realised = dict(norm=np.linalg.norm(x, axis=-1),
                    cross=np.sum(s*x, axis=-1),
                    perturbation=np.linalg.norm(x-s, axis=-1))
    result = dict(n=int(np.prod(s.shape[:-1])), finite=bool(np.isfinite(x).all()))
    for key in targets:
        error = np.abs(realised[key]-targets[key])
        allowed = GEOM_ATOL + GEOM_RTOL*np.abs(targets[key])
        result[key] = dict(max_absolute_error=float(error.max()),
                           max_tolerance_fraction=float(np.max(error/allowed)),
                           violations=int(np.sum(error > allowed)))
    result['passed'] = result['finite'] and all(result[k]['violations'] == 0 for k in targets)
    return result


def synthetic_test():
    rng = np.random.default_rng(100)
    s = rng.normal(size=(5, 3, 16)).astype(np.float32)
    w = rng.normal(size=s.shape).astype(np.float32)
    own, _ = geometry_match(s, w, w.astype(float)-s)
    np.testing.assert_array_equal(own, w)
    zero, _ = geometry_match(s, s, np.zeros_like(s))
    np.testing.assert_array_equal(zero, s)
    x, _ = geometry_match(s, w, rng.normal(size=s.shape))
    assert check_geometry(s, w, x)['passed']
    # A reversed perpendicular component must change direction, not norm or dot.
    opposite, _ = geometry_match(s, w, s.astype(float)-w)
    assert check_geometry(s, w, opposite)['passed']
    assert np.linalg.norm(opposite-w) > 1
    try:
        geometry_match(s, w, np.zeros_like(s))
    except ValueError:
        pass
    else:
        raise AssertionError('A degenerate direction was silently accepted')
    # Known signature geometry: self at recipient, wording 1/4 of axis,
    # control 1/2 of axis, so both ratios are exactly 1/2.
    r = np.array([[0., 0.]])
    d = np.array([[4., 0.]])
    assert paired(np.array([[1., 0.]]), r, d, np.ones_like(r))['score'][0] == .25
    assert paired(np.array([[2., 0.]]), r, d, np.ones_like(r))['score'][0] == .5
    return dict(passed=True, checks=['zero_equals_self', 'own_direction_equals_wording',
                'norm_cross_and_perturbation_matching', 'direction_changes',
                'undefined_direction_rejected', 'known_F_and_A_ratio'])
