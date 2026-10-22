"""Shared helpers for the post hoc analyses in this folder.

Everything here reads saved arrays only; no model inference and no writes to studies/.
Requires NumPy only.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / 'results'
DRAWS = 50_000
SEED = 2026100107


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def study_rows(study, model=None):
    """Final-test rows and paired donor indices for Bridge 003, 004 or 005."""
    base = ROOT / 'studies' / study
    if study == 'bridge-005':
        sel = read(base / 'data/selection.json')
        candidates = read(base / 'data/candidates.json')
        test = base / 'results/test'
    elif study == 'bridge-003':
        sel = read(base / 'data/broad-selection.json')['selected'][model]
        candidates = read(base / f'data/{model}-candidates.json')
        test = base / f'results/{model}/test'
    else:
        sel = read(base / f'data/{model}-selection.json')
        candidates = read(base / f'data/{model}-candidates.json')
        test = base / f'results/{model}/test'
    fams = sel['families']
    rows = [dict(r, wording=fams.index(r['family'])) for r in candidates
            if r['split'] == 'test' and r['family'] in fams]
    key = {(r['template'], r['pair'], r['role'], r['wording'], r['order']): i for i, r in enumerate(rows)}
    donor = np.array([key[(r['template'], r['pair'], r['role'], 1 - r['wording'], r['order'])] for r in rows])
    scale = np.asarray(sel['reference']['scale'], dtype=np.float64)[:, None, :]
    return dict(rows=rows, donor=donor, scale=scale, families=fams, test=test, selection=sel)


class Crossed:
    """Crossed (pigeonhole) bootstrap over context blocks x name triples, as in the original analyses.

    Each block x triple cell is averaged first; blocks and triples are then resampled
    independently with replacement. Intervals are pointwise percentile intervals.
    """

    def __init__(self, rows, draws=DRAWS, seed=SEED):
        self.T = sorted({r['template'] for r in rows})
        self.P = sorted({r['pair'] for r in rows})
        self.t = np.array([self.T.index(r['template']) for r in rows])
        self.p = np.array([self.P.index(r['pair']) for r in rows])
        rng = np.random.default_rng(seed)
        self.tt = rng.integers(len(self.T), size=(draws, len(self.T)))
        self.pp = rng.integers(len(self.P), size=(draws, len(self.P)))
        self.n = len(rows)

    def __call__(self, values, mask=None):
        x = np.asarray(values, dtype=np.float64)
        mask = np.ones(self.n, bool) if mask is None else mask
        grid = np.full((len(self.T), len(self.P)), np.nan)
        for a in range(len(self.T)):
            for b in range(len(self.P)):
                grid[a, b] = x[mask & (self.t == a) & (self.p == b)].mean()
        assert np.isfinite(grid).all()
        boot = grid[self.tt[:, :, None], self.pp[:, None, :]].mean((1, 2))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        return dict(mean=float(grid.mean()), ci95=[float(lo), float(hi)])


def geometry(edited, recipient, donor, scale):
    """Paired endpoint geometry in the fixed scaled coordinates (Section 3.3 of the paper)."""
    u = (edited - recipient) / scale
    v = (donor - recipient) / scale
    n = np.linalg.norm(v, axis=-1)
    S = (u * v).sum(-1) / n**2
    qR = np.linalg.norm(u, axis=-1) / n
    qD = np.linalg.norm(u - v, axis=-1) / n
    O = np.linalg.norm(u - S[..., None] * v, axis=-1) / n
    return dict(S=S, qR=qR, qD=qD, M=qR - qD, O=O, norm=n)

