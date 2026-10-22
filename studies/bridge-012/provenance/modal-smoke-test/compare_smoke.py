"""Compare the Modal reproduction with the saved Study 6 held-out arrays (project environment, no model).

    .venv/bin/python studies/bridge-012/provenance/modal-smoke-test/compare_smoke.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TOL = 5e-4
REC = HERE / 'records'
remote = json.loads((REC / 'remote-record.json').read_text())
test = [r for r in json.loads((ROOT / 'studies/bridge-011/data/rows.json').read_text()) if r['split'] == 'test']
ids = {r['id']: k for k, r in enumerate(test)}
ii = [ids[s] for s in remote['rows']]
checks = {}
with np.load(REC / 'modal-signatures.npz') as got:
    names = sorted({k.split('__')[0] for k in got.files})
    for name in names:
        with np.load(ROOT / f'studies/bridge-011/results/test/{name}.npz') as saved:
            checks[name] = dict(raw=float(np.abs(got[f'{name}__raw'] - saved['raw'][:, ii]).max()),
                                behavior=float(np.abs(got[f'{name}__behavior'] - saved['behavior'][ii]).max()))
worst = max(max(v.values()) for v in checks.values())
target = REC / 'comparison.json'
assert not target.exists(), target
target.write_text(json.dumps(dict(tolerance=TOL, max_abs_error=worst, passed=worst <= TOL, per_condition=checks,
                                  rows=remote['rows'], conditions=names,
                                  compared_with='studies/bridge-011/results/test/<condition>.npz at the eight singleton rows'),
                             indent=2) + '\n')
print(json.dumps(dict(passed=worst <= TOL, max_abs_error=worst, conditions=len(names))))
