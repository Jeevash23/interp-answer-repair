"""Smoke test of held-out, replay, analysis and audit code on Study 2 calibration rows; scratch only."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]   # release: the release root
S = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'studies/bridge-011/code')]
import run011, analyze011, audit011
from analyze007 import Crossed
rows = [dict(r, wording=[10, 11].index(r['family']), split='test') for r in json.loads((ROOT / 'studies/bridge-007/data/candidates.json').read_text())
        if r['split'] == 'calibration' and r['family'] in (10, 11)]
freeze = S / 'freeze.json'
run011.OUT = S / 'results'; run011.FREEZE = freeze; run011.DEV_FREEZE = freeze; run011.rows_for = lambda split: rows
run011.replay_rows = lambda: (rows, [0, 5])
t0 = time.time()
e = run011.Engine011('cuda')
run011.test(e); print('test', round(time.time() - t0, 1)); t0 = time.time()
del e
run011.replay(run011.Engine011('cuda'), 'gpu'); print('gpu replay', round(time.time() - t0, 1)); t0 = time.time()
run011.replay(run011.Engine011('cpu'), 'cpu'); print('cpu replay', round(time.time() - t0, 1))
study = S / 'study'; (study / 'data').mkdir(parents=True, exist_ok=True)
for name in ('protocol-freeze.json', 'input-freeze.json', 'development-freeze.json'):
    (study / 'data' / name).write_text(json.dumps(dict(files={})))
(study / 'data/rows.json').write_text(json.dumps(rows))
boot = Crossed(rows, draws=50000, seed=analyze011.BOOT_SEED)
np.savez_compressed(study / 'data/bootstrap-test.npz', blocks=boot.tt, triples=boot.pp)
if not (study / 'results').exists(): (study / 'results').symlink_to(S / 'results')
dev = json.loads((S / 'results/development.json').read_text())
analyze011.HERE = study
r, sig, attr, held, inputs = analyze011.load_split('test', dev)
result, per = analyze011.analyse(r, sig, attr, held, json.loads((ROOT / 'studies/bridge-007/data/selection.json').read_text())['reference'],
                                 boot_indices=np.load(study / 'data/bootstrap-test.npz'))
(S / 'results/analysis').mkdir(exist_ok=True)
(S / 'results/analysis/summary.json').write_text(json.dumps(result, indent=1))
print('decisions', result['decisions'], 'gates', result['clean_gates']['qualified'])
audit011.HERE = study; audit011.RES = study / 'results'
if (S / 'results/verification.json').exists(): (S / 'results/verification.json').unlink()
audit011.main()
