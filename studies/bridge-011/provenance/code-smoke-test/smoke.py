"""Smoke test of Study 6 code on Study 2 calibration rows; scratch outputs only."""
import json, sys, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]   # release: the release root
S = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'studies/bridge-011/code')]
import run011, common011
rows = [dict(r, wording=[10, 11].index(r['family']), split='x') for r in json.loads((ROOT / 'studies/bridge-007/data/candidates.json').read_text())
        if r['split'] == 'calibration' and r['family'] in (10, 11)]
freeze = S / 'freeze.json'
freeze.write_text(json.dumps(dict(files={'configs/study.json': hashlib.sha256((ROOT / 'configs/study.json').read_bytes()).hexdigest()})))
run011.OUT = S / 'results'; run011.FREEZE = freeze; run011.rows_for = lambda split: rows
t0 = time.time()
e = run011.Engine011('cuda')
run011.calibration(e)
print('calibration', round(time.time() - t0, 1), 's'); t0 = time.time()
dev = run011.development(e)
print('development', round(time.time() - t0, 1), 's', dev['held'], dev['kappa'], [ (t['k'], t['conditions']) for t in dev['trials']])
print(json.dumps(json.loads((S / 'results/calibration.json').read_text())['checks'], indent=1))
