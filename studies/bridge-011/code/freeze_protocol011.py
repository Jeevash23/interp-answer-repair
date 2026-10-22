"""Study 6 protocol freeze: hash the protocol, predictions, runtime record and inherited inputs.

Run once before the author's approval; later runs only verify. The inherited Study 2
files are first checked against Study 2's own freezes and canonical audit. This freeze
precedes bank construction and all Study 6 inference; the full input freeze comes later.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = ROOT / 'studies/bridge-007'
TARGET = HERE / 'data/protocol-freeze.json'
OWN = ['docs/PROTOCOL.md', 'docs/PREDICTIONS.md', 'code/freeze_protocol011.py',
       'provenance/runtime-benchmark/benchmark011.py', 'provenance/runtime-benchmark/runtime-benchmark.json']
INHERITED = ['configs/study.json', 'studies/bridge-007/data/selection.json', 'studies/bridge-007/results/fit/bank.npz',
             'studies/bridge-007/results/analysis/summary.json', 'studies/bridge-007/results/verification.json',
             'studies/bridge-007/data/candidates.json', 'studies/bridge-007/code/runner007.py',
             'studies/bridge-007/code/analyze007.py', 'studies/bridge-007/code/prepare007.py',
             'studies/bridge-006/code/cpu_engine.py', 'studies/bridge-005/code/common005.py',
             'src/transport_bridge/engine.py', 'src/transport_bridge/statistics.py']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if TARGET.exists():
        for name, expected in read(TARGET)['files'].items():
            assert sha(ROOT / name) == expected, name
        print('Existing Study 6 protocol freeze verified')
        return
    assert not (HERE / 'results').exists(), 'Study 6 results predate the protocol freeze'
    # Every inherited file that Study 2 froze must still match each Study 2 freeze.
    checked = {}
    for freeze in sorted((BASE / 'data').glob('*freeze.json')):
        for name, expected in read(freeze)['files'].items():
            if name in INHERITED:
                checked.setdefault(name, sha(ROOT / name))
                assert checked[name] == expected, (freeze.name, name)
    assert {'studies/bridge-007/data/selection.json', 'studies/bridge-007/results/fit/bank.npz'} <= set(checked)
    audit = read(BASE / 'results/verification.json')
    assert audit['status'] == 'passed'
    assert sha(BASE / 'results/analysis/summary.json') == audit['analysis_summary_sha256']
    files = {name: sha(ROOT / name) for name in sorted(INHERITED)}
    files |= {str((HERE / name).relative_to(ROOT)): sha(HERE / name) for name in OWN}
    value = dict(utc=datetime.now(timezone.utc).isoformat(),
                 kind='Local protocol freeze before the author\'s approval, bank construction and any Study 6 inference; '
                      'not public preregistration',
                 checked_against_study2_freezes=sorted(checked), files=dict(sorted(files.items())))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with TARGET.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(sha256=sha(TARGET), files=len(files), utc=value['utc'],
                          checked_against_study2_freezes=sorted(checked))))


if __name__ == '__main__':
    main()
