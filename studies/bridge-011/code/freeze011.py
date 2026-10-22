"""Study 6 local freezes (not public preregistration).

    freeze011.py input        # code, banks, docs, bootstrap indices, inherited inputs; before any inference
    freeze011.py development  # development record and measurements; before held-out inference
    freeze011.py verify       # check every existing Study 6 freeze
"""
from datetime import datetime, timezone
from pathlib import Path
import sys as _release_sys  # release: frozen-file checks accept documented changes
_release_sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
import release_paths as _release_paths  # noqa: E402
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = ROOT / 'studies/bridge-007'
sys.path[:0] = [str(HERE / 'code'), str(BASE / 'code')]
FREEZES = ['protocol-freeze.json', 'input-freeze.json', 'development-freeze.json']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_once(target, kind, files):
    files = sorted({Path(p).resolve() for p in files})
    assert all(p.is_file() for p in files)
    value = dict(utc=datetime.now(timezone.utc).isoformat(), kind=kind,
                 files={str(p.relative_to(ROOT)): sha(p) for p in files})
    with target.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(freeze=target.name, sha256=sha(target), files=len(files), utc=value['utc'])))


def verify():
    checked = 0
    for name in FREEZES:
        target = HERE / 'data' / name
        if target.exists():
            for rel, expected in read(target)['files'].items():
                assert _release_paths.matches(ROOT / rel, expected, rel), (name, rel)
                checked += 1
    print(f'Study 6 freezes verified ({checked} file hashes)')


def input_freeze():
    target = HERE / 'data/input-freeze.json'
    if target.exists():
        return verify()
    assert not (HERE / 'results').exists(), 'Results predate the input freeze'
    for rel, expected in read(HERE / 'data/protocol-freeze.json')['files'].items():
        assert sha(ROOT / rel) == expected, rel
    assert (HERE / 'docs/AUTHOR-APPROVAL.txt').exists()
    import analyze011
    from analyze007 import Crossed
    checks = HERE / 'data/pre-freeze-analysis-checks.json'
    if not checks.exists():
        result = analyze011.self_test()
        checks.write_text(json.dumps(dict(result, utc=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')
    assert read(checks)['passed']
    rows = read(HERE / 'data/rows.json')
    for split in ('development', 'test'):
        path = HERE / f'data/bootstrap-{split}.npz'
        if not path.exists():
            boot = Crossed([r for r in rows if r['split'] == split], draws=analyze011.DRAWS, seed=analyze011.BOOT_SEED)
            with path.open('xb') as stream:
                np.savez_compressed(stream, blocks=boot.tt, triples=boot.pp, seed=np.array(analyze011.BOOT_SEED))
    files = set((HERE / 'code').glob('*.py')) | set((HERE / 'docs').glob('*'))
    files |= {p for p in (HERE / 'data').glob('*') if p.name != 'input-freeze.json'}
    files |= set((HERE / 'provenance').rglob('*'))
    files |= {ROOT / rel for rel in read(HERE / 'data/protocol-freeze.json')['files']}
    files |= set((BASE / 'data').glob('*freeze.json')) | {BASE / 'results/verification.json'}
    files |= {ROOT / 'pyproject.toml', ROOT / 'uv.lock', ROOT / 'configs/study.json',
              ROOT / 'studies/transport/results/experiment-004/localization/selected-heads.json'}
    for folder in ['src', 'studies/bridge-002/code', 'studies/bridge-003/code', 'studies/bridge-004/code',
                   'studies/bridge-005/code', 'studies/bridge-006/code', 'studies/bridge-007/code']:
        files |= {p for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts}
    files |= {p for p in (ROOT / 'models/gpt2-medium').rglob('*') if p.is_file()}
    files = {p for p in files if p.is_file() and '__pycache__' not in p.parts}
    write_once(target, 'Study 6 input freeze after author approval, before calibration and any Study 6 inference', files)


def development_freeze():
    target = HERE / 'data/development-freeze.json'
    if target.exists():
        return verify()
    verify()
    record = read(HERE / 'results/development.json')
    assert read(HERE / 'results/calibration.json')['passed']
    assert not (HERE / 'results/test').exists(), 'Held-out results predate the development freeze'
    files = {HERE / 'data/input-freeze.json', HERE / 'results/development.json', HERE / 'results/calibration.json'}
    files |= {p for p in (HERE / 'results/development').glob('*') if p.is_file()}
    files |= {p for p in (HERE / 'results/calibration').glob('*') if p.is_file()}
    write_once(target, f"Study 6 development freeze: held={record['held']}, kappa={record['kappa']}; "
                       'before held-out inference', files)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'verify'
    dict(input=input_freeze, development=development_freeze, verify=verify)[stage]()
