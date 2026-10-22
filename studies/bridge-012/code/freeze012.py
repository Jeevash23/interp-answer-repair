"""Study 7 local freezes (not public preregistration).

    freeze012.py input   # bootstrap indices, then code, banks, docs and inherited inputs; before any GPT-2 Large inference
    freeze012.py verify  # protocol and input freezes, and any pulled-back stage freeze for files present locally
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE / 'code'), str(ROOT / 'studies/bridge-007/code')]
INHERITED_DIRS = ['src', 'studies/bridge-002/code', 'studies/bridge-003/code', 'studies/bridge-004/code',
                  'studies/bridge-005/code', 'studies/bridge-006/code', 'studies/bridge-007/code', 'studies/bridge-011/code']
INHERITED_FILES = ['configs/study.json', 'studies/transport/results/experiment-004/localization/selected-heads.json',
                   'pyproject.toml', 'uv.lock']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify():
    checked, remote = 0, 0
    paths = [HERE / 'data/protocol-freeze.json', HERE / 'data/input-freeze.json'] + sorted((HERE / 'results/freezes').glob('*.json'))
    for path in paths:
        if not path.exists():
            continue
        for rel, expected in read(path)['files'].items():
            if (ROOT / rel).exists():
                assert sha(ROOT / rel) == expected, (path.name, rel)
                checked += 1
            else:
                remote += 1
    print(f'Study 7 freezes verified locally ({checked} file hashes; {remote} listed files are kept on the Modal Volume)')


def input_freeze():
    target = HERE / 'data/input-freeze.json'
    if target.exists():
        return verify()
    assert not (HERE / 'results').exists(), 'Results predate the input freeze'
    for rel, expected in read(HERE / 'data/protocol-freeze.json')['files'].items():
        assert sha(ROOT / rel) == expected, rel
    assert (HERE / 'docs/AUTHOR-APPROVAL.txt').exists()
    import analyze007 as a7
    from common012 import rows_for
    seed_b = 2026101704
    for name, split, seed in (('bootstrap-test-a', 'test', None), ('bootstrap-test-b', 'test', seed_b),
                              ('bootstrap-development-b', 'development', seed_b)):
        path = HERE / f'data/{name}.npz'
        if not path.exists():
            rows = rows_for(split, [10, 11])  # the grid depends only on the block and triple counts
            boot = a7.Crossed(rows) if seed is None else a7.Crossed(rows, seed=seed)
            with path.open('xb') as stream:
                np.savez_compressed(stream, blocks=boot.tt, triples=boot.pp, seed=np.array(boot.seed), draws=np.array(boot.draws))
    files = set((HERE / 'code').glob('*.py')) | set((HERE / 'docs').glob('*'))
    files |= {p for p in (HERE / 'data').glob('*') if p.name != 'input-freeze.json'}
    for d in INHERITED_DIRS:
        files |= {p for p in (ROOT / d).rglob('*.py') if '__pycache__' not in p.parts}
    files |= {ROOT / f for f in INHERITED_FILES}
    value = dict(utc=datetime.now(timezone.utc).isoformat(),
                 kind='Study 7 input freeze after the author approval and the GPT-2 Medium code test, before any GPT-2 Large inference on task prompts',
                 seeds=dict(names=2026101701, contexts=2026101702, random_controls=2026101703, family_b_bootstrap=seed_b,
                            family_a_bootstrap='2026100407 (Study 2 analysis code, unchanged)'),
                 hardware=dict(platform='Modal', gpu='L4', fallback='T4 with calibration repeated',
                               image='debian_slim(python 3.12) + uv_sync(extras=[experiment], frozen) from uv.lock; weights fetched at pinned revisions in the image build'),
                 weights=dict(repo='openai-community/gpt2-large', revision='32b71b12589c2f8d625668d2335a01cac3249519',
                              sha256='5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce', bytes=3247159078),
                 files={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})
    with target.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(freeze=target.name, sha256=sha(target), files=len(value['files']), utc=value['utc'])))


if __name__ == '__main__':
    dict(input=input_freeze, verify=verify)[sys.argv[1] if len(sys.argv) > 1 else 'verify']()
