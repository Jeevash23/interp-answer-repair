"""Bind the exploratory coverage protocol and reused benchmark before inference."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = ROOT / 'studies/bridge-007'
TARGET = HERE / 'data/input-freeze.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    if TARGET.exists():
        for name, expected in read(TARGET)['files'].items():
            assert sha(ROOT / name) == expected, name
        print('Existing Bridge008 freeze verified'); return
    assert not any((HERE / 'results' / name).exists() for name in
                   ('natural', 'deltas', 'test', 'calibration.json')), 'New outcomes predate freeze'
    base_audit = read(BASE / 'results/verification.json')
    assert base_audit['status'] == 'passed'
    # Preserve the original identity instead of legitimising changed upstream
    # bytes by merely computing a new hash for the exploratory study.
    checked = {}
    for manifest in sorted((BASE / 'data').glob('*freeze.json')):
        for name, expected in read(manifest)['files'].items():
            actual = checked.setdefault(name, None)
            if actual is None:
                actual = checked[name] = sha(ROOT / name)
            assert actual == expected, (manifest.name, name)
    audited_outputs = {row['path']: row['aggregate_sha256']
                       for row in base_audit['checks']['runner_records']}
    assert read(BASE / 'results/status.json')['outcome'] == 'final_test_completed'
    assert read(BASE / 'results/computational-replay.json')['passed']
    assert sha(BASE / 'results/analysis/summary.json') == base_audit['analysis_summary_sha256']
    summary = read(BASE / 'results/analysis/summary.json')
    selected = read(BASE / 'data/selection.json')
    assert summary['families'] == selected['families'] == [10, 11]
    eligible = bool(summary['qualification']['qualified'] and
                    summary['test_qualification']['qualified'])
    assert eligible == base_audit['checks']['final_decisions']['qualified']
    rows = [dict(r, wording=selected['families'].index(r['family']))
            for r in read(BASE / 'data/candidates.json')
            if r['split'] == 'test' and r['family'] in selected['families']]
    assert len(rows) == 3200
    design = dict(utc=datetime.now(timezone.utc).isoformat(),
                  scope='Exploratory paired extension on the inspected Bridge007 benchmark',
                  families=selected['families'], eligible=eligible,
                  eligibility_basis='Independent and final clean gates only; primary effects do not gate this study',
                  n=len(rows), blocks=20, triples=20, row_ids=[r['id'] for r in rows],
                  source_layers=list(range(6, 13)), narrow_heads=12, broad_heads=112,
                  seed=2026100408, bootstrap_draws=50000, primary_multiplicity=4,
                  primary_outcomes_previously_inspected=True)
    write(HERE / 'data/design.json', design)
    files = {ROOT / p for p in read(BASE / 'data/input-freeze.json')['files']}
    files |= {HERE / 'code' / name for name in
              ('freeze008.py', 'operators008.py', 'run008.py', 'analyze008.py')}
    files |= {HERE / 'docs/PROTOCOL.md', HERE / 'docs/DESIGN-REVIEW.md', HERE / 'data/design.json'}
    files |= set((BASE / 'data').glob('*freeze.json'))
    files |= {BASE / 'data/selection.json', BASE / 'results/fit/bank.npz',
              BASE / 'results/analysis/summary.json', BASE / 'results/verification.json',
              BASE / 'results/status.json', BASE / 'results/inference-complete.json',
              BASE / 'results/computational-replay.json', BASE / 'results/qualification-summary.json',
              BASE / 'results/selected-pair-replay.json'}
    for name in ('clean', 'corrupt', 'wording', 'self', 'natural-clean', 'natural-corrupt'):
        aggregate = BASE / f'results/test/{name}.npz'
        record_path = BASE / f'results/test/{name}-record.json'
        record = read(record_path)
        assert record['status'] == 'completed' and record['ids'] == design['row_ids']
        assert sha(aggregate) == record['aggregate_sha256'] == audited_outputs[str(record_path.relative_to(BASE))]
        files |= {aggregate, record_path}
    assert all(p.is_file() for p in files), sorted(str(p) for p in files if not p.is_file())
    write(TARGET, dict(utc=datetime.now(timezone.utc).isoformat(),
                       kind='Local prospective operator freeze on reused outcome-informed benchmark; not public preregistration',
                       files={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}))
    if not eligible:
        write(HERE / 'results/status.json', dict(status='completed', outcome='ineligible_base',
              test_measured=False, reason='Bridge007 final clean qualification failed', freeze_sha256=sha(TARGET)))
    print(json.dumps(dict(freeze=str(TARGET), sha256=sha(TARGET), files=len(files), eligible=eligible)))


if __name__ == '__main__':
    main()
