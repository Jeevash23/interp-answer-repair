"""Immutable local Study 5 protocol/input/draw/code freeze, before any new inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys as _release_sys  # release: frozen-file checks accept documented changes
_release_sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
import release_paths as _release_paths  # noqa: E402

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1];BASE=ROOT/'studies/bridge-007'
TARGET=HERE/'data/input-freeze.json'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    if TARGET.exists():
        for name,expected in read(TARGET)['files'].items():assert _release_paths.matches(ROOT/name,expected,name),name
        print('Existing Study 5 freeze verified');return
    assert not (HERE/'results').exists(),'Scientific results predate freeze'
    checked={}
    for freeze in (BASE/'data').glob('*freeze.json'):
        for name,expected in read(freeze)['files'].items():
            if name not in checked:checked[name]=sha(ROOT/name)
            assert checked[name]==expected,(freeze,name)
    audit=read(BASE/'results/verification.json');assert audit['status']=='passed'
    assert read(BASE/'results/status.json')['outcome']=='final_test_completed'
    assert read(BASE/'results/computational-replay.json')['passed']
    assert sha(BASE/'results/analysis/summary.json')==audit['analysis_summary_sha256']
    summary=read(BASE/'results/analysis/summary.json')
    assert summary['qualification']['qualified'] and summary['test_qualification']['qualified']
    assert read(HERE/'data/pre-freeze-synthetic-checks.json')['passed']
    assert read(HERE/'data/pre-freeze-analysis-checks.json')['passed']
    rows=read(HERE/'data/rows.json');ids=[r['id'] for r in rows]
    files={ROOT/p for p in checked}
    files |= set((HERE/'code').glob('*.py')) | set((HERE/'docs').glob('*'))
    files |= set((HERE/'data').glob('*.json')) | set((HERE/'data/draws').glob('*.npz'))
    files |= set((BASE/'data').glob('*freeze.json'))
    files |= {BASE/'data/selection.json',BASE/'results/fit/bank.npz',
              BASE/'results/analysis/summary.json',BASE/'results/verification.json',
              BASE/'results/status.json',BASE/'results/inference-complete.json',
              BASE/'results/computational-replay.json',BASE/'results/selected-pair-replay.json'}
    audited={r['path']:r['aggregate_sha256'] for r in audit['checks']['runner_records']}
    for name in ('clean','corrupt','self','wording','natural-clean','natural-corrupt'):
        p=BASE/f'results/test/{name}.npz';rp=p.with_name(p.stem+'-record.json');record=read(rp)
        assert record['status']=='completed' and record['ids']==ids
        assert sha(p)==record['aggregate_sha256']==audited[str(rp.relative_to(BASE))]
        files|={p,rp}
    for entry in read(HERE/'data/design.json')['runtime']['basis']:files.add(ROOT/entry['path'])
    value=dict(utc=datetime.now(timezone.utc).isoformat(),
               kind='Author-approved local freeze before Study 5 inference; exploratory on inspected Study 2 bank',
               files={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    TARGET.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(sha256=sha(TARGET),files=len(files),utc=value['utc'])))


if __name__=='__main__':main()
