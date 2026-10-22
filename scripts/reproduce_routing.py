"""Recover the original routing study with its omitted pilot calibration input.

The frozen source archive is untouched. --smoke runs the recovered engine's full
numerical validation; --full additionally repeats all original model inference.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'studies/routing'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    g=p.add_mutually_exclusive_group();g.add_argument('--smoke',action='store_true');g.add_argument('--full',action='store_true')
    a=p.parse_args();dest=a.output.resolve();dest.mkdir(parents=True,exist_ok=False)
    work=dest/'work/routing-project';work.mkdir(parents=True)
    archive=SOURCE/'data/study-002-frozen-source.tar.gz'
    with tarfile.open(archive) as t:t.extractall(work,filter='data')
    freeze=read(work/'data/study-002-freeze.json')
    for name,h in freeze['files'].items():assert sha(work/name)==h,name
    pilot=SOURCE/'data/pilot-001.jsonl';expected=read(SOURCE/'data/pilot-001-freeze.json')['files']['data/pilot-001.jsonl']
    assert sha(pilot)==expected;shutil.copy2(pilot,work/'data/pilot-001.jsonl')
    analyzer=SOURCE/'scripts/analyze_study2.py';assert sha(analyzer)==read(SOURCE/'data/study-002-analysis-freeze.json')['analyzer_sha256']
    shutil.copy2(analyzer,work/'scripts/analyze_study2.py')
    model=ROOT/'models/gpt2-small'
    for name,h in freeze['model']['files'].items():assert sha(model/name)==h,name
    link=work/freeze['model']['path'];link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(model,target_is_directory=True)
    env=dict(os.environ,PYTHONPATH=str(work/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    record=dict(utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha(archive),frozen_files=len(freeze['files']),
                recovered_pilot_sha256=expected,scope='Recovered-source validation; optional full original inference',smoke_passed=False,full_inference=False)
    if a.smoke or a.full:
        code="import json,torch; from pathlib import Path; from causal_circuit_routing.study2_engine import Engine; c=json.loads(Path('configs/study-002.json').read_text()); e=Engine(c);\nwith torch.inference_mode():\n r=e.validate([])\nPath('validation-recovered.json').write_text(json.dumps(r,indent=2)+'\\n');print(r)"
        subprocess.run([sys.executable,'-c',code],cwd=work,env=env,check=True)
        record['smoke_passed']=True;record['validation']=read(work/'validation-recovered.json')
    if a.full:
        subprocess.run([sys.executable,'scripts/run_study2_family.py'],cwd=work,env=env,check=True)
        (work/'data/study-002-analysis-freeze.json').write_text(json.dumps(dict(analyzer_sha256=sha(work/'scripts/analyze_study2.py'),before_test_inference=True,
          frozen_utc=datetime.now(timezone.utc).isoformat(),selection_freeze_sha256=sha(work/'data/study-002-selection-freeze.json')),indent=2)+'\n')
        for script in ['run_study2_transfer.py','analyze_study2.py']:
            subprocess.run([sys.executable,'scripts/'+script],cwd=work,env=env,check=True)
        record['full_inference']=True
    (dest/'recovery-verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
