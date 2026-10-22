"""Inspect and replay the complete activation-transfer project without inference."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

REPLAYS={
    'transport': ('studies/transport/scripts/reproduce_saved_statistics.py',[], '--output-dir',None),
    'routing': ('studies/routing/scripts/analyze_study2.py',[], '--output-dir','studies/routing/results/study-002/analysis'),
    'initial-medium': ('scripts/analyze.py',['--model','gpt2-medium'],'--output','results/gpt2-medium/analysis'),
    'initial-small': ('scripts/analyze.py',['--model','gpt2-small'],'--output','results/gpt2-small/analysis'),
    'grammatical-medium': ('studies/bridge-002/code/analyze.py',['--model','gpt2-medium'],'--output','studies/bridge-002/results/gpt2-medium/analysis'),
    'grammatical-small': ('studies/bridge-002/code/analyze.py',['--model','gpt2-small'],'--output','studies/bridge-002/results/gpt2-small/analysis'),
    'opposing-groups': ('studies/bridge-003/code/analyze_medium003.py',[],'--output','studies/bridge-003/results/medium-cancellation/analysis'),
    'broader-medium': ('studies/bridge-003/code/analyze_broad003.py',['--model','gpt2-medium'],'--output','studies/bridge-003/results/gpt2-medium/analysis'),
    'broader-pythia': ('studies/bridge-003/code/analyze_broad003.py',['--model','pythia-410m-deduped'],'--output','studies/bridge-003/results/pythia-410m-deduped/analysis'),
    'paired-medium': ('studies/bridge-004/code/analyze004_corrected.py',['--model','gpt2-medium'],'--output','studies/bridge-004/results/gpt2-medium/analysis'),
    'paired-tinyllama': ('studies/bridge-004/code/analyze004_corrected.py',['--model','tinyllama-1.1b-base'],'--output','studies/bridge-004/results/tinyllama-1.1b-base/analysis'),
    'qualified-replication': ('studies/bridge-005/code/analyze005.py',[],'--output','studies/bridge-005/results/analysis'),
}


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify():
    manifest=json.loads((ROOT/'provenance/evidence-manifest.json').read_text())
    for item in manifest['files']:
        path=ROOT/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:
            raise ValueError('Retained evidence differs: '+item['path'])
    changes=json.loads((ROOT/'provenance/source-relocations.json').read_text())['files']
    for path,record in changes.items():
        if sha(ROOT/path)!=record['relocated_sha256']:
            raise ValueError('Relocated source differs: '+path)
    for name in ['gpt2-small','gpt2-medium','gpt2-large','pythia-410m-deduped','tinyllama-1.1b-base']:
        if not (ROOT/'models'/name/'config.json').is_file():raise ValueError('Missing model: '+name)
    import circuit_transport,causal_circuit_routing,transport_bridge
    for module in [circuit_transport,causal_circuit_routing,transport_bridge]:
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise ValueError('An import escaped the consolidated project')
    result={'status':'passed','unchanged_evidence_files':len(manifest['files']),
            'documented_source_relocations':len(changes),'all_packages_inside_project':True,
            'scope':'Saved evidence hashes, documented source relocations, local imports and model configuration locations; no fresh model inference.'}
    print(json.dumps(result,indent=2),flush=True)
    return result


def replay(output,only=None):
    output=output.resolve()
    if output.exists():raise FileExistsError('Choose a new replay output directory')
    output.mkdir(parents=True)
    verification=verify();outcomes={}
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    for name,(script,args,flag,reference) in REPLAYS.items():
        if only and name not in only:continue
        target=output/name;log=output/(name+'.log')
        with log.open('w') as stream:
            run=subprocess.run([sys.executable,str(ROOT/script),*args,flag,str(target)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result={'exit_code':run.returncode,'log':log.name}
        if run.returncode==0 and reference:
            left=json.loads((ROOT/reference/'summary.json').read_text())
            right=json.loads((target/'summary.json').read_text())
            result['summary_exact']=left==right
            # Saved array equality checks numerical content, independent of ZIP timestamps.
            candidates=sorted((ROOT/reference).glob('*.npz'))
            if candidates:
                import numpy as np
                result['arrays_exact']=True
                result['array_files_checked']=[]
                for candidate in candidates:
                    with np.load(candidate) as a,np.load(target/candidate.name) as b:
                        result['arrays_exact'] &= set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files)
                    result['array_files_checked'].append(candidate.name)
        result['passed']=run.returncode==0 and result.get('summary_exact',True) and result.get('arrays_exact',True)
        outcomes[name]=result
        print(name, 'PASS' if result['passed'] else 'FAIL',flush=True)
        (output/'replay-verification.json').write_text(json.dumps({'integrity':verification,'analyses':outcomes,'all_passed':all(x['passed'] for x in outcomes.values())},indent=2)+'\n')
    if not all(x['passed'] for x in outcomes.values()):raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify');sub.add_parser('list')
    p=sub.add_parser('replay');p.add_argument('--output',type=Path,required=True);p.add_argument('--only',nargs='+',choices=list(REPLAYS))
    args=parser.parse_args()
    if args.command=='verify':verify()
    elif args.command=='list':print('\n'.join(REPLAYS))
    else:replay(args.output,args.only)


if __name__=='__main__':main()
