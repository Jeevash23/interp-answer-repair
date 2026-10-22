"""Local ordering records, never a claim of public preregistration."""
import argparse
from common004 import *

def inputs():
    assert not (HERE/'data/input-freeze.json').exists()
    assert read(HERE/'results/design-tests.json')['passed']
    for m in MODELS:
        assert read(HERE/'results'/m/'technical.json').get('passed',True)
        assert read(HERE/'results'/m/'matched-boundary-replay.json')['passed']
    checkpoints={}
    for folder in [ROOT / 'models/gpt2-medium',ROOT/'models/tinyllama-1.1b-base']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.json','.safetensors','.model']:
                checkpoints[os.path.relpath(p,HERE)]={'sha256':digest(p),'bytes':p.stat().st_size}
    write(HERE/'data/checkpoints.json',checkpoints)
    paths=list((HERE/'code').glob('*.py'))+[HERE/'docs/PROTOCOL.md']+list((HERE/'data').glob('*.json'))+list((HERE/'results').glob('*.json'))
    for m in MODELS:paths+=list((HERE/'results'/m).glob('technical.json'))+list((HERE/'results'/m).glob('matched-boundary-replay.*'))
    paths+=list((ROOT/'src/transport_bridge').glob('*.py'))+list((PREV/'code').glob('*.py'))+list((ROOT/'studies/bridge-002/code').glob('*.py'))
    paths += [ROOT/'configs/study.json',PREV/'data/medium-selection.json']
    freeze('input-freeze.json',paths);print('INPUT FREEZE',now(),flush=True)

def discovery(model):
    verify('input-freeze.json')
    paths=[HERE/'data/input-freeze.json',HERE/'results'/model/'screen.json']
    if (HERE/'results'/model/'screen.npz').exists():paths.append(HERE/'results'/model/'screen.npz')
    freeze(model+'-discovery-freeze.json',paths)

def test(model):
    verify('input-freeze.json');verify(model+'-discovery-freeze.json')
    sel=read(HERE/'data'/f'{model}-selection.json');assert sel['source_heads']
    record=read(HERE/'results'/model/'discovery/run.json');assert record['status']=='completed'
    paths=[HERE/'data/input-freeze.json',HERE/'data'/f'{model}-selection.json',HERE/'data'/f'{model}-discovery-freeze.json']
    paths+=list((HERE/'results'/model/'discovery').glob('*'))+list((HERE/'results'/model/'localization').glob('*'))
    freeze(model+'-test-freeze.json',paths);print('TEST FREEZE',model,now(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inputs','discovery','test']);p.add_argument('--model',choices=MODELS);a=p.parse_args();inputs() if a.stage=='inputs' else globals()[a.stage](a.model)
