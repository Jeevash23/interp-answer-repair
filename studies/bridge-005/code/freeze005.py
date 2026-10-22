"""Freeze once, before any Bridge 005 GPU inference."""
from common005 import *

def main():
    assert read(HERE/'results/design-tests.json')['status']=='passed'
    old=read(ROOT/'output/combined-artifact-manifest-v4.json');n=0
    for name,sha in old['files'].items():assert digest(ROOT.parent/name)==sha,name;n+=1
    files={}
    for p in (ROOT / 'models/gpt2-medium').iterdir():
        if p.is_file() and p.suffix in ['.json','.safetensors']:files[os.path.relpath(p,HERE)]=dict(sha256=digest(p),bytes=p.stat().st_size)
    write(HERE/'data/checkpoint.json',files)
    write(HERE/'data/prior-preservation.json',dict(utc=now(),files=n,manifest_sha256=digest(ROOT/'output/combined-artifact-manifest-v4.json'),draft_v2={ext:digest(ROOT/'paper/draft-v2'/('manuscript.'+ext)) for ext in ['pdf','docx']}))
    paths=list((HERE/'code').glob('*.py'))+list((HERE/'docs').glob('*.md'))+list((HERE/'data').glob('*.json'))+[HERE/'results/design-tests.json',ROOT/'uv.lock',ROOT/'pyproject.toml',ROOT/'configs/study.json',ROOT / 'studies/transport/results/experiment-004/localization/selected-heads.json']
    for folder in [ROOT/'src/transport_bridge',ROOT/'studies/bridge-002/code',ROOT/'studies/bridge-003/code',PREV/'code']:paths+=list(folder.glob('*.py'))
    freeze('input-freeze.json',paths)
    print('INPUTS FROZEN',now(),'prior files preserved',n,flush=True)
if __name__=='__main__':main()
