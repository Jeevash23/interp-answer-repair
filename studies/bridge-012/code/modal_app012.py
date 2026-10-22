"""Study 7 on Modal: one GPU (L4) and one CPU entry point, results on a Modal Volume.

From the AI-01 project root:
    modal run studies/bridge-012/code/modal_app012.py --stages calibration,calibration_cpu
    AI01_STUDY7_VOLUME=ai01-study7-codetest modal run studies/bridge-012/code/modal_app012.py --stages all --test

GPU stages run on an L4 (timeout 1 h, no warm containers, at most one container). The model and
packages are fetched inside Modal; the local machine uploads code and banks and receives only JSON records.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REMOTE = '/root/ai01'
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1] if len(HERE.parents) > 1 and (HERE.parents[1] / 'uv.lock').exists() else Path(REMOTE)
RESULTS = f'{REMOTE}/studies/bridge-012/results'
VOLUME = os.environ.get('AI01_STUDY7_VOLUME', 'ai01-study7')
GPU = os.environ.get('AI01_STUDY7_GPU', 'L4')
WEIGHTS = {
    'gpt2-medium': dict(repo='openai-community/gpt2-medium', revision='6dcaa7a952f72f9298047fd5137cd6e4f05f41da',
                        sha256='fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd', bytes=1519984962),
    'gpt2-large': dict(repo='openai-community/gpt2-large', revision='32b71b12589c2f8d625668d2335a01cac3249519',
                       sha256='5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce', bytes=3247159078),
}
MODEL_FILES = ['config.json', 'generation_config.json', 'merges.txt', 'vocab.json', 'tokenizer.json',
               'tokenizer_config.json', 'model.safetensors']
CODE_DIRS = ['src', 'studies/bridge-002/code', 'studies/bridge-003/code', 'studies/bridge-004/code',
             'studies/bridge-005/code', 'studies/bridge-006/code', 'studies/bridge-007/code', 'studies/bridge-011/code',
             'studies/bridge-012/code', 'studies/bridge-012/data', 'studies/bridge-012/docs']
DATA_FILES = ['configs/study.json', 'studies/transport/results/experiment-004/localization/selected-heads.json',
              'pyproject.toml', 'uv.lock']
GPU_STAGES = ['calibration', 'screen', 'sources', 'fit_select', 'development', 'qualification', 'test', 'replay_gpu']
CPU_STAGES = ['calibration_cpu', 'replay_cpu', 'analysis', 'audit']
ORDER = ['calibration', 'calibration_cpu', 'screen', 'sources', 'fit_select', 'development', 'qualification', 'test',
         'replay_gpu', 'replay_cpu', 'analysis', 'audit']


def download_weights():
    """Image build step: pinned downloads inside Modal, checked against the recorded hashes."""
    import hashlib as h
    from pathlib import Path as P
    from huggingface_hub import hf_hub_download
    for name, spec in WEIGHTS.items():
        target = P(REMOTE) / 'models' / name
        target.mkdir(parents=True, exist_ok=True)
        for f in MODEL_FILES:
            hf_hub_download(spec['repo'], f, revision=spec['revision'], local_dir=str(target))
        weights = target / 'model.safetensors'
        with weights.open('rb') as stream:
            digest = h.file_digest(stream, 'sha256').hexdigest()
        assert digest == spec['sha256'] and weights.stat().st_size == spec['bytes'], (name, digest)


image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(uv_project_dir='./', extras=['experiment'], frozen=True)
         .run_function(download_weights)
         .env({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'PYTHONDONTWRITEBYTECODE': '1'}))
for d in CODE_DIRS:
    image = image.add_local_dir(ROOT / d, f'{REMOTE}/{d}', ignore=['**/__pycache__', '**/*.pyc'])
for f in DATA_FILES:
    image = image.add_local_file(ROOT / f, f'{REMOTE}/{f}')

volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
APP_NAME = 'ai01-study7-codetest' if 'codetest' in VOLUME else 'ai01-study7'
app = modal.App(APP_NAME, image=image, tags={'project': 'AI-01', 'study': 'bridge-012'})


def _run(stage: str, test: bool, volume_name: str) -> str:
    os.chdir(REMOTE)
    sys.path[:0] = [f'{REMOTE}/studies/bridge-012/code']
    vol = modal.Volume.from_name(volume_name)
    vol.reload()
    started = time.time()
    if stage == 'analysis':
        import analyze012
        result = analyze012.run(test)
    elif stage == 'audit':
        import audit012
        result = audit012.run(test)
    else:
        import stages012
        fn = {**stages012.STAGES, **stages012.CPU_STAGES}[stage]
        ctx = stages012.Ctx(test)
        ctx.commit = vol.commit
        result = fn(ctx)
    vol.commit()
    keep = {k: v for k, v in result.items() if k not in ('scores', 'reference', 'cells', 'ranking', 'kappa_grid')}
    return json.dumps(dict(stage=stage, seconds=time.time() - started, result=keep), default=str)


@app.function(gpu=GPU, timeout=3600, volumes={RESULTS: volume}, memory=32768, max_containers=1)
def gpu_stage(stage: str, test: bool, volume_name: str) -> str:
    return _run(stage, test, volume_name)


@app.function(cpu=4.0, memory=24576, timeout=3600, volumes={RESULTS: volume}, max_containers=1)
def cpu_stage(stage: str, test: bool, volume_name: str) -> str:
    return _run(stage, test, volume_name)


def _billing():
    try:
        out = subprocess.run([sys.executable, '-m', 'modal', 'billing', 'report', '--for', 'today', '--json'],
                             capture_output=True, text=True, timeout=120).stdout
        return round(sum(float(r['cost']) for r in json.loads(out) if r['description'] == APP_NAME), 4)
    except Exception as error:  # billing is advisory; never stops a stage
        return f'unavailable: {error}'


@app.local_entrypoint()
def main(stages: str = 'calibration', test: bool = False):
    plan = ORDER if stages == 'all' else stages.split(',')
    assert all(s in ORDER for s in plan), plan
    log = HERE / 'provenance' / ('codetest-log.jsonl' if test else 'stage-log.jsonl')
    log.parent.mkdir(parents=True, exist_ok=True)
    for stage in plan:
        fn = gpu_stage if stage in GPU_STAGES else cpu_stage
        t0 = time.time()
        reply = json.loads(fn.remote(stage, test, VOLUME))
        entry = dict(stage=stage, test=test, volume=VOLUME, gpu=GPU if stage in GPU_STAGES else 'cpu', client_seconds=time.time() - t0,
                     remote_seconds=reply['seconds'], result=reply['result'], app=APP_NAME, billing_app_today_usd=_billing(),
                     utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        with log.open('a') as stream:
            stream.write(json.dumps(entry, default=str) + '\n')
        print(json.dumps(entry, default=str)[:2000], flush=True)
        if not test and (reply['result'].get('stopped') or reply['result'].get('passed') is False):
            print(f'STOPPED at {stage}', flush=True)
            break
