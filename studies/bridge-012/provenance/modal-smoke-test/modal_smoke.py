"""Modal smoke test for Study 7 (GPT-2 Large on Modal): reproduce saved Study 6 results remotely.

Run from the AI-01 project root (the image's uv environment is built from ./pyproject.toml and ./uv.lock):

    modal run studies/bridge-012/provenance/modal-smoke-test/modal_smoke.py

1. Reproduction. On a Modal T4, recompute Study 6's eight fixed held-out configurations
   (the singleton-replay IDs) in GPT-2 Medium for all nine signature conditions, with the
   frozen Study 6 engine, and compare them locally with the saved held-out arrays at the
   project tolerance (5e-4 logits).
2. Timing. Time the same signature computation for GPT-2 Large on random token inputs
   (no task prompts) on a T4 and an L4, for the Study 7 cost estimate.

Weights download inside Modal from Hugging Face at the pinned revisions and are checked
against the recorded SHA-256 hashes; the local machine uploads only code and small inputs and
receives only small result arrays. Nothing is written outside this folder.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

REMOTE = '/root/ai01'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] if len(HERE.parents) > 3 else Path(REMOTE)  # the container sees a shallow path
TOL = 5e-4
WEIGHTS = {
    'gpt2-medium': dict(repo='openai-community/gpt2-medium', revision='6dcaa7a952f72f9298047fd5137cd6e4f05f41da',
                        sha256='fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd', bytes=1519984962),
    'gpt2-large': dict(repo='openai-community/gpt2-large', revision='32b71b12589c2f8d625668d2335a01cac3249519',
                       sha256='5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce', bytes=3247159078),
}
MODEL_FILES = ['config.json', 'generation_config.json', 'merges.txt', 'vocab.json', 'tokenizer.json',
               'tokenizer_config.json', 'model.safetensors']
CODE_DIRS = ['src', 'studies/bridge-002/code', 'studies/bridge-003/code', 'studies/bridge-004/code',
             'studies/bridge-005/code', 'studies/bridge-006/code', 'studies/bridge-007/code', 'studies/bridge-011/code']
DATA_FILES = ['configs/study.json', 'studies/transport/results/experiment-004/localization/selected-heads.json',
              'studies/bridge-007/data/selection.json', 'studies/bridge-007/results/fit/bank.npz']


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

app = modal.App('ai01-study7-modal-smoke', image=image, tags={'project': 'AI-01', 'study': 'bridge-012-smoke'})


def _environment():
    import platform
    import importlib.metadata as md
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=str(torch.version.cuda),
                device=str(torch.cuda.get_device_name(0)), capability=[int(x) for x in torch.cuda.get_device_capability(0)],
                packages={p: md.version(p) for p in ('numpy', 'transformers', 'transformer-lens', 'huggingface-hub', 'safetensors')},
                model_hashes={n: _sha(f'{REMOTE}/models/{n}/{f}') for n in WEIGHTS for f in MODEL_FILES if f != 'model.safetensors'})


def _sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _large_timing(rows_per_run=16, chunks=(8, 20), batch=8, site=18):
    """GPT-2 Large signature timing on random token inputs (no task prompts)."""
    import numpy as np
    import torch
    sys.path[:0] = [f'{REMOTE}/studies/bridge-011/code']
    import common011 as c
    from assay import CausalEngine
    from common003 import config

    cfg = config()
    cfg['models']['gpt2-large'] = dict(site=site, layers=list(range(site, 36)), heads=20, width=1280,
                                       source_heads=[[layer, 0] for layer in range(site - 12, site)])

    class LargeEngine(c.Engine011):
        def __init__(self):
            CausalEngine.__init__(self, 'gpt2-large', cfg, 'cuda')
            self.model_name = 'gpt2-large'

    rng = np.random.default_rng(0)
    rows = []
    for i in range(rows_per_run):
        tokens = [50256] + rng.integers(1000, 50000, size=19).tolist()
        rows.append(dict(id=f'random-{i}', token_ids=tokens, corrupt_tokens=tokens, io_id=int(tokens[3]),
                         subject_id=int(tokens[5]), alt_id=int(tokens[7]), s2=10, io_position=3, subject_position=5))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    e = LargeEngine()
    load_seconds = time.perf_counter() - started
    bank = torch.as_tensor(rng.normal(0, 0.1, size=(36 - site, 20, 64)), dtype=torch.float32)
    held = [(site + 2, 3), (site + 5, 7), (site + 9, 11)]
    out = dict(load_seconds=load_seconds, rows=rows_per_run, batch=batch, site=site,
               measured_heads=(36 - site) * 20, runs=[])
    with torch.inference_mode():
        for chunk in chunks:
            e.config = dict(e.config, head_chunk=chunk)
            for held_heads in ((), held):
                for repeat in range(2):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for start in range(0, rows_per_run, batch):
                        e.measure(rows[start:start + batch], bank, (), None, held_heads)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - t0
                    out['runs'].append(dict(head_chunk=chunk, held=bool(held_heads), repeat=repeat,
                                            seconds_per_row=seconds / rows_per_run))
    out['peak_gpu_gb'] = float(torch.cuda.max_memory_allocated() / 1e9)
    return out


@app.function(gpu='T4', timeout=900, max_containers=1)
def reproduce_and_time(payload: bytes) -> bytes:
    import numpy as np
    import torch
    sys.path[:0] = [f'{REMOTE}/studies/bridge-011/code']
    import common011 as c
    from common005 import delta, hooks
    task = pickle.loads(payload)
    rows, ii = task['rows'], task['recipients']
    held, kappa = [tuple(h) for h in task['held']], task['kappa']
    result = dict(started=datetime.now(timezone.utc).isoformat(), environment=_environment())
    t0 = time.perf_counter()
    e = c.Engine011('cuda')
    result['medium_load_seconds'] = time.perf_counter() - t0
    bank = torch.tensor(c.fit_bank())
    with torch.inference_mode():
        t1 = time.perf_counter()
        hr = {}
        for kind in ('clean', 'corrupt'):
            hr[kind] = np.empty((len(rows), 12, 1024), np.float32)
            for part in c.batches(rows, 8):
                rr = [dict(rows[i], token_ids=rows[i]['corrupt_tokens']) if kind == 'corrupt' else rows[i] for i in part]
                _, cache = e.forward(rr)
                hr[kind][part] = e.extract(rr, cache)[2].numpy()
        nat = {'clean_head_results': hr['clean'], 'corrupt_head_results': hr['corrupt']}
        d_self = delta(nat, rows, 'self')
        index = {(r['template'], r['pair'], r['role'], r['wording'], r['order']): k for k, r in enumerate(rows)}
        donor = lambda r, flip: index[(r['template'], r['pair'], r['role'], r['wording'] ^ (flip == 'wording'),
                                       r['order'] ^ (flip == 'order'))]
        conds = dict(clean=('clean', None, (), None), corrupt=('corrupt', None, (), None),
                     wording=('corrupt', 'wording', (), None), self=('corrupt', 'self', (), None),
                     order=('corrupt', 'order', (), None), replay=('corrupt', None, (), 'wording'),
                     self_kappa=('corrupt', 'self_kappa', (), None), n1=('clean', None, held, None),
                     n2=('corrupt', 'self', held, None))
        sigs = {}
        for name, (recipient, edit, h, replay) in conds.items():
            raw = np.empty((2, len(ii), 176, 2), np.float32); beh = np.empty((len(ii), 6), np.float32)
            for k, i in enumerate(ii):
                r = rows[i]
                rr = [dict(r, token_ids=r['corrupt_tokens']) if recipient == 'corrupt' else r]
                if edit is None:
                    d = None
                elif edit in ('self', 'self_kappa'):
                    d = d_self[[i]] * (kappa if edit == 'self_kappa' else 1.0)
                else:
                    j = donor(r, edit)
                    d = torch.tensor(hr['clean'][[j]] - hr['corrupt'][[j]])
                edits = () if d is None else hooks(e, d, rr)
                boundary = None
                if replay:
                    _, cache = e.forward([rows[donor(r, 'wording')]])
                    boundary = cache[f'blocks.{e.site}.hook_resid_pre']
                x, y, _ = e.measure(rr, bank, edits, boundary, h)
                raw[:, k], beh[k] = x[:, 0], y[0]
            sigs[name] = dict(raw=raw, behavior=beh)
        torch.cuda.synchronize()
        result['reproduction_seconds'] = time.perf_counter() - t1
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{f'{k}__{m}': v[m] for k, v in sigs.items() for m in ('raw', 'behavior')})
    result['npz'] = buffer.getvalue()
    result['large_timing'] = _large_timing()
    result['finished'] = datetime.now(timezone.utc).isoformat()
    return pickle.dumps(result)


@app.function(gpu='L4', timeout=600, max_containers=1)
def time_large_l4() -> bytes:
    return pickle.dumps(dict(environment=_environment(), large_timing=_large_timing()))


@app.local_entrypoint()
def main():
    out_dir = HERE / 'records'
    assert not out_dir.exists(), out_dir
    study6 = ROOT / 'studies/bridge-011'
    test = [r for r in json.loads((study6 / 'data/rows.json').read_text()) if r['split'] == 'test']
    index = {(r['template'], r['pair'], r['role'], r['wording'], r['order']): k for k, r in enumerate(test)}
    singles = json.loads((study6 / 'data/design.json').read_text())['singleton_ids']
    ids = {r['id']: k for k, r in enumerate(test)}
    needed = []
    for sid in singles:
        r = test[ids[sid]]
        for w, o in ((0, 0), (1, 0), (0, 1)):
            j = index[(r['template'], r['pair'], r['role'], r['wording'] ^ w, r['order'] ^ o)]
            if j not in needed:
                needed.append(j)
    rows = [test[j] for j in needed]
    recipients = [needed.index(ids[s]) for s in singles]
    dev = json.loads((study6 / 'results/development.json').read_text())
    payload = pickle.dumps(dict(rows=rows, recipients=recipients, held=dev['held'], kappa=dev['kappa']))
    uploaded = {d: sum(p.stat().st_size for p in (ROOT / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
                for d in CODE_DIRS}
    uploaded.update({f: (ROOT / f).stat().st_size for f in DATA_FILES + ['pyproject.toml', 'uv.lock']})
    t0 = time.perf_counter()
    t4_bytes = reproduce_and_time.remote(payload)
    t4_wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    l4_bytes = time_large_l4.remote()
    l4_wall = time.perf_counter() - t1
    t4, l4 = pickle.loads(t4_bytes), pickle.loads(l4_bytes)
    out_dir.mkdir()
    (out_dir / 'modal-signatures.npz').write_bytes(t4.pop('npz'))
    record = dict(
        purpose='Smoke test of Modal for Study 7: reproduce saved Study 6 GPT-2 Medium results remotely; time GPT-2 Large on random tokens',
        rows=singles, t4=dict(t4, client_wall_seconds=t4_wall), l4=dict(l4, client_wall_seconds=l4_wall), weights=WEIGHTS,
        image=dict(base='modal.Image.debian_slim(python_version="3.12")',
                   uv_sync=dict(extras=['experiment'], frozen=True, lock_sha256=hashlib.sha256((ROOT / 'uv.lock').read_bytes()).hexdigest())),
        transfer=dict(uploaded_bytes=sum(uploaded.values()) + len(payload), uploaded_detail=uploaded, payload_bytes=len(payload),
                      received_bytes=len(t4_bytes) + len(l4_bytes),
                      note='Weights and Python packages were downloaded inside Modal, not through the local machine'),
        utc=datetime.now(timezone.utc).isoformat())
    (out_dir / 'remote-record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(t4_wall=t4_wall, l4_wall=l4_wall, uploaded_mb=record['transfer']['uploaded_bytes'] / 1e6,
                          received_mb=record['transfer']['received_bytes'] / 1e6)))
