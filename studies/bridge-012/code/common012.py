"""Study 7 engine and measurement helpers. No inference on import.

The engine is Study 6's (causal-prefix END signatures; heads held at the bank at END in the base
pass and every ablated pass), generalised to a model specification so that GPT-2 Large can be
measured without changing configs/study.json. GPT-2 Medium is used only by the code test.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'studies/bridge-007/code')]
import runner007  # noqa: E402  (adds src/ and bridge-006/code to the path)
from runner007 import _measure, save_arrays, load_arrays, now  # noqa: E402,F401
from cpu_engine import CPUEngine  # noqa: E402
from common005 import batches, delta, hooks, donor_indices  # noqa: E402,F401
from common003 import config  # noqa: E402
from assay import CausalEngine  # noqa: E402
from transport_bridge.engine import Engine, outcomes, result_hooks  # noqa: E402

TOL = 5e-4
SPECS = {
    'gpt2-large': dict(site=19, layers=list(range(19, 36)), heads=20, width=1280, n_layers=36, source_layers=list(range(19)),
                       technical_sources=[(layer, (7 * layer) % 20) for layer in range(7, 19)],
                       technical_held=[(22, 0), (27, 5), (31, 9)]),
    # Code test only (never a scientific measurement in Study 7).
    'gpt2-medium': dict(site=13, layers=list(range(13, 24)), heads=16, width=1024, n_layers=24, source_layers=list(range(13)),
                        technical_sources=[(layer, (7 * layer) % 16) for layer in range(1, 13)],
                        technical_held=[(16, 0), (19, 5), (22, 9)]),
}
PAIRS = [[10, 11], [10, 13], [11, 12], [12, 13]]
KAPPAS = [round(0.50 + 0.05 * i, 2) for i in range(11)]
RANDOM_SEED = 2026101703


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows_for(split, families=None):
    rows = [r for r in read(HERE / 'data/rows.json') if r['split'] == split]
    if families is None:
        return rows
    return [dict(r, wording=families.index(r['family'])) for r in rows if r['family'] in families]


class Engine012(CPUEngine):
    def __init__(self, model, sources, device='cuda'):
        spec = SPECS[model]
        cfg = config()
        cfg['models'][model] = dict(site=spec['site'], layers=spec['layers'], heads=spec['heads'],
                                    width=spec['width'], source_heads=[list(s) for s in sources])
        cfg = dict(cfg, head_chunk=spec['heads'])
        CausalEngine.__init__(self, model, cfg, device)
        self.model_name, self.n_layers = model, spec['n_layers']
        self.final_resid = f'blocks.{spec["n_layers"] - 1}.hook_resid_post'
        self.names = self.names + [self.final_resid]
        self.runner_batch_size = 8

    def head_index(self, layer, head):
        return self.layers.index(layer) * self.heads + head

    def hold_hooks(self, bank, held):
        result = []
        for layer in sorted({l for l, _ in held}):
            heads = [h for l, h in held if l == layer]
            values = bank[self.layers.index(layer), heads].to(self.device)

            def hold(value, hook, *, heads=heads, values=values):
                del hook
                out = value.clone()
                out[:, -1, heads] = values
                return out
            result.append((f'blocks.{layer}.attn.hook_z', hold))
        return result

    def measure(self, rows, bank, edits=(), boundary=None, held=(), prefix=True):
        """END signature under zero and bank references; held heads stay at the bank in every pass."""
        bank = torch.as_tensor(bank, dtype=torch.float32).to(self.device)
        hold = self.hold_hooks(bank, held) if held else []
        base_hooks = list(edits) + hold
        if boundary is not None:
            def replace(value, hook):
                del value, hook
                return boundary.clone().to(self.device)
            base_hooks = [(f'blocks.{self.site}.hook_resid_pre', replace)] + base_hooks
        base, cache = self.forward(rows, base_hooks)
        n = len(rows)
        raw = np.empty((2, n, len(self.layers) * self.heads, 2), np.float32)
        chunk = self.config['head_chunk']
        for li, layer in enumerate(self.layers):
            for start in range(0, self.heads, chunk):
                variants = [(b, h) for b in range(2) for h in range(start, min(start + chunk, self.heads))]
                which = torch.tensor([h for _, h in variants], device=self.device).repeat_interleave(n)
                replacements = torch.stack([torch.zeros_like(bank[li, h]) if b == 0 else bank[li, h]
                                            for b, h in variants]).repeat_interleave(n, 0)

                def ablate(value, hook, *, which=which, replacements=replacements):
                    del hook
                    out = value.clone()
                    out[torch.arange(len(out), device=out.device), -1, which] = replacements
                    return out
                suffix_hooks = hold + [(f'blocks.{layer}.attn.hook_z', ablate)]
                if prefix:
                    hidden = cache[f'blocks.{layer}.hook_resid_pre'][:, -1:]
                    logits = self.end_suffix(hidden.repeat(len(variants), 1, 1), layer, cache, len(variants), suffix_hooks)
                else:
                    hidden = cache[f'blocks.{layer}.hook_resid_pre']
                    logits = Engine.suffix(self, hidden.repeat(len(variants), 1, 1), layer, suffix_hooks)
                values = outcomes(logits, rows * len(variants)).reshape(len(variants), n, 6)
                effects = (base[None, :, :2] - values[:, :, :2]).cpu().numpy()
                for vi, (bi, head) in enumerate(variants):
                    raw[bi, :, li * self.heads + head] = effects[vi]
        return raw, base.cpu().numpy(), cache

    def attribution(self, rows, cache):
        model = self.model.original_model
        ln, head = model.transformer.ln_f, model.lm_head
        x = cache[self.final_resid][:, -1].double()
        sigma = (x - x.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).add(ln.eps).sqrt()
        gamma, beta = ln.weight.double(), ln.bias.double()
        unembed = head.weight.double()
        io = torch.tensor([r['io_id'] for r in rows], device=self.device)
        sub = torch.tensor([r['subject_id'] for r in rows], device=self.device)
        direction = unembed[io] - unembed[sub]
        constant = (direction * beta).sum(-1)
        if head.bias is not None:
            constant = constant + head.bias.double()[io] - head.bias.double()[sub]
        total = (((x - x.mean(-1, keepdim=True)) / sigma * gamma) * direction).sum(-1) + constant
        z = torch.stack([cache[f'blocks.{l}.attn.hook_z'][:, -1] for l in self.layers], 1).double()
        out = torch.einsum('nlhk,lhkd->nlhd', z, self.weights[self.layers].double())
        out = out - out.mean(-1, keepdim=True)
        dla = ((out / sigma[:, None, None] * gamma) * direction[:, None, None]).sum(-1)
        attention = self.attention(cache, [(l, h) for l in self.layers for h in range(self.heads)], rows)
        return dict(dla=dla.reshape(len(rows), -1).float().cpu().numpy(),
                    linearised_gap=total.float().cpu().numpy(), attention=attention.astype(np.float32))


def recipient_rows(rows, ids, recipient):
    return [dict(rows[i], token_ids=rows[i]['corrupt_tokens']) if recipient == 'corrupt' else rows[i] for i in ids]


@torch.inference_mode()
def measure_natural(e, rows, out, freeze, recipient, store_z=False):
    """Behaviour and source-head S2 outputs (and END z when the bank is fitted)."""
    def run(ids):
        rr = recipient_rows(rows, ids, recipient)
        behavior, cache = e.forward(rr)
        _, z, head_results = e.extract(rr, cache)
        value = dict(behavior=behavior.cpu().numpy(), head_results=head_results.numpy())
        if store_z:
            value['z'] = z.numpy()
        return value
    return _measure(e, rows, out, freeze, f'natural|recipient={recipient}|z={store_z}', run)


@torch.inference_mode()
def measure_signature(e, rows, bank, out, freeze, recipient, d=None, held=(), donors=None):
    bank = np.asarray(bank, np.float32)
    d = None if d is None else torch.as_tensor(d, dtype=torch.float32)
    kind = (f'signature|recipient={recipient}|held={sorted(map(list, held))}|'
            f'boundary={"wording-donor-site" if donors is not None else None}')

    def run(ids):
        rr = recipient_rows(rows, ids, recipient)
        edits = () if d is None else hooks(e, d[ids], rr)
        boundary = None
        if donors is not None:
            _, cache = e.forward([rows[donors[i]] for i in ids])
            boundary = cache[f'blocks.{e.site}.hook_resid_pre']
        raw, behavior, _ = e.measure(rr, bank, edits, boundary, held)
        return dict(raw=raw, behavior=behavior)
    return _measure(e, rows, out, freeze, kind, run, bank, d)


@torch.inference_mode()
def measure_behaviour(e, rows, bank, out, freeze, recipient, d=None, held=()):
    d = None if d is None else torch.as_tensor(d, dtype=torch.float32)
    kind = f'behavior|recipient={recipient}|held={sorted(map(list, held))}'

    def run(ids):
        rr = recipient_rows(rows, ids, recipient)
        edits = [] if d is None else hooks(e, d[ids], rr)
        held_hooks = e.hold_hooks(torch.as_tensor(bank), held) if held else []
        return dict(behavior=e.forward(rr, edits + held_hooks)[0].cpu().numpy())
    return _measure(e, rows, out, freeze, kind, run, delta=d)['behavior']


@torch.inference_mode()
def measure_attribution(e, rows, bank, out, freeze, recipient, d=None, held=()):
    d = None if d is None else torch.as_tensor(d, dtype=torch.float32)
    kind = f'attribution|recipient={recipient}|held={sorted(map(list, held))}'

    def run(ids):
        rr = recipient_rows(rows, ids, recipient)
        edits = [] if d is None else hooks(e, d[ids], rr)
        held_hooks = e.hold_hooks(torch.as_tensor(bank), held) if held else []
        behavior, cache = e.forward(rr, edits + held_hooks)
        return dict(behavior=behavior.cpu().numpy(), **e.attribution(rr, cache))
    return _measure(e, rows, out, freeze, kind, run, delta=d)


def random_delta(e, dd, rows, kind, draw):
    """Study 2's norm-matched random controls (per head), with Study 7's seed."""
    result = []
    for hi, (layer, head) in enumerate(e.source):
        weight = e.weights[layer, head].cpu()
        vectors = []
        for r in rows:
            label = f'{RANDOM_SEED}:{kind}:{draw}:{r["id"]}:{layer}:{head}'
            seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'little') % (2 ** 63 - 1)
            vectors.append(torch.randn(weight.shape[0] if kind == 'projected' else weight.shape[1],
                                       generator=torch.Generator().manual_seed(seed)))
        v = torch.stack(vectors)
        if kind == 'projected':
            v = v @ weight
        v *= dd[:, hi].norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result.append(v)
    return torch.stack(result, 1)


def freeze(path, files, kind):
    """Write a stage freeze once (paths relative to the project root), then verify it."""
    path = Path(path)
    if not path.exists():
        files = sorted({Path(p).absolute() for p in files if Path(p).is_file()})  # Volume mounts are symlinks: never resolve
        write(path, dict(utc=now(), kind=kind, files={str(p.relative_to(ROOT)): sha(p) for p in files}))
    for name, expected in read(path)['files'].items():
        assert sha(ROOT / name) == expected, name
    return sha(path)
