"""Study 6 engine and fixed definitions. No inference on import.

The engine extends the inherited causal-prefix signature engine in one way: heads can be
held at the fit-mean bank at END in the base pass and in every recomputed ablation pass.
Edits, references, scores and the prefix shortcut keep Study 2's semantics.
"""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = ROOT / 'studies/bridge-007'
sys.path[:0] = [str(BASE / 'code')]
import runner007  # noqa: E402  (adds src/ and bridge-006/code to the path)
from runner007 import _measure, save_arrays, load_arrays, digest, now, verify_freeze  # noqa: E402,F401
from cpu_engine import CPUEngine, Engine003  # noqa: E402
from common005 import batches, delta, hooks  # noqa: E402,F401
from transport_bridge.engine import outcomes  # noqa: E402
from analyze007 import read, write, load  # noqa: E402,F401

TOL = 5e-4
REFS = ('zero', 'fit_mean')
DRAWS = 50_000
BOOT_SEED = 2026101603
MULTIPLICITY = 6
KAPPAS = [round(0.50 + 0.05 * i, 2) for i in range(11)]
TECH_HELD = [(16, 0), (19, 5), (22, 9)]
BATCH = 8
HEAD_CHUNK = 8
FREEZE = HERE / 'data/input-freeze.json'
DEV_FREEZE = HERE / 'data/development-freeze.json'
FINAL_RESID = 'blocks.23.hook_resid_post'
SIGNATURES = ('clean', 'corrupt', 'wording', 'self', 'self_kappa', 'n1', 'n2', 'order', 'replay')


def rows_for(split):
    return [r for r in read(HERE / 'data/rows.json') if r['split'] == split]


def fit_bank():
    return load(BASE / 'results/fit/bank.npz')['bank']


def reference():
    return read(BASE / 'data/selection.json')['reference']


def head_index(layer, head):
    return (layer - 13) * 16 + head


class Engine011(CPUEngine):
    def __init__(self, device='cuda'):
        Engine003.__init__(self, 'gpt2-medium', device)
        self.config = dict(self.config, head_chunk=HEAD_CHUNK)
        self.runner_batch_size = BATCH
        if FINAL_RESID not in self.names:
            self.names = self.names + [FINAL_RESID]

    def hold_hooks(self, bank, held):
        """Hold the listed END heads at their bank values (the only position in a suffix pass)."""
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
        """176-head END signature under zero and bank references; Study 2 coordinates.

        Held heads stay at the bank in the base pass and in every ablated pass; the
        coordinate's own ablation hook runs after them. prefix=False recomputes the
        whole sequence from each layer, as the inherited full-sequence reference does.
        """
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
                    logits = self.end_suffix(hidden.repeat(len(variants), 1, 1), layer, cache,
                                             len(variants), suffix_hooks)
                else:
                    hidden = cache[f'blocks.{layer}.hook_resid_pre']
                    logits = Engine003.suffix(self, hidden.repeat(len(variants), 1, 1), layer, suffix_hooks)
                values = outcomes(logits, rows * len(variants)).reshape(len(variants), n, 6)
                effects = (base[None, :, :2] - values[:, :, :2]).cpu().numpy()
                for vi, (bi, head) in enumerate(variants):
                    raw[bi, :, li * self.heads + head] = effects[vi]
        return raw, base.cpu().numpy(), cache

    def attribution(self, rows, cache):
        """Direct logit attribution of each END head to the IO-minus-subject gap.

        Each head's END output passes through the final LayerNorm with the run's own
        scale held fixed. Also returns the same linearisation of the whole final
        residual, which must reproduce the logit gap, and END attention to IO, S1 and S2.
        """
        model = self.model.original_model
        ln, head = model.transformer.ln_f, model.lm_head
        x = cache[FINAL_RESID][:, -1].double()
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
        group = [(l, h) for l in self.layers for h in range(self.heads)]
        attention = self.attention(cache, group, rows)
        return dict(dla=dla.reshape(len(rows), -1).float().cpu().numpy(),
                    linearised_gap=total.float().cpu().numpy(), attention=attention.astype(np.float32))


def recipient_rows(rows, ids, recipient):
    return [dict(rows[i], token_ids=rows[i]['corrupt_tokens']) if recipient == 'corrupt' else rows[i] for i in ids]


@torch.inference_mode()
def measure_signature(e, rows, out, freeze, recipient, d=None, held=(), donors=None):
    """Resumable, hash-recorded signature measurement (Study 2 record format)."""
    bank = fit_bank()
    d = None if d is None else torch.as_tensor(d, dtype=torch.float32)
    kind = (f'signature|recipient={recipient}|held={sorted(map(list, held))}|'
            f'boundary={"wording-donor-layer13" if donors is not None else None}')

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
def measure_behaviour(e, rows, out, freeze, recipient, d=None, held=()):
    bank = torch.as_tensor(fit_bank())
    d = None if d is None else torch.as_tensor(d, dtype=torch.float32)
    kind = f'behavior|recipient={recipient}|held={sorted(map(list, held))}'

    def run(ids):
        rr = recipient_rows(rows, ids, recipient)
        edits = [] if d is None else hooks(e, d[ids], rr)
        held_hooks = e.hold_hooks(bank, held) if held else []
        return dict(behavior=e.forward(rr, edits + held_hooks)[0].cpu().numpy())

    return _measure(e, rows, out, freeze, kind, run, delta=d)['behavior']


@torch.inference_mode()
def measure_attribution(e, rows, out, freeze, recipient, d=None, held=()):
    bank = torch.as_tensor(fit_bank())
    d = None if d is None else torch.as_tensor(d, dtype=torch.float32)
    kind = f'attribution|recipient={recipient}|held={sorted(map(list, held))}'

    def run(ids):
        rr = recipient_rows(rows, ids, recipient)
        edits = [] if d is None else hooks(e, d[ids], rr)
        held_hooks = e.hold_hooks(bank, held) if held else []
        behavior, cache = e.forward(rr, edits + held_hooks)
        return dict(behavior=behavior.cpu().numpy(), **e.attribution(rr, cache))

    return _measure(e, rows, out, freeze, kind, run, delta=d)
