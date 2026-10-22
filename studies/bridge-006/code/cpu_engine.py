"""Exact causal-prefix reuse for END-only ablations, checked against Engine003.

The full edited run supplies every non-END key/value. Causal masking ensures an
END intervention cannot alter them. Only the END residual is propagated afresh.
No checkpoint weights, interventions, or measured outcomes are changed.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'studies/bridge-005/code'))
from common005 import Engine003
from transport_bridge.engine import outcomes


class PrefixCache:
    def __init__(self, cache, copies):
        self.cache, self.copies = cache, copies

    def update(self, key, value, layer, *args, **kwargs):
        prefix_k = self.cache[f'blocks.{layer}.attn.hook_k'][:, :-1].transpose(1, 2)
        prefix_v = self.cache[f'blocks.{layer}.attn.hook_v'][:, :-1].transpose(1, 2)
        return (torch.cat((prefix_k.repeat(self.copies, 1, 1, 1), key), dim=2),
                torch.cat((prefix_v.repeat(self.copies, 1, 1, 1), value), dim=2))


class CPUEngine(Engine003):
    def __init__(self):
        super().__init__('gpt2-medium', 'cpu')

    def end_suffix(self, hidden, start, cache, copies, hooks=()):
        prefix = PrefixCache(cache, copies)
        model = self.model.original_model
        with self.model.hooks(fwd_hooks=list(hooks)):
            for block in model.transformer.h[start:]:
                hidden = block(hidden, past_key_values=prefix, attention_mask=None, use_cache=True)
            return model.lm_head(model.transformer.ln_f(hidden))

    def signatures(self, rows, mean_bank, hooks=(), boundary=None):
        if boundary is None:
            base, cache = self.forward(rows, hooks)
        else:
            def replace(value, hook):
                return boundary.clone().to(self.device)
            base, cache = self.forward(rows, [(f'blocks.{self.site}.hook_resid_pre', replace)])
        n = len(rows)
        raw = np.empty((2, n, len(self.layers) * self.heads, 2), np.float32)
        chunk = self.config['head_chunk']
        for li, layer in enumerate(self.layers):
            hidden = cache[f'blocks.{layer}.hook_resid_pre'][:, -1:]
            for start in range(0, self.heads, chunk):
                variants = [(b, h) for b in range(2) for h in range(start, min(start + chunk, self.heads))]
                expanded = hidden.repeat(len(variants), 1, 1)
                which = torch.tensor([h for _, h in variants]).repeat_interleave(n)
                replacements = torch.stack([torch.zeros_like(mean_bank[li, h]) if b == 0 else mean_bank[li, h]
                                            for b, h in variants]).repeat_interleave(n, 0)
                def ablate(value, hook, *, which=which, replacements=replacements):
                    out = value.clone()
                    out[torch.arange(len(out)), -1, which] = replacements
                    return out
                logits = self.end_suffix(expanded, layer, cache, len(variants),
                                         [(f'blocks.{layer}.attn.hook_z', ablate)])
                values = outcomes(logits, rows * len(variants)).reshape(len(variants), n, 6)
                effects = (base[None, :, :2] - values[:, :, :2]).cpu().numpy()
                for vi, (bi, head) in enumerate(variants):
                    raw[bi, :, li * self.heads + head] = effects[vi]
        return raw, base.cpu().numpy()
