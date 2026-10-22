"""Float32 GPT-2 inference, including vectorized exact head interventions."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from .common import ROOT, SEED, stable_seed


def outcomes(logits, rows):
    last = logits[:, -1].float()
    ix = torch.arange(len(rows), device=last.device)
    io, sub, alt = [
        torch.tensor([r[k] for r in rows], device=last.device)
        for k in ["io_id", "subject_id", "alt_id"]
    ]
    p = last.softmax(-1)
    return torch.stack(
        [
            last[ix, io] - last[ix, sub],
            p[ix, io] - p[ix, sub],
            (last[ix, io] > last[ix, sub]).float(),
            (last.argmax(-1) == io).float(),
            last[ix, alt] - last[ix, io],
            p[ix, io],
        ],
        dim=1,
    )


def state_hook(layer, positions, delta):
    def add(value, hook):
        del hook
        v = value.clone()
        v[torch.arange(len(v), device=v.device), positions.to(v.device)] += delta.to(v.device)
        return v

    return f"blocks.{layer}.hook_resid_pre", add


def result_hooks(heads, deltas, positions):
    layers = sorted({h[0] for h in heads})
    hooks = []
    for layer in layers:
        delta = deltas[:, [i for i, h in enumerate(heads) if h[0] == layer]].sum(1)

        def add(value, hook, *, delta=delta):
            del hook
            v = value.clone()
            v[torch.arange(len(v), device=v.device), positions.to(v.device)] += delta.to(v.device)
            return v

        hooks.append((f"blocks.{layer}.hook_attn_out", add))
    return hooks


def random_matched(delta, ids, label, draw):
    out = []
    for d, rid in zip(delta.cpu(), ids, strict=True):
        g = torch.Generator().manual_seed(stable_seed(SEED, rid, label, draw))
        v = torch.randn(d.shape, generator=g)
        # Match per-head norms for groups and whole-vector norms for residual edits.
        v *= d.norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        out.append(v)
    return torch.stack(out).to(delta.device)


class Engine:
    def __init__(self, model_name, config, device="cuda"):
        from transformer_lens.model_bridge import TransformerBridge

        torch.set_num_threads(config["threads"])
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.spec = config["models"][model_name]
        self.config = config
        self.site = self.spec["site"]
        self.layers = self.spec["layers"]
        self.heads = self.spec["heads"]
        self.source = [tuple(x) for x in self.spec["source_heads"]]
        self.model = TransformerBridge.boot_transformers(
            str(ROOT / "models" / model_name),
            device=torch.device("cpu"),
            dtype=torch.float32,
            trust_remote_code=False,
        )
        self.model.enable_compatibility_mode()
        self.model.set_use_attn_result(False)
        self.model.to(self.device)
        self.model.eval()
        self.names = [f"blocks.{l}.hook_resid_pre" for l in self.layers]
        self.names += [
            f"blocks.{l}.attn.hook_z"
            for l in sorted(set(self.layers) | {h[0] for h in self.source})
        ]
        self.weights = self.model.W_O.detach()

    def forward(self, rows, hooks=()):
        t = torch.tensor([r["token_ids"] for r in rows], device=self.device)
        with self.model.hooks(fwd_hooks=list(hooks)):
            logits, cache = self.model.run_with_cache(t, names_filter=self.names)
        return outcomes(logits, rows), cache

    def suffix(self, hidden, start, hooks=()):
        n = hidden.shape[1]
        mask = torch.full((1, 1, n, n), torch.finfo(hidden.dtype).min, device=hidden.device).triu(1)
        m = self.model.original_model
        with self.model.hooks(fwd_hooks=list(hooks)):
            for block in m.transformer.h[start:]:
                hidden = block(hidden, attention_mask=mask, use_cache=False)
            return m.lm_head(m.transformer.ln_f(hidden[:, -1:]))

    def extract(self, rows, cache):
        ix = torch.arange(len(rows), device=self.device)
        pos = torch.tensor([r["s2"] for r in rows], device=self.device)
        state = cache[f"blocks.{self.site}.hook_resid_pre"][ix, pos]
        z = torch.stack([cache[f"blocks.{l}.attn.hook_z"][:, -1] for l in self.layers], 1)
        results = torch.stack(
            [
                cache[f"blocks.{l}.attn.hook_z"][ix, pos, h] @ self.weights[l, h]
                for l, h in self.source
            ],
            1,
        )
        return state.cpu(), z.cpu(), results.cpu()

    def signatures(self, rows, mean_bank, hooks=(), boundary=None):
        if boundary is None:
            base, cache = self.forward(rows, hooks)
        else:

            def replace(value, hook):
                del value, hook
                return boundary.clone().to(self.device)

            base, cache = self.forward(rows, [(f"blocks.{self.site}.hook_resid_pre", replace)])
        n, hcount = len(rows), self.heads
        raw = np.empty((2, n, len(self.layers) * hcount, 2), np.float32)
        chunk = self.config["head_chunk"]
        for li, layer in enumerate(self.layers):
            hidden = cache[f"blocks.{layer}.hook_resid_pre"]
            for start in range(0, hcount, chunk):
                variants = [
                    (b, h) for b in range(2) for h in range(start, min(start + chunk, hcount))
                ]
                expanded = hidden.repeat(len(variants), 1, 1)
                which = torch.tensor(
                    [h for _, h in variants], device=self.device
                ).repeat_interleave(n)
                replacements = torch.stack(
                    [
                        torch.zeros_like(mean_bank[li, h]) if b == 0 else mean_bank[li, h]
                        for b, h in variants
                    ]
                )
                replacements = replacements.repeat_interleave(n, 0).to(self.device)

                def ablate(value, hook, *, which=which, replacements=replacements):
                    del hook
                    v = value.clone()
                    v[torch.arange(len(v), device=v.device), -1, which] = replacements
                    return v

                logits = self.suffix(expanded, layer, [(f"blocks.{layer}.attn.hook_z", ablate)])
                values = outcomes(logits, rows * len(variants)).reshape(len(variants), n, 6)
                effects = (base[None, :, :2] - values[:, :, :2]).cpu().numpy()
                for vi, (bi, head) in enumerate(variants):
                    raw[bi, :, li * hcount + head] = effects[vi]
        return raw, base.cpu().numpy()

    def validate(self, rows):
        tokens = torch.tensor([r["token_ids"] for r in rows], device=self.device)
        pos = torch.tensor([r["s2"] for r in rows], device=self.device)
        _, cache = self.forward(rows)
        direct = self.model(tokens)[:, -1:]
        errors = []
        for layer in [self.site, self.layers[-1]]:
            errors.append(
                float(
                    (self.suffix(cache[f"blocks.{layer}.hook_resid_pre"], layer) - direct)
                    .abs()
                    .max()
                )
            )
        delta = torch.randn(len(rows), self.spec["width"], device=self.device) * 0.05
        edits = [
            state_hook(self.site, pos, delta),
            *result_hooks(
                self.source,
                torch.randn(len(rows), len(self.source), self.spec["width"], device=self.device)
                * 0.01,
                pos,
            ),
        ]
        for edit in edits:
            _, edited = self.forward(rows, [edit])
            layer = self.layers[0]

            def zero(value, hook):
                del hook
                v = value.clone()
                v[:, -1, 0] = 0
                return v

            ab = (f"blocks.{layer}.attn.hook_z", zero)
            slow = self.model.run_with_hooks(tokens, fwd_hooks=[edit, ab])[:, -1:]
            fast = self.suffix(edited[f"blocks.{layer}.hook_resid_pre"], layer, [ab])
            errors.append(float((slow - fast).abs().max()))
        bank = torch.stack(
            [cache[f"blocks.{l}.attn.hook_z"][:, -1].mean(0).cpu() for l in self.layers]
        )
        raw, baseline = self.signatures(rows, bank)
        # Explicit whole-boundary positive control, including every measured coordinate.
        corrupted_rows = []
        for r in rows:
            ids = list(r["token_ids"])
            ids[r["s2"]] = r["io_id"]
            corrupted_rows.append({**r, "token_ids": ids})
        transplanted, restored = self.signatures(
            corrupted_rows, bank, boundary=cache[f"blocks.{self.site}.hook_resid_pre"].cpu()
        )
        errors += [
            float(np.max(np.abs(raw - transplanted))),
            float(np.max(np.abs(baseline - restored))),
        ]
        identity, _ = self.forward(rows, [state_hook(self.site, pos, torch.zeros_like(delta))])
        errors.append(float(np.max(np.abs(identity.cpu().numpy() - baseline))))
        # Result-sum insertion equals insertion at the individual head result.
        layer, head = self.source[0]
        one_delta = torch.randn(len(rows), 1, self.spec["width"], device=self.device) * 0.05
        summed = self.model.run_with_hooks(
            tokens, fwd_hooks=result_hooks([(layer, head)], one_delta, pos)
        )[:, -1:]
        self.model.set_use_attn_result(True)

        def add_result(value, hook):
            del hook
            v = value.clone()
            v[torch.arange(len(v), device=v.device), pos, head] += one_delta[:, 0]
            return v

        individual = self.model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_result", add_result)]
        )[:, -1:]
        self.model.set_use_attn_result(False)
        errors.append(float((summed - individual).abs().max()))
        # Expanded ablation versus singleton full forward, both baselines, late/early heads.
        for bi in range(2):
            for layer, head in [(self.layers[0], 0), (self.layers[-1], self.heads - 1)]:
                replacement = (
                    torch.zeros_like(bank[0, 0])
                    if bi == 0
                    else bank[self.layers.index(layer), head]
                )

                def rep(value, hook, *, replacement=replacement, head=head):
                    del hook
                    v = value.clone()
                    v[:, -1, head] = replacement.to(v.device)
                    return v

                y = outcomes(
                    self.model.run_with_hooks(
                        tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", rep)]
                    ),
                    rows,
                )
                measured = baseline[:, :2] - y[:, :2].cpu().numpy()
                errors.append(
                    float(
                        np.max(
                            np.abs(
                                raw[bi, :, self.layers.index(layer) * self.heads + head] - measured
                            )
                        )
                    )
                )
        if max(errors) > self.config["numerical_tolerance"]:
            raise RuntimeError(f"Numerical validation failed: {errors}")
        return {
            "max_error": max(errors),
            "errors": errors,
            "all_measured_heads": len(self.layers) * self.heads,
            "scope": "Full/suffix, nested edits, expanded head ablation, whole-boundary identity",
        }
