"""Validated CPU suffix evaluation of the same frozen head-node estimand."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from activation_transfer.paths import recorded_path
import torch

from causal_circuit_routing.interventions import head_hook, outcomes
from causal_circuit_routing.io import ROOT

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"


def batches(records, size):
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[len(record["token_ids"])].append(i)
    return [ix[k : k + size] for _, ix in sorted(groups.items()) for k in range(0, len(ix), size)]


def state_hook(layer, position, delta, kind="residual"):
    name = (
        f"blocks.{layer}.hook_resid_pre" if kind == "residual" else f"blocks.{layer}.hook_mlp_out"
    )

    def add(value, hook):
        del hook
        output = value.clone()
        output[torch.arange(len(value)), position] += delta
        return output

    return name, add


class Engine:
    def __init__(self, config):
        from transformer_lens.model_bridge import TransformerBridge

        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        self.config = config
        self.model = TransformerBridge.boot_transformers(
            str(recorded_path(ROOT / config["model_path"])),
            device=torch.device("cpu"),
            dtype=torch.float32,
            trust_remote_code=False,
        )
        self.model.enable_compatibility_mode()
        self.model.set_use_attn_result(False)
        self.model.eval()
        self.cache_names = [f"blocks.{l}.hook_resid_pre" for l in range(12)] + [
            f"blocks.{l}.attn.hook_z" for l in range(6, 12)
        ]
        self.sites = [
            (l, p) for l in config["candidate_layers"] for p in config["candidate_positions"]
        ]

    def suffix(self, hidden, start, hooks=()):
        n = hidden.shape[1]
        mask = torch.full((1, 1, n, n), torch.finfo(hidden.dtype).min).triu(1)
        original = self.model.original_model
        with self.model.hooks(fwd_hooks=list(hooks)):
            for block in original.transformer.h[start:]:
                hidden = block(hidden, attention_mask=mask, use_cache=False)
            return original.lm_head(original.transformer.ln_f(hidden[:, -1:, :]))

    def forward_cache(self, rows, hooks=()):
        tokens = torch.tensor([r["token_ids"] for r in rows])
        with self.model.hooks(fwd_hooks=list(hooks)):
            logits, cache = self.model.run_with_cache(tokens, names_filter=self.cache_names)
        metrics = outcomes(logits, rows)
        last = logits[:, -1]
        alt = torch.tensor([r.get("alternative_answer_token_id", r["io_token_id"]) for r in rows])
        io = torch.tensor([r["io_token_id"] for r in rows])
        metrics["alternative_minus_io_logit"] = (
            last[torch.arange(len(rows)), alt] - last[torch.arange(len(rows)), io]
        )
        metrics["alternative_probability"] = last.softmax(-1)[torch.arange(len(rows)), alt]
        features = torch.stack(
            [
                cache[f"blocks.{l}.hook_resid_pre"][
                    torch.arange(len(rows)), torch.tensor([r["positions"][p] for r in rows])
                ]
                for l, p in self.sites
            ],
            dim=1,
        )
        z = torch.stack([cache[f"blocks.{l}.attn.hook_z"][:, -1] for l in range(6, 12)], 1)
        return metrics, features, z, cache

    def signatures(self, rows, corrupt_z, hooks=()):
        base, features, z, cache = self.forward_cache(rows, hooks)
        n = len(rows)
        raw = np.empty((2, n, 72), dtype=np.float32)
        probability = np.empty_like(raw)
        positions = torch.full((n,), len(rows[0]["token_ids"]) - 1)
        for li, layer in enumerate(range(6, 12)):
            hidden = cache[f"blocks.{layer}.hook_resid_pre"]
            for head in range(12):
                ci = li * 12 + head
                for bi in range(2):
                    replace = torch.zeros(n, 64) if bi == 0 else corrupt_z[:, li, head]
                    hk, count = head_hook(layer, head, replace, positions)
                    patched = outcomes(self.suffix(hidden.clone(), layer, [hk]), rows)
                    assert count["count"] == 1
                    raw[bi, :, ci] = (
                        base["logit_difference"] - patched["logit_difference"]
                    ).numpy()
                    probability[bi, :, ci] = (
                        base["probability_difference"] - patched["probability_difference"]
                    ).numpy()
        return raw, probability, base, features, z

    def validate(self, rows):
        from causal_circuit_routing.io import load_records

        # Reuse already-observed pilot examples for implementation checks, not study test data.
        del rows
        old = load_records()[:4]
        rows = [{**r, "token_ids": r["clean_token_ids"]} for r in old]
        tokens = torch.tensor([r["token_ids"] for r in rows])
        baseline, _, z, cache = self.forward_cache(rows)
        direct = self.model(tokens)[:, -1:]
        errors = []
        for layer in (0, 2, 4, 6, 8, 11):
            out = self.suffix(cache[f"blocks.{layer}.hook_resid_pre"].clone(), layer)
            errors.append(float((out - direct).abs().max()))
        positions = torch.full((len(rows),), tokens.shape[1] - 1)
        identity = []
        nested = []
        delta = torch.randn(len(rows), 768) * 0.1
        for site in (2, 4, 6):
            for pos in (positions, torch.tensor([r["positions"]["subject_second"] for r in rows])):
                edit = state_hook(site, pos, delta)
                _, _, _, edited_cache = self.forward_cache(rows, [edit])
                for layer, head in ((6, 3), (9, 9), (11, 0)):
                    hk, _ = head_hook(layer, head, torch.zeros(len(rows), 64), positions)
                    slow = self.model.run_with_hooks(tokens, fwd_hooks=[edit, hk])[:, -1:]
                    fast = self.suffix(
                        edited_cache[f"blocks.{layer}.hook_resid_pre"].clone(), layer, [hk]
                    )
                    nested.append(float((slow - fast).abs().max()))
        for layer in range(6, 12):
            for head in range(12):
                hk, _ = head_hook(layer, head, z[:, layer - 6, head], positions)
                out = outcomes(
                    self.suffix(cache[f"blocks.{layer}.hook_resid_pre"].clone(), layer, [hk]), rows
                )
                identity.append(
                    float((out["logit_difference"] - baseline["logit_difference"]).abs().max())
                )
        report = {
            "unmodified_suffix_max_error": max(errors),
            "nested_full_vs_suffix_max_error": max(nested),
            "all72_identity_max_error": max(identity),
            "nested_checks": len(nested),
        }
        if max(max(errors), max(nested), max(identity)) > self.config["numerical_tolerance"]:
            raise RuntimeError(f"Suffix validation failed: {report}")
        return report
