"""New-study utilities; the completed bridge engines remain unchanged."""

import os
from activation_transfer.paths import verify_frozen_file
import sys
from pathlib import Path

import numpy as np
import torch

from transport_bridge.common import ROOT, digest, now, read, write

HERE = Path(__file__).resolve().parents[1]
OLD = ROOT / "studies/bridge-002"
sys.path.insert(0, str(OLD / "code"))
import assay as _assay
from assay import CausalEngine, edit_hooks

load_arrays = _assay.load_arrays
save_arrays = _assay.save_arrays

SEED = 2026090803
MODELS = ["gpt2-medium", "pythia-410m-deduped"]


def config():
    c = read(ROOT / "configs/study.json")
    selected = read(
        ROOT / "studies/transport/results/experiment-004/localization/selected-heads.json"
    )
    source = sorted(
        {tuple(h) for v in selected["conditions"].values() for h in v["s2"]["selected_heads"]}
    )
    c["models"][MODELS[1]] = {
        "site": 12,
        "layers": list(range(12, 24)),
        "heads": 16,
        "width": 1024,
        "source_heads": source,
    }
    c["batch_size"] = 4
    return c


def batches(rows, size=4):
    lengths = sorted({len(r["token_ids"]) for r in rows})
    for length in lengths:
        ids = [i for i, r in enumerate(rows) if len(r["token_ids"]) == length]
        for start in range(0, len(ids), size):
            yield np.array(ids[start : start + size])


def verify_freeze(name):
    for p, sha in read(HERE / "data" / name)["files"].items():
        verify_frozen_file(HERE / p, sha)


def freeze(name, paths):
    target = HERE / "data" / name
    assert not target.exists()
    write(
        target,
        {"utc": now(), "files": {os.path.relpath(p, HERE): digest(p) for p in sorted(set(paths))}},
    )


class Engine003(CausalEngine):
    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        c = config()
        if model_name == "gpt2-medium":
            super().__init__(model_name, c, device)
            return
        torch.set_num_threads(c["threads"])
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.config = c
        self.spec = c["models"][model_name]
        self.site, self.layers, self.heads = (
            self.spec["site"],
            self.spec["layers"],
            self.spec["heads"],
        )
        self.source = [tuple(x) for x in self.spec["source_heads"]]
        sys.path.insert(0, str(ROOT / "studies/transport/src"))
        from circuit_transport.pythia import load_pythia_for_hooks

        self.model, self.raw_model, _ = load_pythia_for_hooks()
        self.raw_model.set_attn_implementation("eager")
        from neox_numerics import align

        align(self.model, self.raw_model)
        self.model.set_use_attn_result(False)
        self.model.to(self.device)
        self.names = [f"blocks.{l}.hook_resid_pre" for l in self.layers]
        self.names += [
            f"blocks.{l}.attn.hook_z"
            for l in sorted(set(self.layers) | {h[0] for h in self.source})
        ]
        self.names += [
            f"blocks.{l}.attn.hook_{k}" for l in self.layers for k in ["q", "k", "v", "pattern"]
        ]
        self.weights = self.model.W_O.detach()

    def suffix(self, hidden, start, hooks=()):
        if self.model_name == "gpt2-medium":
            return super().suffix(hidden, start, hooks)
        with self.model.hooks(fwd_hooks=list(hooks)):
            return self.model(hidden, start_at_layer=start, return_type="logits")[:, -1:]

    def raw_check(self, rows):
        if self.model_name != "pythia-410m-deduped":
            return None
        tokens = torch.tensor([r["token_ids"] for r in rows], device=self.device)
        self.raw_model.to(self.device)
        from neox_numerics import align_rotary

        align_rotary(self.model, self.raw_model, self.device)
        reference = self.raw_model(tokens).logits[:, -1].detach().cpu()
        converted = self.model(tokens)[:, -1].detach().cpu()
        error = float((reference - converted).abs().max())
        assert error < 0.0005, error
        del self.raw_model
        return error


def natural(engine, rows):
    result = {}
    for v in ["clean", "corrupt"]:
        values, states, head_results, zs = [], [], [], []
        order = []
        for ids in batches(rows):
            rr = [
                dict(rows[i], token_ids=rows[i]["token_ids" if v == "clean" else "corrupt_tokens"])
                for i in ids
            ]
            b, cache = engine.forward(rr)
            s, z, h = engine.extract(rr, cache)
            values.append(b.cpu().numpy())
            states.append(s.numpy())
            head_results.append(h.numpy())
            zs.append(z.numpy())
            order.extend(ids)
        sort = np.argsort(order)
        for k, a in [
            ("behavior", values),
            ("state", states),
            ("head_results", head_results),
            ("z", zs),
        ]:
            result[v + "_" + k] = np.concatenate(a)[sort]
    return result


def donor_indices(rows, factor):
    index = {
        (r["template"], r["pair"], r["role"], r["wording"], r["order"]): i
        for i, r in enumerate(rows)
    }
    return np.array(
        [
            index[
                (
                    r["template"],
                    r["pair"],
                    r["role"] ^ int("role" in factor),
                    r["wording"] ^ int("wording" in factor),
                    r["order"] ^ int("order" in factor),
                )
            ]
            for r in rows
        ]
    )


def delta(nat, rows, factor):
    ids = donor_indices(rows, factor)
    return torch.tensor(nat["clean_head_results"][ids] - nat["corrupt_head_results"][ids])


def hooks(engine, d, rows):
    return edit_hooks(engine, {"kind": "head"}, d, rows)


def signature(engine, rows, bank, d=None):
    raw = np.empty((2, len(rows), len(engine.layers) * engine.heads, 2), np.float32)
    behavior = np.empty((len(rows), 6), np.float32)
    for n, ids in enumerate(batches(rows)):
        rr = [
            dict(
                rows[i],
                token_ids=rows[i]["corrupt_tokens"] if d is not None else rows[i]["token_ids"],
            )
            for i in ids
        ]
        a, b = engine.signatures(rr, bank, () if d is None else hooks(engine, d[ids], rr))
        raw[:, ids] = a
        behavior[ids] = b
        if (n + 1) % 16 == 0:
            print("signatures", n + 1, "batches", flush=True)
    return {"raw": raw, "behavior": behavior}
