"""Upstream transport and explicit downstream component-clamp experiments."""

import os
from activation_transfer.paths import verify_frozen_file

import numpy as np
import torch
from design import HERE, SEED

from transport_bridge.common import digest, read
from transport_bridge.engine import Engine, random_matched, result_hooks, state_hook
from transport_bridge.statistics import fit_reference
from transport_bridge.study import batches


def save_arrays(path, **arrays):
    assert all(np.isfinite(a).all() for a in arrays.values())
    temp = path.with_suffix(".partial")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def load_arrays(path):
    with np.load(path) as data:
        return dict(data)


def check_freeze(filename):
    frozen = read(HERE / "data" / filename)
    for name, sha in frozen["files"].items():
        verify_frozen_file(HERE / name, sha)


class CausalEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.names += [
            f"blocks.{l}.attn.hook_{kind}"
            for l in self.layers
            for kind in ["q", "k", "v", "pattern"]
        ]

    def clamp(self, cache, group, component="z", replacement=None):
        hooks = []
        for layer in sorted({h[0] for h in group}):
            heads = [h for l, h in group if l == layer]
            name = f"blocks.{layer}.attn.hook_{component}"
            source = cache[name].detach().clone()

            def apply(value, hook, *, source=source, heads=heads, layer=layer):
                del hook
                out = value.clone()
                for h in heads:
                    if replacement is not None:
                        out[:, -1, h] = replacement[self.layers.index(layer), h].to(out.device)
                    elif component in ["z", "q"]:
                        out[:, -1, h] = source[:, -1, h]
                    else:
                        out[:, :, h] = source[:, :, h]
                return out

            hooks.append((name, apply))
        return hooks

    def attention(self, cache, group, rows):
        ix = torch.arange(len(rows), device=self.device)
        coordinates = []
        for layer, head in group:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
            coordinates.append(
                torch.stack(
                    [
                        pattern[
                            ix, head, -1, torch.tensor([r[key] for r in rows], device=self.device)
                        ]
                        for key in ["io_position", "subject_position", "s2"]
                    ],
                    -1,
                )
            )
        return torch.stack(coordinates, 1).cpu().numpy()


def collect(engine, rows, output, bank=None):
    arrays = {}
    for variant in ["clean", "corrupt"]:
        states, zs, results, behaviors = [], [], [], []
        for ix in batches(len(rows), engine.config["batch_size"]):
            rr = [
                dict(
                    rows[i],
                    token_ids=rows[i]["token_ids" if variant == "clean" else "corrupt_tokens"],
                )
                for i in ix
            ]
            behavior, cache = engine.forward(rr)
            s, z, a = engine.extract(rr, cache)
            states.append(s.numpy())
            zs.append(z.numpy())
            results.append(a.numpy())
            behaviors.append(behavior.cpu().numpy())
        arrays[variant + "_state"] = np.concatenate(states)
        arrays[variant + "_z"] = np.concatenate(zs)
        arrays[variant + "_head_results"] = np.concatenate(results)
        arrays[variant + "_behavior"] = np.concatenate(behaviors)
        if variant == "clean" and bank is None:
            bank = torch.tensor(arrays["clean_z"].mean(0))
    for variant in ["clean", "corrupt"]:
        raw = []
        for bi, ix in enumerate(batches(len(rows), engine.config["batch_size"])):
            rr = [
                dict(
                    rows[i],
                    token_ids=rows[i]["token_ids" if variant == "clean" else "corrupt_tokens"],
                )
                for i in ix
            ]
            effects, behavior = engine.signatures(rr, bank)
            np.testing.assert_allclose(
                behavior, arrays[variant + "_behavior"][ix], atol=5e-4, rtol=0
            )
            raw.append(effects)
            if (bi + 1) % 16 == 0:
                print(f"{output.parent.name}: {variant} signatures {bi + 1} batches", flush=True)
        arrays[variant + "_raw"] = np.concatenate(raw, axis=1)
    arrays["mean_bank"] = bank.numpy()
    save_arrays(output, **arrays)
    return arrays


def delta_for(spec, rows, natural):
    index = {r["id"]: i for i, r in enumerate(rows)}
    donors = np.array([index[r["donors"][spec["donor"]]] for r in rows])
    if spec["kind"] == "full":
        delta = torch.tensor(natural["clean_state"][donors] - natural["corrupt_state"])
    else:
        delta = torch.tensor(
            natural["clean_head_results"][donors] - natural["corrupt_head_results"][donors]
        )
        if spec["kind"] == "random":
            delta = random_matched(
                delta, [r["id"] for r in rows], "bridge002-wording", spec["draw"]
            )
    return delta * spec["dose"], donors


def edit_hooks(engine, spec, delta, rows):
    pos = torch.tensor([r["s2"] for r in rows], device=engine.device)
    if spec["kind"] == "full":
        return [state_hook(engine.site, pos, delta.to(engine.device))]
    return result_hooks(engine.source, delta.to(engine.device), pos)


def measure(engine, rows, natural, spec, output):
    delta, donors = delta_for(spec, rows, natural)
    raw, behaviors = [], []
    bank = torch.tensor(natural["mean_bank"])
    for ix in batches(len(rows), engine.config["batch_size"]):
        rr = [dict(rows[i], token_ids=rows[i]["corrupt_tokens"]) for i in ix]
        effects, behavior = engine.signatures(rr, bank, edit_hooks(engine, spec, delta[ix], rr))
        raw.append(effects)
        behaviors.append(behavior)
    result = {
        "raw": np.concatenate(raw, axis=1),
        "behavior": np.concatenate(behaviors),
        "delta_norm": delta.flatten(1).norm(dim=-1).numpy(),
        "donors": donors,
    }
    save_arrays(output, **result)
    return result


def select_groups(engine, rows, natural, edits):
    reference = fit_reference(natural["clean_raw"], rows)
    scale = np.asarray(reference["scale"])
    clean = natural["clean_raw"][..., 0]
    groups, scores = {}, {}
    for f in ["wording", "order", "self"]:
        if f == "self":
            score = clean.mean((0, 1))
        else:
            score = (
                np.abs(edits[f]["raw"][..., 0] - natural["corrupt_raw"][..., 0]) / scale[:, None]
            ).mean((0, 1))
        ordered = sorted(range(len(score)), key=lambda i: (-float(score[i]), i))
        selected = ordered[:4]
        heads = [[engine.layers[i // engine.heads], i % engine.heads] for i in selected]
        rng = np.random.default_rng(SEED + {"wording": 1, "order": 2, "self": 3}[f])
        controls = []
        for _ in range(3):
            group = []
            for layer in sorted({h[0] for h in heads}):
                count = sum(l == layer for l, _ in heads)
                available = [h for h in range(engine.heads) if [layer, h] not in heads]
                group += [[layer, int(h)] for h in rng.choice(available, size=count, replace=False)]
            controls.append(group)
        total = float((edits[f]["behavior"][:, 0] - natural["corrupt_behavior"][:, 0]).mean())
        groups[f] = {
            "selected": heads,
            "controls": controls,
            "indices": selected,
            "orientation": 1 if total >= 0 else -1,
            "discovery_recovery": total,
        }
        scores[f] = score.tolist()
    return {"reference": reference, "groups": groups, "selection_scores": scores}


def trace(engine, rows, natural, spec, selected, output):
    f = spec["donor"]
    group = selected["groups"][f]
    delta, _ = delta_for(spec, rows, natural)
    variants = [("z", group["selected"])] + [("z", g) for g in group["controls"]]
    variants += [(kind, group["selected"]) for kind in ["q", "k", "v"]]
    bank = torch.tensor(natural["mean_bank"])
    all_results = []
    for bi, ix in enumerate(batches(len(rows), engine.config["batch_size"])):
        rr = [dict(rows[i], token_ids=rows[i]["corrupt_tokens"]) for i in ix]
        base, cache = engine.forward(rr)
        hooks = edit_hooks(engine, spec, delta[ix], rr)
        edited, edited_cache = engine.forward(rr, hooks)
        clamped, noedit = [], []
        for component, heads in variants:
            clamp = engine.clamp(cache, heads, component)
            b0, _ = engine.forward(rr, clamp)
            b1, clamped_cache = engine.forward(rr, hooks + clamp)
            np.testing.assert_allclose(b0.cpu(), base.cpu(), atol=5e-4, rtol=0)
            for layer, h in heads:
                name = f"blocks.{layer}.attn.hook_{component}"
                pos = -1 if component in ["q", "z"] else slice(None)
                torch.testing.assert_close(
                    clamped_cache[name][:, pos, h], cache[name][:, pos, h], atol=0, rtol=0
                )
            noedit.append(b0.cpu().numpy())
            clamped.append(b1.cpu().numpy())
        joint = []
        for replacement in [torch.zeros_like(bank), bank]:
            ab = engine.clamp(cache, group["selected"], replacement=replacement)
            c, _ = engine.forward(rr, ab)
            e, _ = engine.forward(rr, hooks + ab)
            joint.append(np.stack([c.cpu().numpy(), e.cpu().numpy()]))
        all_results.append(
            {
                "base": base.cpu().numpy(),
                "edited": edited.cpu().numpy(),
                "clamped_base": np.stack(noedit),
                "clamped": np.stack(clamped),
                "joint": np.stack(joint),
                "attention": np.stack(
                    [
                        engine.attention(cache, group["selected"], rr),
                        engine.attention(edited_cache, group["selected"], rr),
                    ]
                ),
            }
        )
        if (bi + 1) % 16 == 0:
            print(f"{f}: component clamps {bi + 1} batches", flush=True)
    arrays = {
        k: np.concatenate(
            [x[k] for x in all_results],
            axis={
                "base": 0,
                "edited": 0,
                "clamped_base": 1,
                "clamped": 1,
                "joint": 2,
                "attention": 1,
            }[k],
        )
        for k in all_results[0]
    }
    save_arrays(output, **arrays)
    return arrays


def technical(engine, rows):
    result = engine.validate(rows)
    base, cache = engine.forward(rows)
    group = [[engine.layers[0], 0], [engine.layers[-1], 1]]
    errors = []
    for component in ["q", "k", "v", "z"]:
        value, _ = engine.forward(rows, engine.clamp(cache, group, component))
        errors.append(float((base - value).abs().max()))
        # A deliberately shifted cached source checks that these hooks are active.
        shifted = {k: v for k, v in cache.items()}
        for layer, _ in group:
            name = f"blocks.{layer}.attn.hook_{component}"
            shifted[name] = cache[name] + 0.05
        _, actual = engine.forward(rows, engine.clamp(shifted, group, component))
        for layer, h in group:
            name = f"blocks.{layer}.attn.hook_{component}"
            pos = -1 if component in ["z", "q"] else slice(None)
            torch.testing.assert_close(
                actual[name][:, pos, h], shifted[name][:, pos, h], atol=0, rtol=0
            )
    result["component_identity_errors"] = errors
    assert max(errors) <= 0.0005
    return result
