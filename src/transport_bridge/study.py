"""Fixed intervention inventory and discovery/test execution."""

import time

import numpy as np
import torch

from .common import ROOT, digest, now, read, verify, write
from .engine import Engine, random_matched, result_hooks, state_hook
from .statistics import fit_predictor, fit_reference, prediction_scores


def conditions():
    out = [{"name": "head_self", "kind": "head", "factor": "self", "base": "corrupt"}]
    out += [
        {"name": "head_" + f, "kind": "head", "factor": f, "base": "corrupt"}
        for f in ["wording", "order"]
    ]
    out += [
        {
            "name": f"head_random_{d}",
            "kind": "head_random",
            "factor": "wording",
            "base": "corrupt",
            "draw": d,
        }
        for d in range(3)
    ]
    out += [{"name": "full_self_corrupt", "kind": "full", "factor": "self", "base": "corrupt"}]
    for f in ["wording", "order"]:
        for base in ["corrupt", "clean"]:
            for kind in ["full", "rank1", "changed"]:
                out.append({"name": f"{kind}_{f}_{base}", "kind": kind, "factor": f, "base": base})
        out.append(
            {"name": f"matched_{f}_corrupt", "kind": "matched", "factor": f, "base": "corrupt"}
        )
        out += [
            {"name": f"random_{f}_{d}", "kind": "random", "factor": f, "base": "corrupt", "draw": d}
            for d in range(3)
        ]
    assert len(out) == 27
    return out


def batches(n, size):
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def verify_model(name):
    freeze = read(ROOT / "data/source-freeze.json")
    for path, sha in freeze["models"][name].items():
        if digest(ROOT.parent / path) != sha:
            raise RuntimeError("Cached model differs from freeze: " + path)


def collect(engine, rows, library, output, bank=None, include_corrupt=False):
    n = len(rows)
    arrays = {}
    groups = batches(n, engine.config["batch_size"])
    for variant in ["clean", "corrupt", "changed"]:
        states, zs, results, behaviors = [], [], [], []
        for ix in groups:
            prompts = [library[rows[i][variant]] for i in ix]
            # Keep the recipient's scoring labels, including alternative name.
            scoring = [
                {**rows[i], "token_ids": p["token_ids"]} for i, p in zip(ix, prompts, strict=True)
            ]
            behavior, cache = engine.forward(scoring)
            s, z, a = engine.extract(scoring, cache)
            states.append(s.numpy())
            zs.append(z.numpy())
            results.append(a.numpy())
            behaviors.append(behavior.cpu().numpy())
        arrays[variant + "_state"] = np.concatenate(states)
        arrays[variant + "_behavior"] = np.concatenate(behaviors)
        if variant != "changed":
            arrays[variant + "_head_results"] = np.concatenate(results)
        if variant == "clean" and bank is None:
            bank = torch.tensor(np.concatenate(zs).mean(0))
    for variant in ["clean", "corrupt"] if include_corrupt else ["clean"]:
        raw = []
        start = time.monotonic()
        for bi, ix in enumerate(groups):
            rr = [{**rows[i], "token_ids": library[rows[i][variant]]["token_ids"]} for i in ix]
            effects, behavior = engine.signatures(rr, bank)
            np.testing.assert_allclose(
                behavior, arrays[variant + "_behavior"][ix], atol=5e-4, rtol=0
            )
            raw.append(effects)
            if (bi + 1) % 8 == 0 or bi + 1 == len(groups):
                print(
                    f"{output.name} {variant} signatures {bi + 1}/{len(groups)}; {time.monotonic() - start:.1f}s",
                    flush=True,
                )
        arrays[variant + "_raw"] = np.concatenate(raw, axis=1)
    arrays["mean_bank"] = bank.numpy()
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(output, **arrays)
    return arrays


def fit(model_name, device):
    config = read(ROOT / "configs/study.json")
    verify(read(ROOT / "data/source-freeze.json")["files"])
    verify_model(model_name)
    output = ROOT / "results" / model_name / "fit"
    output.mkdir(parents=True, exist_ok=False)
    rows = read(ROOT / "data/recipients.json")
    library = read(ROOT / "data/prompts.json")
    info = {"status": "running", "started": now(), "model": model_name, "device": device}
    write(output / "run.json", info)
    with torch.inference_mode():
        engine = Engine(model_name, config, device)
        discovery_rows = [r for r in rows if r["split"] == "discovery"]
        dev_rows = [r for r in rows if r["split"] == "development"]
        discovery = collect(engine, discovery_rows, library, output / "discovery.npz")
        development = collect(
            engine,
            dev_rows,
            library,
            output / "development.npz",
            torch.tensor(discovery["mean_bank"]),
        )
    reference = fit_reference(discovery["clean_raw"], discovery_rows)
    predictors = {
        f: fit_predictor(
            discovery["clean_state"], discovery["clean_raw"], discovery_rows, reference, f
        )
        for f in ["wording", "order"]
    }
    scores = {
        f: prediction_scores(
            development["clean_state"],
            development["clean_raw"],
            dev_rows,
            reference,
            predictors[f],
            f,
        )
        for f in predictors
    }
    fitted = {
        "reference": reference,
        "predictors": predictors,
        "development_prediction": scores,
        "site": config["models"][model_name]["site"],
        "frozen_before_test": True,
    }
    write(output / "selection.json", fitted)
    files = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            output / "selection.json",
            output / "discovery.npz",
            output / "development.npz",
            ROOT / "scripts/analyze.py",
            ROOT / "src/transport_bridge/statistics.py",
        ]
    }
    write(
        output / "test-freeze.json",
        {"utc": now(), "files": files, "source_freeze": digest(ROOT / "data/source-freeze.json")},
    )
    info.update(status="completed", completed=now(), development_prediction=scores)
    write(output / "run.json", info)
    print("FIT COMPLETE", scores, flush=True)


def prepare_delta(spec, rows, arrays, selection):
    index = {r["id"]: i for i, r in enumerate(rows)}
    factor, kind, base = spec["factor"], spec["kind"], spec["base"]
    donors = np.array(
        [i if factor == "self" else index[r["donors"][factor]] for i, r in enumerate(rows)]
    )
    if kind.startswith("head"):
        delta = torch.tensor(
            arrays["clean_head_results"][donors] - arrays["corrupt_head_results"][donors]
        )
        if kind == "head_random":
            delta = random_matched(delta, [r["id"] for r in rows], "head_wording", spec["draw"])
    else:
        donor_state = (
            arrays["changed_state"][donors] if kind == "changed" else arrays["clean_state"][donors]
        )
        delta = torch.tensor(donor_state - arrays[base + "_state"])
        if kind in ["rank1", "matched", "random"]:
            u = torch.tensor(selection["predictors"][factor]["direction"], dtype=delta.dtype)
            rank = (delta @ u)[:, None] * u
            if kind == "rank1":
                delta = rank
            elif kind == "matched":
                delta = (
                    delta
                    * rank.norm(dim=-1, keepdim=True)
                    / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                )
            else:
                delta = random_matched(rank, [r["id"] for r in rows], factor, spec["draw"])
    return delta, donors


def test(model_name, device):
    config = read(ROOT / "configs/study.json")
    verify(read(ROOT / "data/source-freeze.json")["files"])
    verify_model(model_name)
    fit_dir = ROOT / "results" / model_name / "fit"
    verify(read(fit_dir / "test-freeze.json")["files"])
    output = ROOT / "results" / model_name / "test"
    output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in read(ROOT / "data/recipients.json") if r["split"] == "test"]
    library = read(ROOT / "data/prompts.json")
    selected = read(fit_dir / "selection.json")
    with np.load(fit_dir / "discovery.npz") as d:
        bank = torch.tensor(d["mean_bank"])
    info = {
        "status": "running",
        "started": now(),
        "model": model_name,
        "device": device,
        "condition_inventory": conditions(),
        "completed": [],
        "record_ids": [r["id"] for r in rows],
        "test_freeze_sha256": digest(fit_dir / "test-freeze.json"),
    }
    write(output / "run.json", info)
    start = time.monotonic()
    with torch.inference_mode():
        engine = Engine(model_name, config, device)
        arrays = collect(engine, rows, library, output / "natural.npz", bank, include_corrupt=True)
        for spec in conditions():
            delta, donors = prepare_delta(spec, rows, arrays, selected)
            raw, behavior = [], []
            begun = time.monotonic()
            for ix in batches(len(rows), config["batch_size"]):
                rr = [
                    {**rows[i], "token_ids": library[rows[i][spec["base"]]]["token_ids"]}
                    for i in ix
                ]
                pos = torch.tensor([r["s2"] for r in rr], device=engine.device)
                hooks = (
                    result_hooks(engine.source, delta[ix].to(engine.device), pos)
                    if spec["kind"].startswith("head")
                    else [state_hook(engine.site, pos, delta[ix].to(engine.device))]
                )
                a, b = engine.signatures(rr, bank, hooks)
                raw.append(a)
                behavior.append(b)
            out = {
                "raw": np.concatenate(raw, axis=1),
                "behavior": np.concatenate(behavior),
                "delta_norm": delta.flatten(1).norm(dim=-1).numpy(),
                "donor_indices": donors,
            }
            assert all(np.isfinite(a).all() for a in out.values())
            np.savez_compressed(output / (spec["name"] + ".npz"), **out)
            info["completed"].append(spec["name"])
            info["elapsed_seconds"] = time.monotonic() - start
            write(output / "run.json", info)
            print(
                f"{model_name}: {spec['name']} {len(info['completed'])}/27; {time.monotonic() - begun:.1f}s; total {info['elapsed_seconds'] / 60:.1f}min",
                flush=True,
            )
    info.update(
        status="completed", finished=now(), hashes={p.name: digest(p) for p in output.glob("*.npz")}
    )
    write(output / "run.json", info)
