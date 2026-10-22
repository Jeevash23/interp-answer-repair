"""Frozen-sign partitions and fresh Medium component-clamp evaluation."""

import argparse
import time

import numpy as np
import torch
from common003 import (
    HERE,
    OLD,
    ROOT,
    SEED,
    Engine003,
    batches,
    delta,
    digest,
    freeze,
    hooks,
    load_arrays,
    natural,
    now,
    read,
    save_arrays,
    verify_freeze,
    write,
)


def partition():
    target = HERE / "data/medium-selection.json"
    assert not target.exists()
    old = OLD / "results/gpt2-medium"
    selected = read(old / "fit/selection.json")["groups"]["wording"]["selected"]
    cfg = read(ROOT / "configs/study.json")["models"]["gpt2-medium"]
    clean = load_arrays(old / "fit/head_wording.npz")["raw"]
    base = load_arrays(old / "fit/natural.npz")["corrupt_raw"]
    groups = {"A": [], "B": []}
    signs = {}
    for layer, h in selected:
        i = cfg["layers"].index(layer) * cfg["heads"] + h
        score = (clean[:, :, i, 0] - base[:, :, i, 0]).mean(1)
        assert np.all(score > 0) or np.all(score < 0)
        groups["A" if np.all(score > 0) else "B"].append([layer, h])
        signs[f"{layer}.{h}"] = score.tolist()
    assert len(groups["A"]) == len(groups["B"]) == 2
    rng = np.random.default_rng(SEED)
    for label in ["A", "B"]:
        for draw in range(3):
            control = []
            for layer in sorted({l for l, h in groups[label]}):
                count = sum(l == layer for l, h in groups[label])
                available = [h for h in range(16) if [layer, h] not in selected]
                control += [[layer, int(h)] for h in rng.choice(available, count, replace=False)]
            groups[f"{label}_control{draw}"] = control
    write(
        target,
        {
            "utc": now(),
            "groups": groups,
            "union": selected,
            "discovery_signs": signs,
            "old_selection_sha256": digest(old / "fit/selection.json"),
            "old_natural_sha256": digest(old / "fit/natural.npz"),
            "old_wording_sha256": digest(old / "fit/head_wording.npz"),
        },
    )
    paths = [
        HERE / "code/medium003.py",
        HERE / "code/analyze_medium003.py",
        HERE / "code/common003.py",
        HERE / "data/input-freeze.json",
        HERE / "data/medium-selection.json",
        HERE / "results/gpt2-medium/technical.json",
    ]
    paths += [ROOT / name for name in read(ROOT / "data/source-freeze.json")["files"]]
    paths += [OLD / "code/assay.py", OLD / "code/design.py"]
    freeze("medium-freeze.json", paths)
    print("Frozen Medium partition", groups, flush=True)


def variants(selection):
    groups = selection["groups"]
    out = [("A", "z", groups["A"]), ("B", "z", groups["B"]), ("union", "z", selection["union"])]
    out += [(f"head{i}", "z", [h]) for i, h in enumerate(selection["union"])]
    out += [
        (f"{g}_control{d}", "z", groups[f"{g}_control{d}"]) for g in ["A", "B"] for d in range(3)
    ]
    out += [(f"{g}_{k}", k, groups[g]) for g in ["A", "B"] for k in ["q", "k", "v"]]
    assert len(out) == 19
    return out


def run(device):
    verify_freeze("input-freeze.json")
    verify_freeze("medium-freeze.json")
    for name, sha in read(HERE / "data/checkpoints.json").items():
        assert digest(ROOT.parent / name) == sha
    rows = [r for r in read(HERE / "data/gpt2-medium-candidates.json") if r["split"] == "medium"]
    selection = read(HERE / "data/medium-selection.json")
    modes = variants(selection)
    out = HERE / "results/medium-cancellation"
    out.mkdir(exist_ok=True)
    record_path = out / "run.json"
    record = (
        read(record_path)
        if record_path.exists()
        else {
            "started": now(),
            "status": "running",
            "hashes": {},
            "freeze": digest(HERE / "data/medium-freeze.json"),
            "record_ids": [r["id"] for r in rows],
            "variants": [v[0] for v in modes],
        }
    )
    assert record["freeze"] == digest(HERE / "data/medium-freeze.json")
    for name, sha in record["hashes"].items():
        assert digest(out / name) == sha
    if record["status"] == "completed":
        return
    write(record_path, record)
    start = time.monotonic()
    with torch.inference_mode():
        engine = Engine003("gpt2-medium", device)
        if "natural.npz" in record["hashes"]:
            nat = load_arrays(out / "natural.npz")
        else:
            assert not (out / "natural.npz").exists()
            nat = natural(engine, rows)
            save_arrays(out / "natural.npz", **nat)
            record["hashes"]["natural.npz"] = digest(out / "natural.npz")
            write(record_path, record)
        for factor in ["wording", "order"]:
            name = factor + ".npz"
            if name in record["hashes"]:
                continue
            assert not (out / name).exists()
            d = delta(nat, rows, factor)
            values = {
                "base": np.empty((len(rows), 6), np.float32),
                "edited": np.empty((len(rows), 6), np.float32),
                "clamped": np.empty((19, len(rows), 6), np.float32),
                "clamped_base": np.empty((19, len(rows), 6), np.float32),
            }
            for bi, ids in enumerate(batches(rows)):
                rr = [dict(rows[i], token_ids=rows[i]["corrupt_tokens"]) for i in ids]
                b, cache = engine.forward(rr)
                hs = hooks(engine, d[ids], rr)
                e, _ = engine.forward(rr, hs)
                values["base"][ids] = b.cpu().numpy()
                values["edited"][ids] = e.cpu().numpy()
                for vi, (_, component, heads) in enumerate(modes):
                    clamp = engine.clamp(cache, heads, component)
                    b0, _ = engine.forward(rr, clamp)
                    b1, cc = engine.forward(rr, hs + clamp)
                    torch.testing.assert_close(b0, b, atol=0.0005, rtol=0)
                    for l, h in heads:
                        key = f"blocks.{l}.attn.hook_{component}"
                        pos = -1 if component in ["z", "q"] else slice(None)
                        torch.testing.assert_close(
                            cc[key][:, pos, h], cache[key][:, pos, h], atol=0, rtol=0
                        )
                    values["clamped"][vi, ids] = b1.cpu().numpy()
                    values["clamped_base"][vi, ids] = b0.cpu().numpy()
                if (bi + 1) % 16 == 0:
                    print("Medium cancellation", factor, bi + 1, "batches", flush=True)
            np.testing.assert_allclose(values["base"], nat["corrupt_behavior"], atol=0.0005, rtol=0)
            save_arrays(out / name, **values)
            record["hashes"][name] = digest(out / name)
            write(record_path, record)
        record.update(
            status="completed", finished=now(), elapsed_last_segment=time.monotonic() - start
        )
        write(record_path, record)
        print("MEDIUM CANCELLATION COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["freeze", "test"])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    partition() if a.stage == "freeze" else run(a.device)
