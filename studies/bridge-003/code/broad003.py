"""Broader-family discovery, frozen selection, and resumable held-out transfers."""

import argparse
import itertools
import time

import numpy as np
import torch
from common003 import (
    HERE,
    MODELS,
    OLD,
    ROOT,
    Engine003,
    delta,
    digest,
    freeze,
    load_arrays,
    natural,
    now,
    read,
    save_arrays,
    signature,
    verify_freeze,
    write,
)

from transport_bridge.engine import random_matched
from transport_bridge.statistics import axis_coordinates, fit_reference


def rows_for(model, split, families):
    return [
        dict(r, wording=families.index(r["family"]))
        for r in read(HERE / "data" / (model + "-candidates.json"))
        if r["split"] == split and r["family"] in families
    ]


def discovery_freeze():
    verify_freeze("input-freeze.json")
    verify_freeze("screen-freeze.json")
    if (HERE / "data/discovery-freeze.json").exists():
        verify_freeze("discovery-freeze.json")
        return
    assert all((HERE / "results" / m / "screen.json").exists() for m in MODELS)
    paths = [
        HERE / "code/common003.py",
        HERE / "code/neox_numerics.py",
        HERE / "code/broad003.py",
        HERE / "code/analyze_broad003.py",
        HERE / "code/test003.py",
        HERE / "data/input-freeze.json",
        HERE / "data/screen-freeze.json",
        HERE / "docs/DISCOVERY-NOTES.md",
        HERE / "docs/NUMERICAL-CALIBRATION.md",
    ]
    for model in MODELS:
        paths += [
            HERE / "results" / model / p for p in ["screen.json", "screen.npz", "technical.json"]
        ]
    paths += [ROOT / name for name in read(ROOT / "data/source-freeze.json")["files"]]
    paths += [
        OLD / "code/assay.py",
        OLD / "code/design.py",
        ROOT / "studies/transport/src/circuit_transport/pythia.py",
    ]
    freeze("discovery-freeze.json", paths)


def fit(model, device):
    verify_freeze("input-freeze.json")
    verify_freeze("discovery-freeze.json")
    screen = read(HERE / "results" / model / "screen.json")
    families = screen["retained"]
    rows = rows_for(model, "discovery", families)
    out = HERE / "results" / model / "discovery"
    out.mkdir(exist_ok=True)
    if (out / "run.json").exists() and read(out / "run.json")["status"] == "completed":
        return
    record = (
        read(out / "run.json")
        if (out / "run.json").exists()
        else {
            "started": now(),
            "status": "running",
            "hashes": {},
            "record_ids": [r["id"] for r in rows],
            "families": families,
            "freeze": digest(HERE / "data/discovery-freeze.json"),
        }
    )
    for name, sha in record["hashes"].items():
        assert digest(out / name) == sha
    write(out / "run.json", record)
    with torch.inference_mode():
        engine = Engine003(model, device)
        engine.raw_check(rows[:2])
        if "natural.npz" in record["hashes"]:
            nat = load_arrays(out / "natural.npz")
        else:
            assert not (out / "natural.npz").exists()
            nat = natural(engine, rows)
            save_arrays(out / "natural.npz", **nat)
            record["hashes"]["natural.npz"] = digest(out / "natural.npz")
            write(out / "run.json", record)
        if "signature.npz" not in record["hashes"]:
            assert not (out / "signature.npz").exists()
            bank = torch.tensor(nat["clean_z"].mean(0))
            result = signature(engine, rows, bank)
            result["bank"] = bank.numpy()
            save_arrays(out / "signature.npz", **result)
            record["hashes"]["signature.npz"] = digest(out / "signature.npz")
        record.update(status="completed", finished=now())
        write(out / "run.json", record)
    print("DISCOVERY COMPLETE", model, flush=True)


def pair_scores(model):
    screen = read(HERE / "results" / model / "screen.json")
    directory = HERE / "results" / model / "discovery"
    record = read(directory / "run.json")
    assert record["status"] == "completed"
    for name, sha in record["hashes"].items():
        assert digest(directory / name) == sha
    raw = load_arrays(directory / "signature.npz")["raw"]
    rows = rows_for(model, "discovery", screen["retained"])
    out = {}
    for a, b in itertools.combinations(screen["eligible"], 2):
        ids = np.array([i for i, r in enumerate(rows) if r["family"] in [a, b]])
        rr = [dict(rows[i], wording=int(rows[i]["family"] == b)) for i in ids]
        x = raw[:, ids]
        pred = np.empty(x.shape[:2], bool)
        for block in range(4):
            train = np.array([r["template"] != block for r in rr])
            test = ~train
            ref = fit_reference(x[:, train], [r for r, k in zip(rr, train, strict=True) if k])
            pred[:, test] = (
                axis_coordinates(
                    x[:, test], [r for r, k in zip(rr, test, strict=True) if k], ref, "wording"
                )
                > 0
            )
        cells = {}
        for w in range(2):
            for o in range(2):
                mask = np.array([r["wording"] == w and r["order"] == o for r in rr])
                cells[f"w{w}_o{o}"] = (pred[:, mask] == bool(w)).mean(1).tolist()
        worst = min(v for values in cells.values() for v in values)
        out[f"{a},{b}"] = {
            "families": [a, b],
            "cells": cells,
            "worst": worst,
            "qualified": worst >= 0.8,
        }
    return out


def select():
    verify_freeze("discovery-freeze.json")
    verify_freeze("input-freeze.json")
    assert not (HERE / "data/broad-selection.json").exists()
    scores = {m: pair_scores(m) for m in MODELS}
    common = set.intersection(
        *[{k for k, v in s.items() if v["qualified"]} for s in scores.values()]
    )
    joint = sorted(common, key=lambda k: (-min(scores[m][k]["worst"] for m in MODELS), k))
    chosen = {}
    for model in MODELS:
        eligible = sorted(
            [k for k, v in scores[model].items() if v["qualified"]],
            key=lambda k: (-scores[model][k]["worst"], k),
        )
        key = joint[0] if joint else (eligible[0] if eligible else None)
        families = scores[model][key]["families"] if key else [0, 1]
        rows = rows_for(
            model, "discovery", read(HERE / "results" / model / "screen.json")["retained"]
        )
        ids = np.array([i for i, r in enumerate(rows) if r["family"] in families])
        rr = [dict(rows[i], wording=families.index(rows[i]["family"])) for i in ids]
        raw = load_arrays(HERE / "results" / model / "discovery/signature.npz")["raw"][:, ids]
        chosen[model] = {
            "families": families,
            "qualified": key is not None,
            "reference": fit_reference(raw, rr),
        }
    write(
        HERE / "data/broad-selection.json",
        {"utc": now(), "scores": scores, "selected": chosen, "common_pair": bool(joint)},
    )
    paths = [HERE / "data/broad-selection.json", HERE / "data/discovery-freeze.json"]
    for model in MODELS:
        paths += list((HERE / "results" / model / "discovery").glob("*"))
    freeze("broad-test-freeze.json", paths)
    print(
        "BROAD SELECTION",
        {m: {k: v for k, v in d.items() if k != "reference"} for m, d in chosen.items()},
        flush=True,
    )


def test(model, device):
    for f in ["input-freeze.json", "discovery-freeze.json", "broad-test-freeze.json"]:
        verify_freeze(f)
    selection = read(HERE / "data/broad-selection.json")["selected"][model]
    rows = rows_for(model, "test", selection["families"])
    out = HERE / "results" / model / "test"
    out.mkdir(exist_ok=True)
    record = (
        read(out / "run.json")
        if (out / "run.json").exists()
        else {
            "started": now(),
            "status": "running",
            "hashes": {},
            "record_ids": [r["id"] for r in rows],
            "freeze": digest(HERE / "data/broad-test-freeze.json"),
        }
    )
    assert record["freeze"] == digest(HERE / "data/broad-test-freeze.json")
    for name, sha in record["hashes"].items():
        assert digest(out / name) == sha
    if record["status"] == "completed":
        return
    write(out / "run.json", record)
    start = time.monotonic()
    with torch.inference_mode():
        engine = Engine003(model, device)
        engine.raw_check(rows[:2])
        if "natural.npz" in record["hashes"]:
            nat = load_arrays(out / "natural.npz")
        else:
            assert not (out / "natural.npz").exists()
            nat = natural(engine, rows)
            save_arrays(out / "natural.npz", **nat)
            record["hashes"]["natural.npz"] = digest(out / "natural.npz")
            write(out / "run.json", record)
        bank = torch.tensor(
            load_arrays(HERE / "results" / model / "discovery/signature.npz")["bank"]
        )
        factors = [
            "clean",
            "corrupt",
            "self",
            "wording",
            "order",
            "role",
            "order_role",
            "random0",
            "random1",
            "random2",
        ]
        for f in factors:
            name = f + ".npz"
            if name in record["hashes"]:
                continue
            assert not (out / name).exists()
            if f == "clean":
                result = signature(engine, rows, bank)
            elif f == "corrupt":
                result = signature(
                    engine, [dict(r, token_ids=r["corrupt_tokens"]) for r in rows], bank
                )
            else:
                d = delta(nat, rows, "wording" if f.startswith("random") else f)
                if f.startswith("random"):
                    d = random_matched(d, [r["id"] for r in rows], "bridge003-wording", int(f[-1]))
                result = signature(engine, rows, bank, d)
                result["delta_norm"] = d.flatten(1).norm(dim=-1).numpy()
            save_arrays(out / name, **result)
            record["hashes"][name] = digest(out / name)
            write(out / "run.json", record)
            print(
                "BROAD TEST",
                model,
                f,
                "complete",
                round((time.monotonic() - start) / 60, 1),
                "min",
                flush=True,
            )
        record.update(
            status="completed", finished=now(), elapsed_last_segment=time.monotonic() - start
        )
        write(out / "run.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["freeze", "fit", "select", "test"])
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.stage == "freeze":
        discovery_freeze()
    elif a.stage == "select":
        select()
    elif a.stage == "fit":
        fit(a.model, a.device)
    else:
        test(a.model, a.device)
