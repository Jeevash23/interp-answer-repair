"""Freeze, validate, fit, and resumably execute the prospective follow-up."""

import argparse
import os
import tarfile
import time

import torch
from assay import (
    CausalEngine,
    check_freeze,
    collect,
    load_arrays,
    measure,
    select_groups,
    technical,
    trace,
)
from design import HERE, specs

from transport_bridge.common import ROOT, digest, now, read, verify, write
from transport_bridge.study import verify_model


def make_rows():
    selection = read(HERE / "data/family-selection-v2.json")
    assert selection["status"] == "passed"
    families = selection["families"]
    rows = [
        dict(r, wording=families.index(r["family"]))
        for r in read(HERE / "data/candidates-v2.json")
        if r["family"] in families
    ]
    index = {
        (r["split"], r["template"], r["pair"], r["role"], r["wording"], r["order"]): r for r in rows
    }
    for r in rows:
        r["donors"] = {}
        for label, dw, do, dr in [
            ("self", 0, 0, 0),
            ("role", 0, 0, 1),
            ("wording", 1, 0, 0),
            ("wording_role", 1, 0, 1),
            ("order", 0, 1, 0),
            ("order_role", 0, 1, 1),
        ]:
            d = index[
                (
                    r["split"],
                    r["template"],
                    r["pair"],
                    r["role"] ^ dr,
                    r["wording"] ^ dw,
                    r["order"] ^ do,
                )
            ]
            r["donors"][label] = d["id"]
    return rows


def freeze():
    path = HERE / "data/scientific-freeze.json"
    if path.exists():
        raise FileExistsError(path)
    check_freeze("screen-freeze.json")
    check_freeze("screen-freeze-v2.json")
    verify(read(ROOT / "data/source-freeze.json")["files"])
    for model in ["gpt2-medium", "gpt2-small"]:
        assert read(HERE / "results" / model / "technical.json")["max_error"] <= 0.0005
    write(HERE / "data/recipients.json", make_rows())
    files = list((HERE / "code").glob("*.py")) + list((HERE / "docs").glob("*.md"))
    files += list((HERE / "data").glob("*.json"))
    files += [ROOT / p for p in read(ROOT / "data/source-freeze.json")["files"]]
    record = {
        "utc": now(),
        "files": {os.path.relpath(p, HERE): digest(p) for p in sorted(set(files))},
        "screening_amendment": "SCREENING-AMENDMENT-001.md",
        "local_not_public_preregistration": True,
    }
    write(path, record)
    with tarfile.open(HERE / "data/frozen-science.tar.gz", "w:gz") as archive:
        for p in sorted(set(files)):
            name = os.path.relpath(p, HERE)
            archive.add(
                p,
                arcname="shared/" + str(p.relative_to(ROOT))
                if name.startswith("..")
                else "study/" + name,
            )
        archive.add(path, arcname="study/data/scientific-freeze.json")
    print("Scientific freeze:", record["utc"], len(files), "files", flush=True)


def execute(model, stage, device):
    if stage != "technical":
        check_freeze("scientific-freeze.json")
    else:
        check_freeze("screen-freeze.json")
        check_freeze("screen-freeze-v2.json")
    verify_model(model)
    config = read(ROOT / "configs/study.json")
    all_rows = make_rows() if stage == "technical" else read(HERE / "data/recipients.json")
    model_dir = HERE / "results" / model
    if stage == "technical":
        rows = [r for r in all_rows if r["split"] == "screen"][:2]
        with torch.inference_mode():
            engine = CausalEngine(model, config, device)
            result = technical(engine, rows)
        result.update(utc=now(), model=model)
        write(model_dir / "technical.json", result)
        print(result, flush=True)
        return
    assert read(model_dir / "technical.json")["max_error"] <= 0.0005
    directory = model_dir / stage
    directory.mkdir(parents=True, exist_ok=True)
    rows = [r for r in all_rows if r["split"] == ("discovery" if stage == "fit" else "test")]
    record_path = directory / "run.json"
    if stage == "test":
        selected = read(model_dir / "fit/selection.json")
        for name, sha in read(model_dir / "fit/test-freeze.json")["files"].items():
            assert digest(HERE / name) == sha
    if record_path.exists():
        info = read(record_path)
        if info["status"] == "completed":
            print(model, stage, "already completed; outputs preserved", flush=True)
            return
        assert info["source_freeze"] == digest(HERE / "data/scientific-freeze.json")
        for name, sha in info["hashes"].items():
            assert digest(directory / name) == sha
        info.setdefault("resumptions", []).append(now())
    else:
        info = {
            "status": "running",
            "model": model,
            "stage": stage,
            "started": now(),
            "source_freeze": digest(HERE / "data/scientific-freeze.json"),
            "record_ids": [r["id"] for r in rows],
            "hashes": {},
            "elapsed_seconds": 0.0,
        }
        if stage == "test":
            info["test_freeze"] = digest(model_dir / "fit/test-freeze.json")
    write(record_path, info)
    start, prior = time.monotonic(), info["elapsed_seconds"]

    def save_progress(path):
        info["hashes"][path.name] = digest(path)
        info["elapsed_seconds"] = prior + time.monotonic() - start
        write(record_path, info)

    def existing(path):
        if path.exists() and path.name not in info["hashes"]:
            raise RuntimeError(
                "Unrecorded existing output; inspect rather than overwrite: " + str(path)
            )
        return path.name in info["hashes"]

    with torch.inference_mode():
        engine = CausalEngine(model, config, device)
        path = directory / "natural.npz"
        if existing(path):
            natural = load_arrays(path)
        else:
            bank = (
                None
                if stage == "fit"
                else torch.tensor(load_arrays(model_dir / "fit/natural.npz")["mean_bank"])
            )
            natural = collect(engine, rows, path, bank)
            save_progress(path)
        inventory = (
            [s for s in specs() if s["name"] in ["head_self", "head_wording", "head_order"]]
            if stage == "fit"
            else specs()
        )
        edits = {}
        for i, spec in enumerate(inventory):
            path = directory / (spec["name"] + ".npz")
            if existing(path):
                result = load_arrays(path)
            else:
                result = measure(engine, rows, natural, spec, path)
                save_progress(path)
            edits[spec["donor"]] = result
            print(
                f"{model} {stage}: {spec['name']} {i + 1}/{len(inventory)}; {info['elapsed_seconds'] / 60:.1f} min",
                flush=True,
            )
        if stage == "fit":
            selected = select_groups(engine, rows, natural, edits)
            write(directory / "selection.json", selected)
            files = {
                str(p.relative_to(HERE)): digest(p)
                for p in [
                    directory / "selection.json",
                    directory / "natural.npz",
                    HERE / "code/analyze.py",
                    *[directory / (s["name"] + ".npz") for s in inventory],
                ]
            }
            write(
                directory / "test-freeze.json",
                {
                    "utc": now(),
                    "files": files,
                    "source_freeze": digest(HERE / "data/scientific-freeze.json"),
                },
            )
            print("Selected groups:", selected["groups"], flush=True)
        else:
            for spec in [
                s for s in specs() if s["name"] in ["head_self", "head_wording", "head_order"]
            ]:
                path = directory / ("trace_" + spec["donor"] + ".npz")
                if not existing(path):
                    trace(engine, rows, natural, spec, selected, path)
                    save_progress(path)
                print(model, "causal trace complete:", spec["donor"], flush=True)
    info.update(
        status="completed", finished=now(), elapsed_seconds=prior + time.monotonic() - start
    )
    write(record_path, info)
    print(model, stage, "COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "technical", "fit", "test"])
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    else:
        execute(args.model, args.stage, args.device)
