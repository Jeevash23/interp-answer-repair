"""Validate the architecture adapter, then screen clean behavior only."""

import argparse

import numpy as np
import torch
from common003 import (
    HERE,
    MODELS,
    ROOT,
    Engine003,
    batches,
    digest,
    freeze,
    now,
    read,
    save_arrays,
    verify_freeze,
    write,
)


def main(model, technical, device):
    verify_freeze("input-freeze.json")
    for name, sha in read(HERE / "data/checkpoints.json").items():
        assert digest(ROOT.parent / name) == sha
    rows = [r for r in read(HERE / "data" / (model + "-candidates.json")) if r["split"] == "screen"]
    output = HERE / "results" / model
    output.mkdir(exist_ok=True)
    path = output / ("technical.json" if technical else "screen.json")
    if path.exists():
        print("Preserved completed stage", path, flush=True)
        return
    if not technical:
        assert all((HERE / "results" / m / "technical.json").exists() for m in MODELS)
        if not (HERE / "data/screen-freeze.json").exists():
            freeze(
                "screen-freeze.json",
                [
                    HERE / "code/common003.py",
                    HERE / "code/neox_numerics.py",
                    HERE / "docs/NUMERICAL-CALIBRATION.md",
                    HERE / "code/screen003.py",
                    HERE / "data/input-freeze.json",
                ],
            )
        verify_freeze("screen-freeze.json")
    with torch.inference_mode():
        engine = Engine003(model, device)
        raw_error = engine.raw_check(rows[:2])
        if technical:
            from assay import technical as validate

            if model == "pythia-410m-deduped":
                engine.config["numerical_tolerance"] = 0.002
            result = validate(engine, rows[:2])
            if model == "pythia-410m-deduped":
                extra = validate(engine, [r for r in rows if r["family"] == 6][:2])
                result["structured_family_validation"] = extra
                result["max_error"] = max(result["max_error"], extra["max_error"])
            result["numerical_tolerance"] = engine.config["numerical_tolerance"]
            result.update(raw_conversion_error=raw_error, utc=now(), model=model)
            write(path, result)
            print("TECHNICAL", model, result, flush=True)
            return
        b = np.empty((len(rows), 6), np.float32)
        for ids in batches(rows, 8):
            values, _ = engine.forward([rows[i] for i in ids])
            b[ids] = values.cpu().numpy()
        save_arrays(output / "screen.npz", behavior=b)
    scores, passed = {}, []
    for f in range(8):
        cells = {}
        for o in range(2):
            mask = np.array([r["family"] == f and r["order"] == o for r in rows])
            cells[str(o)] = {
                "n": int(mask.sum()),
                "pairwise": float(b[mask, 2].mean()),
                "top1": float(b[mask, 3].mean()),
            }
        eligible = all(v["pairwise"] >= 0.95 and v["top1"] >= 0.85 for v in cells.values())
        scores[str(f)] = {"cells": cells, "passes": eligible}
        if eligible:
            passed.append(f)
    selected = sorted(set(passed[:4]) | {0, 1})
    write(
        path,
        {
            "utc": now(),
            "scores": scores,
            "eligible": passed[:4],
            "retained": selected,
            "array_sha256": digest(output / "screen.npz"),
            "screen_freeze": digest(HERE / "data/screen-freeze.json"),
        },
    )
    print(
        "SCREEN", model, "eligible", passed[:4], "retained", selected, "scores", scores, flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--technical", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    main(a.model, a.technical, a.device)
