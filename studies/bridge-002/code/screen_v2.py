"""Behavior-only grammatical-family selection with no test inference."""

import argparse

import numpy as np
import torch
from design import HERE

from transport_bridge.common import ROOT, digest, now, read, write
from transport_bridge.engine import Engine
from transport_bridge.study import batches, verify_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    for name, sha in read(HERE / "data/screen-freeze-v2.json")["files"].items():
        assert digest(HERE / name) == sha
    output = HERE / "results/screen-v2"
    output.mkdir(exist_ok=False)
    rows = [r for r in read(HERE / "data/candidates-v2.json") if r["split"] == "screen"]
    scores = {}
    with torch.inference_mode():
        for model in ["gpt2-medium", "gpt2-small"]:
            verify_model(model)
            engine = Engine(model, read(ROOT / "configs/study.json"), args.device)
            values = []
            for ix in batches(len(rows), 8):
                b, _ = engine.forward([rows[i] for i in ix])
                values.append(b.cpu().numpy())
            values = np.concatenate(values)
            np.savez_compressed(output / (model + ".npz"), behavior=values)
            scores[model] = {}
            for f in [1, 4, 5, 6]:
                for o in range(2):
                    mask = np.array([r["family"] == f and r["order"] == o for r in rows])
                    v = values[mask]
                    scores[model][f"f{f}_o{o}"] = {
                        "n": len(v),
                        "pairwise": float(v[:, 2].mean()),
                        "top1": float(v[:, 3].mean()),
                    }
            print(model, scores[model], flush=True)
            del engine
            torch.cuda.empty_cache()

    def passes(f):
        return all(
            scores[m][f"f{f}_o{o}"]["pairwise"] >= 0.95 and scores[m][f"f{f}_o{o}"]["top1"] >= 0.85
            for m in scores
            for o in range(2)
        )

    chosen = next((f for f in [4, 5, 6] if passes(1) and passes(f)), None)
    record = {
        "utc": now(),
        "scores": scores,
        "status": "passed" if chosen else "failed",
        "families": [1, chosen] if chosen else [],
        "screen_freeze_sha256": digest(HERE / "data/screen-freeze-v2.json"),
    }
    write(HERE / "data/family-selection-v2.json", record)
    print("SELECTION", record["status"], record["families"], flush=True)


if __name__ == "__main__":
    main()
