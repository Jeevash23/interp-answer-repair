"""Resume an interrupted bridge run without changing its frozen scientific code.

Uses the original condition inventory, token order, deterministic per-ID controls,
cached natural measurements and exact engine. Retains the interrupted run record.
"""

import argparse
import shutil
import time

import numpy as np
import torch

from transport_bridge.common import ROOT, digest, now, read, verify, write
from transport_bridge.engine import Engine, result_hooks, state_hook
from transport_bridge.study import batches, conditions, prepare_delta, verify_model

p = argparse.ArgumentParser()
p.add_argument("--model", default="gpt2-medium")
p.add_argument("--device", default="cuda")
args = p.parse_args()
config = read(ROOT / "configs/study.json")
verify(read(ROOT / "data/source-freeze.json")["files"])
verify_model(args.model)
fit_dir = ROOT / "results" / args.model / "fit"
verify(read(fit_dir / "test-freeze.json")["files"])
selected = read(fit_dir / "selection.json")
output = ROOT / "results" / args.model / "test"
info = read(output / "run.json")
assert info["status"] == "running"
assert info["condition_inventory"] == conditions()
assert info["test_freeze_sha256"] == digest(fit_dir / "test-freeze.json")
rows = [r for r in read(ROOT / "data/recipients.json") if r["split"] == "test"]
assert info["record_ids"] == [r["id"] for r in rows]
library = read(ROOT / "data/prompts.json")
with np.load(output / "natural.npz") as data:
    arrays = dict(data)
bank = torch.tensor(arrays["mean_bank"])
existing = {}
for name in info["completed"]:
    path = output / (name + ".npz")
    with np.load(path) as d:
        assert all(np.isfinite(a).all() for a in d.values())
        assert d["behavior"].shape == (256, 6)
    existing[path.name] = digest(path)
remaining = [spec for spec in conditions() if spec["name"] not in info["completed"]]
assert all(not (output / (s["name"] + ".npz")).exists() for s in remaining)
backup = output / "run.before-resume.json"
if backup.exists():
    raise FileExistsError("A resume record already exists; inspect before another resumption")
shutil.copy2(output / "run.json", backup)
record = {
    "resumed_utc": now(),
    "completed_preserved": existing,
    "natural_sha256": digest(output / "natural.npz"),
    "helper_sha256": digest(__file__),
    "reason": "Process absent after session interruption; no scientific parameter changes",
    "remaining": [s["name"] for s in remaining],
}
write(output / "resumption.json", record)
previous_seconds = info["elapsed_seconds"]
start = time.monotonic()
with torch.inference_mode():
    engine = Engine(args.model, config, args.device)
    for spec in remaining:
        delta, donors = prepare_delta(spec, rows, arrays, selected)
        raw, behavior = [], []
        begun = time.monotonic()
        for ix in batches(len(rows), config["batch_size"]):
            rr = [{**rows[i], "token_ids": library[rows[i][spec["base"]]]["token_ids"]} for i in ix]
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
        info["elapsed_seconds"] = previous_seconds + time.monotonic() - start
        info["resumed_utc"] = record["resumed_utc"]
        write(output / "run.json", info)
        print(
            f"Resumed {args.model}: {spec['name']} {len(info['completed'])}/27; {time.monotonic() - begun:.1f}s",
            flush=True,
        )
for name, sha in existing.items():
    assert digest(output / name) == sha
info.update(
    status="completed", finished=now(), hashes={p.name: digest(p) for p in output.glob("*.npz")}
)
write(output / "run.json", info)
print("Resumption complete; previous measurements unchanged", flush=True)
