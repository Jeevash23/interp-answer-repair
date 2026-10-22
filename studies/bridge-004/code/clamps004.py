"""Fixed historical groups on the same fresh F1/F2 wording pair."""
import time
import numpy as np
import torch
from common004 import *
verify_freeze=verify
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


def run():
    verify_freeze("input-freeze.json")
    verify_freeze("gpt2-medium-test-freeze.json")
    selected = read(HERE / 'data/gpt2-medium-selection.json')
    rows = rows_for('gpt2-medium','test',selected['families'])
    selection = read(PREV / 'data/medium-selection.json')
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
            "freeze": digest(HERE / "data/gpt2-medium-test-freeze.json"),
            "record_ids": [r["id"] for r in rows],
            "variants": [v[0] for v in modes],
        }
    )
    assert record["freeze"] == digest(HERE / "data/gpt2-medium-test-freeze.json")
    for name, sha in record["hashes"].items():
        assert digest(out / name) == sha
    if record["status"] == "completed":
        return
    write(record_path, record)
    start = time.monotonic()
    with torch.inference_mode():
        engine = engine_for("gpt2-medium")
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
                "denoised": np.empty((19, len(rows), 6), np.float32),
            }
            for bi, ids in enumerate(batches(rows)):
                rr = [dict(rows[i], token_ids=rows[i]["corrupt_tokens"]) for i in ids]
                b, cache = engine.forward(rr)
                hs = hooks(engine, d[ids], rr)
                e, edited_cache = engine.forward(rr, hs)
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
                    d0, _ = engine.forward(rr, engine.clamp(edited_cache, heads, component))
                    values["denoised"][vi, ids] = d0.cpu().numpy()
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

if __name__=="__main__":run()
