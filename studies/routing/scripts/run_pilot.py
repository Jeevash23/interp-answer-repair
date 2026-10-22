"""Run only the frozen 64-prompt, 72-node family feasibility pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

from causal_circuit_routing.io import (
    ROOT,
    digest,
    load_records,
    now,
    read_json,
    verify_freeze,
    write_json,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
from activation_transfer.paths import recorded_path
import torch
from transformer_lens.model_bridge import TransformerBridge

from causal_circuit_routing.interventions import head_hook, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pilot-001")
    args = parser.parse_args()
    freeze = verify_freeze()
    config = read_json(ROOT / "configs/pilot-001.json")
    if config["device"] != "cpu" or config["dtype"] != "float32":
        raise ValueError("This pilot is frozen for CPU float32; changes need a new protocol")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    records = load_records()
    coordinates = [
        (layer, head) for layer in config["layers"] for head in range(config["heads_per_layer"])
    ]
    info = {
        "started_utc": now(),
        "status": "running",
        "config": config,
        "freeze_sha256": digest(ROOT / "data/pilot-001-freeze.json"),
        "manifest_sha256": freeze["files"]["data/pilot-001.jsonl"],
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "command": sys.argv,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformer-lens", "transformers", "numpy", "safetensors")
        },
        "coordinates": coordinates,
        "record_ids": [r["id"] for r in records],
        "note": "No routing interventions; natural causal-signature pilot only",
    }
    write_json(output / "run.json", info)
    try:
        model = TransformerBridge.boot_transformers(
            str(recorded_path(ROOT / config["model_path"])),
            device=torch.device("cpu"),
            dtype=torch.float32,
            trust_remote_code=False,
        )
        model.eval()
        sample = records[: config["batch_size"]]
        tokens = torch.tensor([r["clean_token_ids"] for r in sample])
        with torch.inference_mode():
            native = model(tokens).float()[:, -1, :].clone()
            model.enable_compatibility_mode()
            model.set_use_attn_result(False)
            model.eval()
            compatible = model(tokens).float()[:, -1, :]
            centered_error = float(
                (
                    (native - native.mean(-1, keepdim=True))
                    - (compatible - compatible.mean(-1, keepdim=True))
                )
                .abs()
                .max()
            )
        if centered_error > config["centered_bridge_tolerance"]:
            raise RuntimeError(f"Native/bridge centered logits mismatch: {centered_error}")
        info["checks"] = {"native_vs_bridge_centered_max_error": centered_error}
        del native, compatible
        print(f"Model ready; centered bridge error={centered_error:.6g}", flush=True)

        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[(r["split"], r["template_id"], len(r["clean_token_ids"]))].append(i)
        bs = config["batch_size"]
        batches = [
            indices[start : start + bs]
            for _, indices in sorted(groups.items())
            for start in range(0, len(indices), bs)
        ]
        n = len(records)
        bank = {variant: torch.empty(n, 6, 12, 64) for variant in ("clean", "corrupt")}
        behavior = {variant: {} for variant in bank}
        filter_names = [f"blocks.{layer}.attn.hook_z" for layer in config["layers"]]
        with torch.inference_mode():
            for variant, variant_bank in bank.items():
                for ix in batches:
                    batch = [records[i] for i in ix]
                    tokens = torch.tensor([r[f"{variant}_token_ids"] for r in batch])
                    logits, cache = model.run_with_cache(tokens, names_filter=filter_names)
                    measured = outcomes(logits, batch)
                    for name, value in measured.items():
                        if name not in behavior[variant]:
                            behavior[variant][name] = np.empty(n, dtype=np.float64)
                        behavior[variant][name][ix] = value.numpy()
                    for li, layer in enumerate(config["layers"]):
                        z = cache[f"blocks.{layer}.attn.hook_z"]
                        if z.shape != (len(ix), tokens.shape[1], 12, 64):
                            raise RuntimeError("Cache has wrong head geometry")
                        variant_bank[ix, li] = z[:, -1].cpu()
                    del logits, cache
                print(f"Cached {variant} behavior and 72 END heads for {n} prompts", flush=True)

            np.savez_compressed(
                output / "behavior.npz",
                **{f"{v}_{k}": a for v, d in behavior.items() for k, a in d.items()},
            )
            behavior_json = []
            for i, r in enumerate(records):
                behavior_json.append(
                    {
                        "id": r["id"],
                        "split": r["split"],
                        "family": r["family"],
                        "template_id": r["template_id"],
                        **{
                            f"{v}_{k}": float(a[i])
                            for v, d in behavior.items()
                            for k, a in d.items()
                        },
                    }
                )
            write_json(output / "behavior.json", behavior_json)
            rates = {}
            for split in ("discovery", "heldout"):
                for family in ("ABBA", "BABA"):
                    ix = [
                        i
                        for i, r in enumerate(records)
                        if r["split"] == split and r["family"] == family
                    ]
                    rates[f"{split}/{family}"] = float(
                        behavior["clean"]["pairwise_correct"][ix].mean()
                    )
            info["behavior_accuracy"] = rates
            print(f"Pairwise clean accuracies: {rates}", flush=True)
            if min(rates.values()) < config["minimum_behavior_accuracy"]:
                info.update(status="behavior_gate_failed", completed_utc=now())
                write_json(output / "run.json", info)
                print("Behavior gate failed; stopping without head sweep", flush=True)
                return

            # Identity checks use the same cached clean node, with two families in the first batch.
            ix = batches[0]
            batch = [records[i] for i in ix]
            tokens = torch.tensor([r["clean_token_ids"] for r in batch])
            positions = torch.tensor([r["positions"]["end"] for r in batch])
            identity_error = 0.0
            for layer, head in coordinates:
                li = config["layers"].index(layer)
                hook, calls = head_hook(layer, head, bank["clean"][ix, li, head], positions)
                checked = outcomes(model.run_with_hooks(tokens, fwd_hooks=[hook]), batch)
                if calls["count"] != 1:
                    raise RuntimeError("Hook did not run exactly once")
                error = np.max(
                    np.abs(
                        checked["logit_difference"].numpy()
                        - behavior["clean"]["logit_difference"][ix]
                    )
                )
                identity_error = max(identity_error, float(error))
            if identity_error > config["identity_tolerance"]:
                raise RuntimeError(f"Identity intervention failed: {identity_error}")
            info["checks"]["all_72_identity_max_logit_error"] = identity_error
            print(f"All 72 identity hooks passed; max error={identity_error:.6g}", flush=True)

            raw = np.empty((len(config["baselines"]), n, len(coordinates)), dtype=np.float64)
            probability_raw = np.empty_like(raw)
            for ci, (layer, head) in enumerate(coordinates):
                li = config["layers"].index(layer)
                for bi, baseline in enumerate(config["baselines"]):
                    for ix in batches:
                        batch = [records[i] for i in ix]
                        tokens = torch.tensor([r["clean_token_ids"] for r in batch])
                        positions = torch.tensor([r["positions"]["end"] for r in batch])
                        replacement = (
                            torch.zeros(len(ix), 64)
                            if baseline == "zero"
                            else bank["corrupt"][ix, li, head]
                        )
                        hook, calls = head_hook(layer, head, replacement, positions)
                        patched = outcomes(model.run_with_hooks(tokens, fwd_hooks=[hook]), batch)
                        if calls["count"] != 1:
                            raise RuntimeError("Missing or repeated intervention hook")
                        raw[bi, ix, ci] = (
                            behavior["clean"]["logit_difference"][ix]
                            - patched["logit_difference"].numpy()
                        )
                        probability_raw[bi, ix, ci] = (
                            behavior["clean"]["probability_difference"][ix]
                            - patched["probability_difference"].numpy()
                        )
                if (ci + 1) % 12 == 0:
                    print(
                        f"Measured {ci + 1}/72 heads under both baselines; "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )

            if not np.isfinite(raw).all() or not np.isfinite(probability_raw).all():
                raise RuntimeError("Non-finite signatures")
            if np.max(np.abs(raw)) < 1e-6:
                raise RuntimeError("All interventions were ineffective; inspect hooks")
            # Check hook cleanup after the last intervention against a clean forward pass.
            post = outcomes(model(tokens), batch)["logit_difference"].numpy()
            cleanup_error = float(np.max(abs(post - behavior["clean"]["logit_difference"][ix])))
            if cleanup_error > config["identity_tolerance"]:
                raise RuntimeError("Intervention hooks leaked into the next run")
            info["checks"]["hook_cleanup_max_logit_error"] = cleanup_error
            np.savez_compressed(output / "signatures.npz", raw=raw, probability_raw=probability_raw)
            # Save the small reference bank for later validation without recomputing old runs.
            np.savez_compressed(
                output / "reference_nodes.npz",
                clean=bank["clean"].numpy(),
                corrupt=bank["corrupt"].numpy(),
            )
        verify_freeze()
        info.update(
            status="completed",
            completed_utc=now(),
            duration_seconds=time.monotonic() - started,
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
        info["result_sha256"] = {
            p.name: digest(p)
            for p in output.iterdir()
            if p.suffix == ".npz" or p.name == "behavior.json"
        }
        write_json(output / "run.json", info)
        print(f"Completed pilot in {info['duration_seconds']:.1f}s", flush=True)
    except Exception as error:
        info.update(
            status="failed",
            failed_utc=now(),
            error=repr(error),
            duration_seconds=time.monotonic() - started,
        )
        write_json(output / "run.json", info)
        raise


if __name__ == "__main__":
    main()
