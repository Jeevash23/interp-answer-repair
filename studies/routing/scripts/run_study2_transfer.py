"""Fixed test-set nested interventions after selection is frozen."""

from __future__ import annotations

import resource
import time

import numpy as np
import torch

from causal_circuit_routing.io import ROOT, digest, now, read_json, write_json
from causal_circuit_routing.study2_engine import Engine, batches, state_hook
from causal_circuit_routing.study2_run import matched_random, measure, project, verify_inputs

DONOR_CELLS = [
    f"{w}_{f}_{a}"
    for w in ("same", "other")
    for f in ("same", "cross")
    for a in ("same", "different")
]
CONTROLS = [
    "shuffled",
    "random_0",
    "random_1",
    "random_2",
    "wrong_position",
    "wrong_layer",
    "mlp_control",
]


def main():
    verify_inputs(selection=True)
    config = read_json(ROOT / "configs/study-002.json")
    selection = read_json(ROOT / "results/study-002/family/selection.json")
    if not selection["family_gate_pass"] or not selection["predictive_gate_pass"]:
        raise RuntimeError("Family/prediction gate failed; do not run a routing confirmation")
    selected = selection["selected"]
    output = ROOT / "results/study-002/transfer"
    output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in read_json(ROOT / "data/study-002-recipients.json") if r["split"] == "test"]
    library = read_json(ROOT / "data/study-002-prompts.json")
    n = len(rows)
    info = {
        "started_utc": now(),
        "status": "running",
        "config": config,
        "record_ids": [r["id"] for r in rows],
        "completed_conditions": [],
        "selection_freeze_sha256": digest(ROOT / "data/study-002-selection-freeze.json"),
        "selected_layer": selected["layer"],
        "selected_position": selected["position"],
    }
    write_json(output / "run.json", info)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            engine = Engine(config)
            natural = measure(engine, rows, library, config, label="test natural signatures")
            np.savez_compressed(output / "natural.npz", **natural)
            donor_ids = sorted({pid for r in rows for pid in r["donors"].values()})
            donors = [library[pid] for pid in donor_ids]
            states = torch.empty(len(donors), 768)
            for ix in batches(donors, config["batch_size"]):
                _, features, _, _ = engine.forward_cache([donors[i] for i in ix])
                states[ix] = features[:, selected["site_index"]]
            state_bank = {pid: states[i] for i, pid in enumerate(donor_ids)}
            np.savez_compressed(output / "donor-states.npz", states=states.numpy())
            write_json(output / "donor-state-ids.json", donor_ids)
            current = torch.tensor(natural["features"][:, selected["site_index"]])
            corrupt_z = torch.tensor(natural["corrupt_z"])
            rng = np.random.default_rng(config["seed"])
            shuffled_ids = [None] * n
            for family in ("ABBA", "BABA"):
                ix = [i for i, r in enumerate(rows) if r["family"] == family]
                perm = rng.permutation(ix)
                for i, j in zip(ix, perm, strict=True):
                    shuffled_ids[i] = rows[j]["donors"]["same_cross_same"]
            for mi, mode in enumerate(config["modes"]):
                for ci, condition in enumerate(DONOR_CELLS + CONTROLS):
                    condition_started = time.monotonic()
                    if condition in DONOR_CELLS:
                        pids = [r["donors"][condition] for r in rows]
                    elif condition == "shuffled":
                        pids = shuffled_ids
                    else:
                        pids = [r["donors"]["same_cross_same"] for r in rows]
                    donor = torch.stack([state_bank[pid] for pid in pids])
                    delta = project(donor - current, selected["unit_direction"], mode)
                    layer = selected["layer"]
                    positions = torch.tensor([r["positions"][selected["position"]] for r in rows])
                    kind = "residual"
                    control_seed = config["seed"] + mi * 100000 + ci * 1000
                    if condition.startswith("random_"):
                        delta = matched_random(delta, control_seed)
                    elif condition == "wrong_position":
                        positions -= 1
                        assert all(
                            r["token_ids"][int(p)] not in (r["io_token_id"], r["subject_token_id"])
                            for r, p in zip(rows, positions, strict=True)
                        )
                    elif condition == "wrong_layer":
                        layer = max(0, layer - 3)
                    elif condition == "mlp_control":
                        layer, kind = 0, "mlp"
                    raw, probability = (
                        np.empty((2, n, 72), np.float32),
                        np.empty((2, n, 72), np.float32),
                    )
                    behavior = {}
                    if condition == "same_same_same":
                        assert delta.abs().max().item() < config["numerical_tolerance"]
                        raw[:] = natural["raw"]
                        probability[:] = natural["probability_raw"]
                        behavior = {
                            k.removeprefix("clean_"): a
                            for k, a in natural.items()
                            if k.startswith("clean_")
                        }
                    else:
                        for ix in batches(rows, config["batch_size"]):
                            batch = [rows[i] for i in ix]
                            edit = state_hook(layer, positions[ix], delta[ix], kind)
                            a, p, outcomes, _, _ = engine.signatures(batch, corrupt_z[ix], [edit])
                            raw[:, ix] = a
                            probability[:, ix] = p
                            for key, value in outcomes.items():
                                if key not in behavior:
                                    behavior[key] = np.empty(n, np.float64)
                                behavior[key][ix] = value.numpy()
                    if not np.isfinite(raw).all() or not np.isfinite(probability).all():
                        raise RuntimeError("Non-finite intervention effects")
                    label = f"{mode}__{condition}"
                    np.savez_compressed(
                        output / f"{label}.npz",
                        raw=raw,
                        probability_raw=probability,
                        delta_norm=delta.norm(dim=-1).numpy(),
                        **behavior,
                    )
                    write_json(
                        output / f"{label}.json",
                        {
                            "mode": mode,
                            "condition": condition,
                            "donor_ids": pids,
                            "donor_io_token_ids": [library[pid]["io_token_id"] for pid in pids],
                            "donor_families": [library[pid]["family"] for pid in pids],
                            "layer": layer,
                            "positions": positions.tolist(),
                            "kind": kind,
                            "control_seed": control_seed,
                            "duration_seconds": time.monotonic() - condition_started,
                            "sha256": digest(output / f"{label}.npz"),
                        },
                    )
                    info["completed_conditions"].append(label)
                    info["elapsed_seconds"] = time.monotonic() - started
                    write_json(output / "run.json", info)
                    print(
                        f"Completed {label} ({len(info['completed_conditions'])}/30); "
                        f"condition={time.monotonic() - condition_started:.1f}s; "
                        f"total={(time.monotonic() - started) / 60:.1f}min",
                        flush=True,
                    )
        verify_inputs(selection=True)
        info.update(
            status="completed",
            completed_utc=now(),
            duration_seconds=time.monotonic() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            hashes={p.name: digest(p) for p in output.glob("*.npz")},
        )
        write_json(output / "run.json", info)
    except Exception as error:
        info.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - started)
        write_json(output / "run.json", info)
        raise


if __name__ == "__main__":
    main()
