"""Discovery/development family validation and candidate selection; never evaluate test rows."""

from __future__ import annotations

import json
import resource
import time

import numpy as np
import torch

from causal_circuit_routing.io import ROOT, digest, now, read_json, write_json
from causal_circuit_routing.metrics import classify, fit_signature, hierarchical_interval
from causal_circuit_routing.study2_engine import Engine


def main():
    from causal_circuit_routing.study2_run import measure, verify_inputs

    verify_inputs()
    config = read_json(ROOT / "configs/study-002.json")
    output = ROOT / "results/study-002/family"
    output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in read_json(ROOT / "data/study-002-recipients.json") if r["split"] != "test"]
    library = read_json(ROOT / "data/study-002-prompts.json")
    started = time.monotonic()
    info = {
        "started_utc": now(),
        "status": "running",
        "record_ids": [r["id"] for r in rows],
        "config": config,
        "freeze_sha256": digest(ROOT / "data/study-002-freeze.json"),
    }
    write_json(output / "run.json", info)
    try:
        with torch.inference_mode():
            engine = Engine(config)
            info["checks"] = engine.validate(rows[:4])
            print("Suffix and nested-edit validation:", info["checks"], flush=True)
            arrays = measure(engine, rows, library, config, label="family validation")
        np.savez_compressed(output / "measurements.npz", **arrays)
        train = np.array([r["split"] == "discovery" for r in rows])
        dev = ~train
        labels = np.array([int(r["family"] == "ABBA") for r in rows])
        fits, scores, y = {}, {}, []
        family_pass = True
        for bi, baseline in enumerate(config["baselines"]):
            raw = arrays["raw"][bi]
            fit = fit_signature(raw[train], labels[train], config["scale_floor"])
            fits[baseline] = {k: v.tolist() for k, v in fit.items()}
            predicted = classify(raw, fit)
            correct = np.where(predicted == -1, 0.5, predicted == labels).astype(float)
            balanced = float(np.mean([correct[dev & (labels == c)].mean() for c in (0, 1)]))
            scaled = raw / fit["scale"]
            axis_coordinate = (
                (scaled - fit["centroids"].mean(0)) @ fit["axis"] / np.sum(fit["axis"] ** 2)
            )
            y.append(axis_coordinate)
            grid = np.zeros((4, 4))
            templates = sorted({r["template_id"] for r in rows if r["split"] == "development"})
            pairs = sorted({r["name_pair_id"] for r in rows if r["split"] == "development"})
            for ti, template in enumerate(templates):
                for pi, pair in enumerate(pairs):
                    ix = np.array(
                        [r["template_id"] == template and r["name_pair_id"] == pair for r in rows]
                    )
                    grid[ti, pi] = (
                        axis_coordinate[ix & (labels == 1)].mean()
                        - axis_coordinate[ix & (labels == 0)].mean()
                    )
            interval = hierarchical_interval(grid, config["bootstrap_draws"], config["seed"])
            family_pass &= balanced >= config["minimum_family_accuracy"] and grid.mean() > 0
            scores[baseline] = {
                "development_balanced_accuracy": balanced,
                "natural_separation": float(grid.mean()),
                "bootstrap_95": interval,
            }
        target = np.mean(y, axis=0)
        candidates = []
        for si, (layer, position) in enumerate(engine.sites):
            x = arrays["features"][:, si].astype(np.float64)
            mean_x = x[train].mean(0)
            mean_y = target[train].mean()
            xc = x[train] - mean_x
            alpha = config["ridge_penalty_fraction"] * np.mean(np.sum(xc**2, axis=1))
            weight = xc.T @ np.linalg.solve(
                xc @ xc.T + alpha * np.eye(train.sum()), target[train] - mean_y
            )
            predicted_y = (x - mean_x) @ weight + mean_y
            mse = float(np.mean((predicted_y[dev] - target[dev]) ** 2))
            r2 = 1 - mse / np.var(target[dev])
            norm = np.linalg.norm(weight)
            if norm < 1e-12:
                raise RuntimeError("Degenerate predictor")
            candidates.append(
                {
                    "site_index": si,
                    "layer": layer,
                    "position": position,
                    "development_r2": float(r2),
                    "development_mse": mse,
                    "alpha": float(alpha),
                    "weight": weight.tolist(),
                    "unit_direction": (weight / norm).tolist(),
                    "mean_x": mean_x.tolist(),
                    "mean_y": float(mean_y),
                }
            )
        candidates.sort(key=lambda c: (-c["development_r2"], c["layer"], c["site_index"]))
        behavior = {}
        behavior_pass, open_vocab_pass = True, True
        for split in ("discovery", "development"):
            for family in ("ABBA", "BABA"):
                mask = np.array([r["split"] == split and r["family"] == family for r in rows])
                pairwise = float(arrays["clean_pairwise_correct"][mask].mean())
                top1 = float(
                    np.mean(
                        arrays["clean_top1_token_id"][mask]
                        == np.array([r["io_token_id"] for r in rows])[mask]
                    )
                )
                behavior_pass &= pairwise >= config["minimum_pairwise_accuracy"]
                open_vocab_pass &= top1 >= config["minimum_open_vocab_accuracy"]
                behavior[f"{split}/{family}"] = {"pairwise": pairwise, "open_vocab_top1": top1}
        selected = candidates[0]
        selection = {
            "selected": selected,
            "all_candidates": candidates,
            "signature_fits": fits,
            "family_validation": scores,
            "behavior": behavior,
            "family_gate_pass": bool(family_pass and behavior_pass),
            "natural_task_validity_pass": bool(open_vocab_pass),
            "predictive_gate_pass": selected["development_r2"] > 0,
            "selected_utc": now(),
            "test_outcomes_accessed": False,
            "source_measurements_sha256": digest(output / "measurements.npz"),
        }
        write_json(output / "selection.json", selection)
        selection_files = [
            "configs/study-002.json",
            "docs/STUDY-002-PROTOCOL.md",
            "results/study-002/family/selection.json",
        ]
        selection_files += [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")]
        selection_files += [
            str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*study2*.py")
        ]
        write_json(
            ROOT / "data/study-002-selection-freeze.json",
            {
                "frozen_utc": now(),
                "before_test_outcomes": True,
                "files": {p: digest(ROOT / p) for p in sorted(selection_files)},
            },
        )
        info.update(
            status="completed",
            duration_seconds=time.monotonic() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            completed_utc=now(),
        )
        write_json(output / "run.json", info)
        print(
            json.dumps(
                {
                    "family": scores,
                    "behavior": behavior,
                    "selected": {k: selected[k] for k in ("layer", "position", "development_r2")},
                    "family_gate": selection["family_gate_pass"],
                    "natural_task_validity": open_vocab_pass,
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as error:
        info.update(status="failed", error=repr(error))
        write_json(output / "run.json", info)
        raise


if __name__ == "__main__":
    main()
