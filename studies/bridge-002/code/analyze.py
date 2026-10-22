"""Frozen paired statistics for valid-wording transport and causal follow-ups."""

import argparse
from activation_transfer.paths import verify_frozen_file

import numpy as np
from assay import check_freeze, load_arrays, save_arrays
from design import HERE, specs

from transport_bridge.common import digest, now, read, write
from transport_bridge.statistics import axis_coordinates, block_interval, geometry


def main(model, output=None):
    check_freeze("scientific-freeze.json")
    directory = HERE / "results" / model
    run = read(directory / "test/run.json")
    assert run["status"] == "completed"
    for name, sha in run["hashes"].items():
        assert digest(directory / "test" / name) == sha
    for name, sha in read(directory / "fit/test-freeze.json")["files"].items():
        verify_frozen_file(HERE / name, sha)
    output = HERE / output if output else directory / "analysis"
    output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in read(HERE / "data/recipients.json") if r["split"] == "test"]
    selected = read(directory / "fit/selection.json")
    reference = selected["reference"]
    natural = load_arrays(directory / "test/natural.npz")
    clean, corrupt = natural["clean_behavior"], natural["corrupt_behavior"]
    masks = {
        f"w{w}_o{o}": np.array([r["wording"] == w and r["order"] == o for r in rows])
        for w in range(2)
        for o in range(2)
    }

    def interval(x, mask=None, mult=1):
        return block_interval(x, rows, mask, multiplicity=mult)

    summary = {
        "model": model,
        "n_test": len(rows),
        "templates": 8,
        "name_triples": 6,
        "families": read(HERE / "data/family-selection-v2.json")["families"],
        "natural": {},
        "conditions": {},
        "primary_wording": {},
        "factorial": {},
        "blocking": {},
        "controls": {},
        "joint": {},
        "attention": {},
        "scope": "Conditional paired measurements of fixed head groups; not unique causal pathways",
        "baselines": ["zero", "discovery_mean"],
    }
    coords = {
        f: axis_coordinates(natural["clean_raw"], rows, reference, f) for f in ["wording", "order"]
    }
    for key, mask in masks.items():
        entry = {
            "n": int(mask.sum()),
            "clean_pairwise": float(clean[mask, 2].mean()),
            "clean_top1": float(clean[mask, 3].mean()),
            "corrupt_pairwise": float(corrupt[mask, 2].mean()),
            "corrupt_top1": float(corrupt[mask, 3].mean()),
            "clean_minus_corrupt_logit": interval(clean[:, 0] - corrupt[:, 0], mask),
            "clean_minus_corrupt_probability": interval(clean[:, 1] - corrupt[:, 1], mask),
            "family_accuracy": {},
        }
        entry["behavior_valid"] = entry["clean_pairwise"] >= 0.95 and entry["clean_top1"] >= 0.85
        for f, coordinates in coords.items():
            targets = np.array([r[f] for r in rows])
            entry["family_accuracy"][f] = [
                float(((a[mask] > 0) == targets[mask]).mean()) for a in coordinates
            ]
        summary["natural"][key] = entry
    metrics, edits = {}, {}
    for spec in specs():
        name = spec["name"]
        data = load_arrays(directory / "test" / (name + ".npz"))
        b = data["behavior"]
        edits[name] = data
        metrics[name + "__behavior"] = b
        rec = b[:, 0] - corrupt[:, 0]
        item = {
            "spec": spec,
            "raw_logit_recovery": interval(rec),
            "raw_probability_recovery": interval(b[:, 1] - corrupt[:, 1]),
            "pairwise_accuracy": float(b[:, 2].mean()),
            "top1_accuracy": float(b[:, 3].mean()),
            "cells": {},
            "geometry": {},
        }
        for key, mask in masks.items():
            item["cells"][key] = {
                "recovery": interval(rec, mask),
                "pairwise": float(b[mask, 2].mean()),
                "top1": float(b[mask, 3].mean()),
            }
        f = (
            "wording"
            if "wording" in spec["donor"]
            else "order"
            if "order" in spec["donor"]
            else None
        )
        if f:
            donor_raw = natural["clean_raw"][:, data["donors"]].copy()
            if spec["donor"].endswith("role"):
                # Recipient scoring exchanges IO and subject relative to this donor.
                donor_raw *= -1
            geo = geometry(data["raw"], natural["clean_raw"], donor_raw, rows, reference, f)
            for k, a in geo.items():
                metrics[name + "__" + k] = a
            for bi, baseline in enumerate(summary["baselines"]):
                cells = {}
                for key, mask in masks.items():
                    mult = 8 if name == "head_wording" else 1
                    s = interval(geo["score"][bi], mask, mult)
                    gain = interval(rec, mask, mult)
                    orth = float(geo["orthogonal"][bi, mask].mean())
                    margin = interval(
                        geo["recipient_distance"][bi] - geo["donor_distance"][bi], mask
                    )
                    loss = float((clean[mask, 2] - b[mask, 2]).mean())
                    valid = (
                        summary["natural"][key]["behavior_valid"]
                        and summary["natural"][key]["family_accuracy"][f][bi] >= 0.75
                    )
                    cells[key] = {
                        "score": s,
                        "recovery": gain,
                        "orthogonal": orth,
                        "donor_margin": margin,
                        "pairwise_loss": loss,
                        "axis_identifiable_and_behavior_valid": valid,
                        "switch": bool(
                            valid
                            and s["simultaneous_ci"][0] > 0.5
                            and orth <= max(s["mean"], 0)
                            and margin["mean"] > 0
                            and loss <= 0.1
                        ),
                    }
                item["geometry"][baseline] = cells
                if name == "head_wording":
                    summary["primary_wording"][baseline] = cells
        summary["conditions"][name] = item
    summary["wording_switch"] = all(
        c["switch"] for cells in summary["primary_wording"].values() for c in cells.values()
    )
    summary["all_behavior_valid"] = all(v["behavior_valid"] for v in summary["natural"].values())
    behavior = {k: v["behavior"][:, 0] for k, v in edits.items()}
    a, r, o, both = [
        behavior[k] for k in ["head_self", "head_role", "head_order", "head_order_role"]
    ]
    factorial = {
        "order_effect": (o + both - a - r) / 2,
        "role_effect": (r + both - a - o) / 2,
        "interaction": both - o - r + a,
    }
    for key, mask in masks.items():
        summary["factorial"][key] = {f: interval(x, mask, 12) for f, x in factorial.items()}
        lo, hi = summary["factorial"][key]["role_effect"]["simultaneous_ci"]
        summary["factorial"][key]["role_equivalent_within_0p2"] = bool(lo > -0.2 and hi < 0.2)
    for f, x in factorial.items():
        metrics["factorial__" + f] = x
    random = np.mean([edits[f"random_{i}"]["behavior"] for i in range(3)], axis=0)
    summary["controls"]["wording_over_random"] = interval(
        edits["head_wording"]["behavior"][:, 0] - random[:, 0]
    )
    for f in ["self", "wording", "order"]:
        trace = load_arrays(directory / "test" / ("trace_" + f + ".npz"))
        np.testing.assert_allclose(trace["base"], corrupt, atol=5e-4, rtol=0)
        np.testing.assert_allclose(
            trace["edited"], edits["head_" + f]["behavior"], atol=5e-4, rtol=0
        )
        effect = trace["edited"][:, 0] - trace["base"][:, 0]
        attenuation = effect[None] - (trace["clamped"][..., 0] - trace["clamped_base"][..., 0])
        orientation = selected["groups"][f]["orientation"]
        oriented = orientation * attenuation
        mult = 8 if f != "self" else 1
        summary["blocking"][f] = {
            "heads": selected["groups"][f],
            "total_effect": interval(effect),
            "selected_signed": interval(attenuation[0]),
            "selected_oriented": interval(oriented[0], mult=mult),
            "over_controls": [interval(oriented[0] - oriented[i], mult=mult) for i in range(1, 4)],
            "components": {},
            "cells": {},
        }
        for i, name in enumerate(
            ["z_selected", "z_control0", "z_control1", "z_control2", "query", "key", "value"]
        ):
            summary["blocking"][f]["components"][name] = interval(attenuation[i])
        for key, mask in masks.items():
            summary["blocking"][f]["cells"][key] = {
                "total_effect": interval(effect, mask),
                "selected_signed": interval(attenuation[0], mask),
            }
        record = summary["blocking"][f]
        record["adjusted_success"] = bool(
            record["selected_oriented"]["simultaneous_ci"][0] > 0
            and all(x["simultaneous_ci"][0] > 0 for x in record["over_controls"])
        )
        metrics[f + "__attenuation"] = attenuation
        group = selected["groups"][f]["indices"]
        summary["joint"][f] = {}
        for bi, baseline in enumerate(summary["baselines"]):
            interaction = []
            for ci, raw in enumerate([natural["corrupt_raw"], edits["head_" + f]["raw"]]):
                baseline_b = trace["base"] if ci == 0 else trace["edited"]
                joint_effect = baseline_b[:, 0] - trace["joint"][bi, ci, :, 0]
                summed = raw[bi, :, :, 0][:, group].sum(1)
                interaction.append(joint_effect - summed)
            summary["joint"][f][baseline] = {
                "corrupted_nonadditivity": interval(interaction[0]),
                "edited_nonadditivity": interval(interaction[1]),
                "change": interval(interaction[1] - interaction[0]),
            }
            metrics[f + "__joint_nonadditivity__" + baseline] = np.stack(interaction)
        attention = trace["attention"]
        preference = (attention[..., 0] - attention[..., 1]).mean(-1)
        summary["attention"][f] = {
            "corrupt_io_minus_subject": interval(preference[0]),
            "edited_io_minus_subject": interval(preference[1]),
            "change": interval(preference[1] - preference[0]),
        }
        metrics[f + "__attention_io_minus_subject"] = preference
    write(output / "summary.json", summary)
    save_arrays(output / "per-example.npz", **metrics)
    write(
        output / "provenance.json",
        {
            "utc": now(),
            "analyzer_sha256": digest(__file__),
            "summary_sha256": digest(output / "summary.json"),
            "test_hashes": run["hashes"],
        },
    )
    print(
        "ANALYSIS COMPLETE",
        model,
        "validity",
        summary["all_behavior_valid"],
        "wording switch",
        summary["wording_switch"],
        "blocking",
        {f: r["adjusted_success"] for f, r in summary["blocking"].items()},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--output")
    args = p.parse_args()
    main(args.model, args.output)
