"""Analyze the frozen Pythia cross-architecture transport study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import analyze_experiment_003 as shared
import matplotlib.pyplot as plt
import numpy as np

from circuit_transport.experiment_003 import pair_arrays_by_template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "experiment-004"
FIGURE_ROOT = RESULT_ROOT / "figures"
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_904


def split_analysis(split: str) -> dict[str, Any]:
    shared.BOOTSTRAP_SAMPLES = BOOTSTRAP_SAMPLES
    shared.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    behavior_rows = shared.load_jsonl(RESULT_ROOT / split / "behavior-per-example.jsonl")
    patch_rows = shared.load_jsonl(RESULT_ROOT / split / "patches-per-example.jsonl")
    behavior = shared.behavior_analysis(behavior_rows)
    transport = {
        site: {
            name: shared.transport_group(patch_rows, site=site, source=source, target=target)
            for name, (source, target) in shared.DIRECTIONS.items()
        }
        for site in ("s2", "end")
    }
    directional = {
        site: shared.directional_contrast(patch_rows, site=site) for site in ("s2", "end")
    }
    controls = {site: shared.control_analysis(patch_rows, site=site) for site in ("s2", "end")}
    stage = shared.stage_analysis(patch_rows)

    end_endpoint = transport["end"]["canonical_to_controlled"]["numerator"]["ci_low"] > 0
    end_direction = directional["end"]["ci_low"] > 0
    end_controls = all(
        row["ci_low"] > 0 and row["holm_pvalue"] < 0.05 for row in controls["end"].values()
    )
    end_local = transport["end"]["canonical_to_controlled"]["local_numerator"]["ci_low"] > 0
    s2_path = (
        transport["s2"]["canonical_to_controlled"]["numerator"]["ci_low"] > 0
        and transport["s2"]["canonical_to_controlled"]["local_numerator"]["ci_low"] > 0
    )
    stage_extension = (
        stage["retention_difference"]["ci_low"] > 0 and stage["contrast_difference"]["ci_low"] > 0
    )
    gates = {
        "end_endpoint_transport": bool(end_endpoint),
        "end_directional_asymmetry": bool(end_direction),
        "end_control_specificity": bool(end_controls),
        "end_local_mechanistic_sign": bool(end_local),
        "full_cross_architecture_replication": bool(
            end_endpoint and end_direction and end_controls and end_local
        ),
        "s2_path_level_gate": bool(s2_path),
        "stage_extension": bool(stage_extension),
    }
    return {
        "behavior": behavior,
        "transport": transport,
        "directional": directional,
        "controls": controls,
        "stage": stage,
        "frozen_gates": gates,
    }


def make_figures(analysis: dict[str, Any]) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    localization = shared.load_jsonl(RESULT_ROOT / "localization" / "head-effects-summary.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    limit = max(
        abs(float(row["mean_logit_effect"])) for row in localization if row["site"] == "end"
    )
    for ax, condition in zip(axes, ("canonical", "controlled"), strict=True):
        matrix = np.zeros((24, 16))
        for row in localization:
            if row["site"] == "end" and row["condition"] == condition:
                matrix[int(row["layer"]), int(row["head"])] = float(row["mean_logit_effect"])
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(condition.title())
        ax.set_xlabel("Head")
        ax.set_xticks((0, 3, 7, 11, 15))
    axes[0].set_ylabel("Layer")
    fig.colorbar(image, ax=axes, label="Mean logit restoration", shrink=0.82)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.13, top=0.88, wspace=0.15)
    fig.savefig(FIGURE_ROOT / "pythia-end-localization.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "pythia-end-localization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    patch_rows = shared.load_jsonl(RESULT_ROOT / "test" / "patches-per-example.jsonl")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    colors = {"s2": "#4c78a8", "end": "#f58518"}
    for site_index, site in enumerate(("s2", "end")):
        group = [
            row
            for row in patch_rows
            if row["site"] == site
            and row["source_condition"] == "canonical"
            and row["target_condition"] == "controlled"
            and row["control"] == "correct"
        ]
        arrays = pair_arrays_by_template(shared.derived_patch_rows(group), ("numerator",))
        values = [float(array.mean()) for array in arrays.values()]
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), site_index) + jitter,
            values,
            color=colors[site],
            zorder=3,
        )
        ax.hlines(np.mean(values), site_index - 0.18, site_index + 0.18, color="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks((0, 1), ("S2", "END"))
    ax.set_ylabel("Template-mean probability effect")
    ax.set_title("Pythia held-out canonical-to-controlled transport")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "pythia-heldout-transport.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "pythia-heldout-transport.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_report(analysis: dict[str, Any]) -> str:
    test = analysis["test"]
    lines = [
        "# Experiment 004 analysis",
        "",
        "## Frozen held-out gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, passed in test["frozen_gates"].items():
        lines.append(f"| {name.replace('_', ' ').title()} | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Held-out directional transport",
            "",
            "| Site | Canonical→controlled CTS | Controlled→canonical CTS | Difference [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for site in ("s2", "end"):
        row = test["directional"][site]
        lines.append(
            f"| {site.upper()} | {row['forward_cts']:.3f} | {row['reverse_cts']:.3f} | "
            f"{row['difference']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Held-out local mechanistic gates",
            "",
            "| Site | Endpoint numerator [95% CI] | Local numerator [95% CI] | Path gate |",
            "|---|---:|---:|---|",
        ]
    )
    for site in ("s2", "end"):
        row = test["transport"][site]["canonical_to_controlled"]
        endpoint = row["numerator"]
        local = row["local_numerator"]
        gate = endpoint["ci_low"] > 0 and local["ci_low"] > 0
        lines.append(
            f"| {site.upper()} | {endpoint['estimate']:.4f} "
            f"[{endpoint['ci_low']:.4f}, {endpoint['ci_high']:.4f}] | "
            f"{local['estimate']:.4f} [{local['ci_low']:.4f}, {local['ci_high']:.4f}] | "
            f"{'pass' if gate else 'fail'} |"
        )
    stage = test["stage"]
    lines.extend(
        [
            "",
            "## Stage extension",
            "",
            (
                f"Mismatch retention is {stage['s2_retention']['estimate']:.3f} at S2 and "
                f"{stage['end_retention']['estimate']:.3f} at END. Their preregistered "
                f"difference is {stage['retention_difference']['estimate']:.3f} "
                f"[{stage['retention_difference']['ci_low']:.3f}, "
                f"{stage['retention_difference']['ci_high']:.3f}]."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    analysis = {
        "protocol": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "sampling_unit": "templates, then adjacent ABBA/BABA entity pairs",
            "control_test": "exact template-level sign flip with Holm correction",
            "primary_replication_site": "end",
        },
        "development": split_analysis("development"),
        "test": split_analysis("test"),
    }
    (RESULT_ROOT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = markdown_report(analysis)
    (RESULT_ROOT / "ANALYSIS.md").write_text(report, encoding="utf-8")
    make_figures(analysis)
    print(report)


if __name__ == "__main__":
    main()
