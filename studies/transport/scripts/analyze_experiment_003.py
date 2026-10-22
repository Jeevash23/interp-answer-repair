"""Run the frozen template-level analysis for Experiment 003."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from circuit_transport.experiment_003 import (
    exact_sign_flip_pvalue,
    holm_adjust,
    pair_arrays_by_template,
    two_stage_bootstrap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "experiment-003"
FIGURE_ROOT = RESULT_ROOT / "figures"
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_903
PROBABILITY_DENOMINATOR_THRESHOLD = 0.01

DIRECTIONS = {
    "canonical_to_canonical": ("canonical", "canonical"),
    "controlled_to_controlled": ("controlled", "controlled"),
    "canonical_to_controlled": ("canonical", "controlled"),
    "controlled_to_canonical": ("controlled", "canonical"),
}
CONTROLS = ("random_head", "random_direction", "off_target", "mismatched_entities")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        denominator = math.copysign(1e-12, denominator if denominator else 1.0)
    return numerator / denominator


def derived_patch_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item["denominator"] = float(row["target_clean_probability_difference"]) - float(
            row["target_corrupt_probability_difference"]
        )
        item["numerator"] = float(row["target_patched_probability_difference"]) - float(
            row["target_corrupt_probability_difference"]
        )
        item["local_denominator"] = float(row["local_clean"]) - float(row["local_corrupt"])
        item["local_numerator"] = float(row["local_patched"]) - float(row["local_corrupt"])
        output.append(item)
    return output


def align_template_arrays(
    *groups: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    template_sets = [set(group) for group in groups]
    if not template_sets or any(template_set != template_sets[0] for template_set in template_sets):
        raise ValueError("Template groups are not aligned")
    output = {}
    for template in sorted(template_sets[0]):
        arrays = [group[template] for group in groups]
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError(f"Pair rows are not aligned for {template}")
        output[template] = np.concatenate(arrays, axis=1)
    return output


def behavior_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["template_id"]))].append(row)
    templates = []
    for (condition, template), group in sorted(grouped.items()):
        accuracy = float(np.mean([row["clean_io_wins_pairwise"] for row in group]))
        probability = float(np.mean([row["clean_probability_difference"] for row in group]))
        templates.append(
            {
                "condition": condition,
                "template_id": template,
                "n": len(group),
                "accuracy": accuracy,
                "probability_difference": probability,
                "gate_passed": bool(accuracy >= 0.70 and probability > 0),
            }
        )
    aggregate = []
    for condition in ("canonical", "controlled"):
        group = [row for row in rows if row["condition"] == condition]
        accuracy = float(np.mean([row["clean_io_wins_pairwise"] for row in group]))
        probability = float(np.mean([row["clean_probability_difference"] for row in group]))
        aggregate.append(
            {
                "condition": condition,
                "n": len(group),
                "accuracy": accuracy,
                "probability_difference": probability,
                "gate_passed": bool(accuracy >= 0.70 and probability > 0),
                "templates_passed": sum(
                    item["gate_passed"] for item in templates if item["condition"] == condition
                ),
                "templates_total": sum(item["condition"] == condition for item in templates),
            }
        )
    return {"aggregate": aggregate, "templates": templates}


def transport_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    site: str,
    source: str,
    target: str,
    control: str = "correct",
) -> dict[str, Any]:
    group = [
        row
        for row in rows
        if row["site"] == site
        and row["source_condition"] == source
        and row["target_condition"] == target
        and row["control"] == control
    ]
    arrays = pair_arrays_by_template(
        derived_patch_rows(group),
        ("denominator", "numerator", "local_denominator", "local_numerator"),
    )

    def statistic(values: np.ndarray) -> np.ndarray:
        denominator, numerator, local_denominator, local_numerator = values.mean(axis=0)
        return np.asarray(
            (
                denominator,
                numerator,
                safe_ratio(numerator, denominator),
                local_denominator,
                local_numerator,
                safe_ratio(local_numerator, local_denominator),
            )
        )

    result = two_stage_bootstrap(arrays, statistic, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    labels = (
        "denominator",
        "numerator",
        "cts",
        "local_denominator",
        "local_numerator",
        "local_ratio",
    )
    output = {
        label: {
            "estimate": result["point"][index],
            "ci_low": result["ci_low"][index],
            "ci_high": result["ci_high"][index],
        }
        for index, label in enumerate(labels)
    }
    output.update(
        {
            "site": site,
            "source_condition": source,
            "target_condition": target,
            "control": control,
            "templates": result["templates"],
            "pairs": result["pairs"],
            "bootstrap_samples": result["bootstrap_samples"],
            "bootstrap_seed": result["bootstrap_seed"],
            "ratio_reportable": abs(output["denominator"]["estimate"])
            >= PROBABILITY_DENOMINATOR_THRESHOLD,
        }
    )
    return output


def directional_contrast(rows: Sequence[Mapping[str, Any]], *, site: str) -> dict[str, Any]:
    def arrays(source: str, target: str) -> dict[str, np.ndarray]:
        group = [
            row
            for row in rows
            if row["site"] == site
            and row["source_condition"] == source
            and row["target_condition"] == target
            and row["control"] == "correct"
        ]
        return pair_arrays_by_template(derived_patch_rows(group), ("denominator", "numerator"))

    aligned = align_template_arrays(
        arrays("canonical", "controlled"), arrays("controlled", "canonical")
    )

    def statistic(values: np.ndarray) -> np.ndarray:
        forward_denominator, forward_numerator, reverse_denominator, reverse_numerator = (
            values.mean(axis=0)
        )
        forward = safe_ratio(forward_numerator, forward_denominator)
        reverse = safe_ratio(reverse_numerator, reverse_denominator)
        return np.asarray((forward, reverse, forward - reverse))

    result = two_stage_bootstrap(aligned, statistic, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    return {
        "forward_cts": result["point"][0],
        "reverse_cts": result["point"][1],
        "difference": result["point"][2],
        "ci_low": result["ci_low"][2],
        "ci_high": result["ci_high"][2],
        "templates": result["templates"],
        "pairs": result["pairs"],
    }


def control_analysis(rows: Sequence[Mapping[str, Any]], *, site: str) -> dict[str, Any]:
    def numerator_arrays(control: str) -> dict[str, np.ndarray]:
        group = [
            row
            for row in rows
            if row["site"] == site
            and row["source_condition"] == "canonical"
            and row["target_condition"] == "controlled"
            and row["control"] == control
        ]
        return pair_arrays_by_template(derived_patch_rows(group), ("numerator",))

    correct = numerator_arrays("correct")
    output: dict[str, Any] = {}
    pvalues = {}
    for control in CONTROLS:
        aligned = align_template_arrays(correct, numerator_arrays(control))

        def statistic(values: np.ndarray) -> np.ndarray:
            correct_mean, control_mean = values.mean(axis=0)
            return np.asarray((correct_mean, control_mean, correct_mean - control_mean))

        result = two_stage_bootstrap(
            aligned, statistic, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        )
        template_differences = np.asarray(
            [np.mean(values[:, 0] - values[:, 1]) for values in aligned.values()]
        )
        pvalue = exact_sign_flip_pvalue(template_differences)
        pvalues[control] = pvalue
        output[control] = {
            "correct_effect": result["point"][0],
            "control_effect": result["point"][1],
            "difference": result["point"][2],
            "ci_low": result["ci_low"][2],
            "ci_high": result["ci_high"][2],
            "template_differences": template_differences.tolist(),
            "pvalue": pvalue,
        }
    adjusted = holm_adjust(pvalues)
    for control, adjusted_pvalue in adjusted.items():
        output[control]["holm_pvalue"] = adjusted_pvalue
    return output


def stage_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def numerator_arrays(site: str, control: str) -> dict[str, np.ndarray]:
        group = [
            row
            for row in rows
            if row["site"] == site
            and row["source_condition"] == "canonical"
            and row["target_condition"] == "controlled"
            and row["control"] == control
        ]
        return pair_arrays_by_template(derived_patch_rows(group), ("numerator",))

    aligned = align_template_arrays(
        numerator_arrays("s2", "correct"),
        numerator_arrays("s2", "mismatched_entities"),
        numerator_arrays("end", "correct"),
        numerator_arrays("end", "mismatched_entities"),
    )

    def statistic(values: np.ndarray) -> np.ndarray:
        s2_correct, s2_mismatch, end_correct, end_mismatch = values.mean(axis=0)
        s2_retention = safe_ratio(s2_mismatch, s2_correct)
        end_retention = safe_ratio(end_mismatch, end_correct)
        s2_contrast = s2_correct - s2_mismatch
        end_contrast = end_correct - end_mismatch
        return np.asarray(
            (
                s2_retention,
                end_retention,
                s2_retention - end_retention,
                s2_contrast,
                end_contrast,
                end_contrast - s2_contrast,
            )
        )

    result = two_stage_bootstrap(aligned, statistic, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    labels = (
        "s2_retention",
        "end_retention",
        "retention_difference",
        "s2_correct_minus_mismatch",
        "end_correct_minus_mismatch",
        "contrast_difference",
    )
    return {
        label: {
            "estimate": result["point"][index],
            "ci_low": result["ci_low"][index],
            "ci_high": result["ci_high"][index],
        }
        for index, label in enumerate(labels)
    }


def split_analysis(split: str) -> dict[str, Any]:
    behavior_rows = load_jsonl(RESULT_ROOT / split / "behavior-per-example.jsonl")
    patch_rows = load_jsonl(RESULT_ROOT / split / "patches-per-example.jsonl")
    behavior = behavior_analysis(behavior_rows)
    transport = {
        site: {
            name: transport_group(patch_rows, site=site, source=source, target=target)
            for name, (source, target) in DIRECTIONS.items()
        }
        for site in ("s2", "end")
    }
    directional = {site: directional_contrast(patch_rows, site=site) for site in ("s2", "end")}
    controls = {site: control_analysis(patch_rows, site=site) for site in ("s2", "end")}
    stage = stage_analysis(patch_rows)
    aggregate_gates = {row["condition"]: row for row in behavior["aggregate"]}
    test_predictions = {
        "behavioral_generalization": bool(
            all(row["gate_passed"] for row in behavior["aggregate"])
            and aggregate_gates["controlled"]["templates_passed"] >= 6
        ),
        "transport_generalization": bool(
            all(
                transport[site]["canonical_to_controlled"]["numerator"]["ci_low"] > 0
                for site in ("s2", "end")
            )
        ),
        "directional_prediction": bool(
            all(directional[site]["ci_low"] > 0 for site in ("s2", "end"))
        ),
        "end_specificity": bool(
            all(
                item["difference"] > 0 and item["holm_pvalue"] < 0.05
                for item in controls["end"].values()
            )
        ),
        "stage_prediction": bool(
            stage["retention_difference"]["ci_low"] > 0
            and stage["contrast_difference"]["ci_low"] > 0
        ),
    }
    return {
        "behavior": behavior,
        "transport": transport,
        "directional": directional,
        "controls": controls,
        "stage": stage,
        "frozen_predictions": test_predictions,
    }


def make_figures(analysis: Mapping[str, Any]) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    test = analysis["test"]

    behavior_rows = test["behavior"]["templates"]
    template_names = sorted({row["template_id"].removeprefix("test-") for row in behavior_rows})
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    positions = np.arange(len(template_names))
    for offset, condition, color in (
        (-0.15, "canonical", "#4c78a8"),
        (0.15, "controlled", "#f58518"),
    ):
        values = [
            next(
                row["accuracy"]
                for row in behavior_rows
                if row["condition"] == condition
                and row["template_id"].removeprefix("test-") == template
            )
            for template in template_names
        ]
        ax.bar(positions + offset, values, width=0.28, label=condition, color=color)
    ax.axhline(0.70, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Clean pairwise accuracy")
    ax.set_xticks(positions, template_names, rotation=30, ha="right")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "multitemplate-behavior.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "multitemplate-behavior.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    patch_rows = load_jsonl(RESULT_ROOT / "test" / "patches-per-example.jsonl")
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
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
        arrays = pair_arrays_by_template(derived_patch_rows(group), ("numerator",))
        values = [float(array.mean()) for array in arrays.values()]
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), site_index) + jitter,
            values,
            color=colors[site],
            label=site.upper(),
            zorder=3,
        )
        ax.hlines(
            np.mean(values), site_index - 0.18, site_index + 0.18, color="black", linewidth=1.5
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks((0, 1), ("S2", "END"))
    ax.set_ylabel("Template-mean probability effect")
    ax.set_title("Held-out canonical-to-controlled transport")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "multitemplate-transport.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "multitemplate-transport.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment 003 analysis",
        "",
        "## Frozen test predictions",
        "",
        "| Prediction | Result |",
        "|---|---|",
    ]
    for key, passed in analysis["test"]["frozen_predictions"].items():
        lines.append(f"| {key.replace('_', ' ').title()} | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            "## Held-out behavior",
            "",
            "| Condition | Accuracy | Probability difference | Templates passing |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in analysis["test"]["behavior"]["aggregate"]:
        lines.append(
            f"| {row['condition']} | {row['accuracy']:.3f} | "
            f"{row['probability_difference']:.4f} | "
            f"{row['templates_passed']}/{row['templates_total']} |"
        )
    lines.extend(
        [
            "",
            "## Held-out directional transport",
            "",
            "| Site | Canonical to controlled CTS | Controlled to canonical CTS | Difference [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for site in ("s2", "end"):
        row = analysis["test"]["directional"][site]
        lines.append(
            f"| {site.upper()} | {row['forward_cts']:.3f} | {row['reverse_cts']:.3f} | "
            f"{row['difference']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |"
        )
    stage = analysis["test"]["stage"]
    lines.extend(
        [
            "",
            "## Stage specificity",
            "",
            (
                f"Mismatch retention is {stage['s2_retention']['estimate']:.3f} at S2 and "
                f"{stage['end_retention']['estimate']:.3f} at END. The predeclared retention "
                f"difference is {stage['retention_difference']['estimate']:.3f} "
                f"[{stage['retention_difference']['ci_low']:.3f}, "
                f"{stage['retention_difference']['ci_high']:.3f}]."
            ),
            "",
            "## Reproducibility",
            "",
            (
                "All estimates use the frozen manifest, Medium head groups, controls, pair "
                "structure, and template-level inference. Exact hashes and independent rerun "
                "checks are recorded in `verification.json`."
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
        },
        "development": split_analysis("development"),
        "test": split_analysis("test"),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULT_ROOT / "ANALYSIS.md").write_text(markdown_report(analysis), encoding="utf-8")
    make_figures(analysis)
    print(markdown_report(analysis))


if __name__ == "__main__":
    main()
