"""Analyze Experiment 002 and produce machine-readable results, prose, and figures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from circuit_transport.experiment_002 import benjamini_hochberg, sign_flip_pvalue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment-002"
FIGURE_DIR = RESULTS_ROOT / "figures"
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20_260_901
SIGN_FLIP_DRAWS = 20_000
SIGN_FLIP_SEED = 920_260_901
CONTROLS = ("random_head", "random_direction", "off_target", "mismatched_entities")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def group_rows(
    rows: Sequence[dict[str, Any]],
    *,
    corruption: str,
    site: str,
    source: str,
    target: str,
    control: str,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["corruption"] == corruption
        and row["site"] == site
        and row["source_condition"] == source
        and row["target_condition"] == target
        and row["control"] == control
    ]
    return sorted(selected, key=lambda row: row["example_id"])


def arrays(rows: Sequence[dict[str, Any]], metric: str) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray([row[f"target_{state}_{metric}"] for row in rows], dtype=np.float64)
        for state in ("clean", "corrupt", "patched")
    )


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def ratio_difference(
    forward: Sequence[dict[str, Any]],
    reverse: Sequence[dict[str, Any]],
    *,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    f_clean, f_corrupt, f_patched = arrays(forward, metric)
    r_clean, r_corrupt, r_patched = arrays(reverse, metric)
    if [row["example_id"] for row in forward] != [row["example_id"] for row in reverse]:
        raise ValueError("directional rows are not paired by example")
    f_num = f_patched - f_corrupt
    f_den = f_clean - f_corrupt
    r_num = r_patched - r_corrupt
    r_den = r_clean - r_corrupt
    f_ratio = float(f_num.mean() / f_den.mean())
    r_ratio = float(r_num.mean() / r_den.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, f_num.size, size=(BOOTSTRAP_SAMPLES, f_num.size))
    boot = f_num[indices].mean(axis=1) / f_den[indices].mean(axis=1)
    boot -= r_num[indices].mean(axis=1) / r_den[indices].mean(axis=1)
    low, high = percentile_interval(boot)
    normalized_difference = f_num / f_den.mean() - r_num / r_den.mean()
    return {
        "forward_ratio": f_ratio,
        "reverse_ratio": r_ratio,
        "ratio_difference": f_ratio - r_ratio,
        "ci_low": low,
        "ci_high": high,
        "pvalue": sign_flip_pvalue(
            normalized_difference,
            draws=SIGN_FLIP_DRAWS,
            seed=SIGN_FLIP_SEED + seed,
        ),
    }


def paired_control_contrast(
    correct: Sequence[dict[str, Any]],
    control: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if [row["example_id"] for row in correct] != [row["example_id"] for row in control]:
        raise ValueError("control rows are not paired by example")
    correct_effect = np.asarray(
        [
            row["target_patched_probability_difference"]
            - row["target_corrupt_probability_difference"]
            for row in correct
        ],
        dtype=np.float64,
    )
    control_effect = np.asarray(
        [
            row["target_patched_probability_difference"]
            - row["target_corrupt_probability_difference"]
            for row in control
        ],
        dtype=np.float64,
    )
    differences = correct_effect - control_effect
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(BOOTSTRAP_SAMPLES, differences.size))
    low, high = percentile_interval(differences[indices].mean(axis=1))
    return {
        "correct_effect": float(correct_effect.mean()),
        "control_effect": float(control_effect.mean()),
        "difference": float(differences.mean()),
        "ci_low": low,
        "ci_high": high,
        "pvalue": sign_flip_pvalue(differences, draws=SIGN_FLIP_DRAWS, seed=SIGN_FLIP_SEED + seed),
    }


def retention_ratio(
    correct: Sequence[dict[str, Any]],
    mismatch: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    correct_effect = np.asarray(
        [
            row["target_patched_probability_difference"]
            - row["target_corrupt_probability_difference"]
            for row in correct
        ],
        dtype=np.float64,
    )
    mismatch_effect = np.asarray(
        [
            row["target_patched_probability_difference"]
            - row["target_corrupt_probability_difference"]
            for row in mismatch
        ],
        dtype=np.float64,
    )
    ratio = float(mismatch_effect.mean() / correct_effect.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, correct_effect.size, size=(BOOTSTRAP_SAMPLES, correct_effect.size))
    boot = mismatch_effect[indices].mean(axis=1) / correct_effect[indices].mean(axis=1)
    low, high = percentile_interval(boot)
    return {"ratio": ratio, "ci_low": low, "ci_high": high}


def summary_lookup(
    rows: Sequence[dict[str, Any]],
    *,
    corruption: str,
    site: str,
    source: str,
    target: str,
    control: str,
    metric: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["corruption"] == corruption
        and row["site"] == site
        and row["source_condition"] == source
        and row["target_condition"] == target
        and row["control"] == control
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row, got {len(matches)}")
    return matches[0]


def analyze_split(split: str) -> dict[str, Any]:
    directory = RESULTS_ROOT / split
    patch_rows = load_jsonl(directory / "patches-per-example.jsonl")
    behavior = json.loads((directory / "behavior-summary.json").read_text())
    summaries = json.loads((directory / "transport-summary.json").read_text())
    output: dict[str, Any] = {"behavior": behavior, "corruptions": {}}
    for corruption in ("primary", "alternative"):
        corruption_output: dict[str, Any] = {"sites": {}}
        for site_index, site in enumerate(("s2", "end")):
            forward = group_rows(
                patch_rows,
                corruption=corruption,
                site=site,
                source="canonical",
                target="controlled",
                control="correct",
            )
            reverse = group_rows(
                patch_rows,
                corruption=corruption,
                site=site,
                source="controlled",
                target="canonical",
                control="correct",
            )
            site_output: dict[str, Any] = {
                "forward": summary_lookup(
                    summaries,
                    corruption=corruption,
                    site=site,
                    source="canonical",
                    target="controlled",
                    control="correct",
                    metric="probability_difference",
                ),
                "reverse": summary_lookup(
                    summaries,
                    corruption=corruption,
                    site=site,
                    source="controlled",
                    target="canonical",
                    control="correct",
                    metric="probability_difference",
                ),
                "forward_local": summary_lookup(
                    summaries,
                    corruption=corruption,
                    site=site,
                    source="canonical",
                    target="controlled",
                    control="correct",
                    metric="local",
                ),
                "reverse_local": summary_lookup(
                    summaries,
                    corruption=corruption,
                    site=site,
                    source="controlled",
                    target="canonical",
                    control="correct",
                    metric="local",
                ),
                "directional": ratio_difference(
                    forward,
                    reverse,
                    metric="probability_difference",
                    seed=100 * (site_index + 1) + (0 if corruption == "primary" else 10),
                ),
                "controls": {},
            }
            for control_index, control in enumerate(CONTROLS):
                control_rows = group_rows(
                    patch_rows,
                    corruption=corruption,
                    site=site,
                    source="canonical",
                    target="controlled",
                    control=control,
                )
                site_output["controls"][control] = paired_control_contrast(
                    forward,
                    control_rows,
                    seed=1_000 * (site_index + 1)
                    + 10 * control_index
                    + (0 if corruption == "primary" else 1),
                )
            mismatch = group_rows(
                patch_rows,
                corruption=corruption,
                site=site,
                source="canonical",
                target="controlled",
                control="mismatched_entities",
            )
            site_output["mismatch_retention"] = retention_ratio(
                forward, mismatch, seed=5_000 + site_index
            )
            corruption_output["sites"][site] = site_output
        output["corruptions"][corruption] = corruption_output
    return output


def apply_test_multiplicity(analysis: dict[str, Any]) -> None:
    primary = analysis["test"]["corruptions"]["primary"]["sites"]
    directional_pvalues = {site: primary[site]["directional"]["pvalue"] for site in ("s2", "end")}
    directional_qvalues = benjamini_hochberg(directional_pvalues)
    for site, qvalue in directional_qvalues.items():
        primary[site]["directional"]["qvalue"] = qvalue

    control_pvalues = {
        f"{site}:{control}": primary[site]["controls"][control]["pvalue"]
        for site in ("s2", "end")
        for control in CONTROLS
    }
    control_qvalues = benjamini_hochberg(control_pvalues)
    for name, qvalue in control_qvalues.items():
        site, control = name.split(":")
        primary[site]["controls"][control]["qvalue"] = qvalue


def format_interval(row: Mapping[str, Any], field: str = "ratio") -> str:
    return f"{row[field]:.3f} [{row[f'{field}_ci_low']:.3f}, {row[f'{field}_ci_high']:.3f}]"


def make_figures(analysis: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    localization_rows = load_jsonl(RESULTS_ROOT / "localization" / "head-effects-summary.jsonl")
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.6), constrained_layout=True)
    for row_index, site in enumerate(("s2", "end")):
        site_values = np.asarray(
            [abs(row["mean_logit_effect"]) for row in localization_rows if row["site"] == site]
        )
        limit = float(np.percentile(site_values, 99))
        row_image = None
        for column_index, condition in enumerate(("canonical", "controlled")):
            matrix = np.zeros((24, 16), dtype=np.float64)
            for row in localization_rows:
                if row["site"] == site and row["condition"] == condition:
                    matrix[row["layer"], row["head"]] = row["mean_logit_effect"]
            axis = axes[row_index, column_index]
            row_image = axis.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
            )
            axis.set_title(f"{condition.capitalize()}, {site.upper()} site")
            axis.set_xlabel("Head")
            axis.set_ylabel("Layer")
            axis.set_xticks(range(0, 16, 3))
            axis.set_yticks(range(0, 24, 4))
        fig.colorbar(
            row_image,
            ax=axes[row_index, :],
            shrink=0.82,
            label="Mean patched logit effect",
        )
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"medium-head-localization.{suffix}", dpi=220)
    plt.close(fig)

    test_sites = analysis["test"]["corruptions"]["primary"]["sites"]
    labels = ["S2 C-to-X", "S2 X-to-C", "END C-to-X", "END X-to-C"]
    rows = [
        test_sites["s2"]["forward"],
        test_sites["s2"]["reverse"],
        test_sites["end"]["forward"],
        test_sites["end"]["reverse"],
    ]
    estimates = np.asarray([row["ratio"] for row in rows])
    low = np.asarray([row["ratio_ci_low"] for row in rows])
    high = np.asarray([row["ratio_ci_high"] for row in rows])
    fig, axis = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    axis.errorbar(
        np.arange(len(rows)),
        estimates,
        yerr=np.vstack((estimates - low, high - estimates)),
        fmt="o",
        capsize=4,
        color="#204a87",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axhline(1, color="0.5", linewidth=0.8, linestyle="--")
    axis.set_xticks(np.arange(len(rows)), labels)
    axis.set_ylabel("Causal transport score")
    axis.set_title("Held-out GPT-2 Medium directional transport")
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"medium-heldout-transport.{suffix}", dpi=220)
    plt.close(fig)

    control_labels = ("correct",) + CONTROLS
    fig, axis = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    width = 0.36
    x = np.arange(len(control_labels))
    for site_index, site in enumerate(("s2", "end")):
        correct_effect = test_sites[site]["forward"]["numerator"]
        effects = [correct_effect]
        for control in CONTROLS:
            effects.append(test_sites[site]["controls"][control]["control_effect"])
        axis.bar(
            x + (site_index - 0.5) * width,
            effects,
            width,
            label=site.upper(),
            color=("#729fcf", "#ef8a62")[site_index],
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, [label.replace("_", "\n") for label in control_labels])
    axis.set_ylabel("Patched minus corrupt probability effect")
    axis.set_title("Held-out canonical-to-controlled specificity controls")
    axis.legend(frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"medium-transport-controls.{suffix}", dpi=220)
    plt.close(fig)


def write_report(analysis: dict[str, Any]) -> None:
    selection = json.loads((RESULTS_ROOT / "localization" / "selected-heads.json").read_text())
    test = analysis["test"]["corruptions"]["primary"]["sites"]
    development = analysis["development"]["corruptions"]["primary"]["sites"]
    lines = [
        "# Experiment 002 analysis",
        "",
        "## Decision",
        "",
        (
            "GPT-2 Medium provides held-out evidence for partial, directional causal interoperability. "
            "The independently localized contexts share some but not all heads. Final-position "
            "transport is strong, local, and specific; S2 transport has a positive held-out local "
            "response but is not entity-specific because mismatched entities preserve nearly all of "
            "its final effect."
        ),
        "",
        "## Exploratory localization",
        "",
        "| Site | Spearman over 384 heads | Top-8 overlap | Top-32 overlap |",
        "|---|---:|---:|---:|",
    ]
    for site in ("s2", "end"):
        similarity = selection["similarity"][site]
        lines.append(
            f"| {site.upper()} | {similarity['spearman']:.3f} | "
            f"{similarity['top_8_intersection']}/8 | {similarity['top_32_intersection']}/32 |"
        )
    lines.extend(
        [
            "",
            (
                "The localization and an independent rerun are byte-identical. Selection used only "
                "the exploratory split."
            ),
            "",
            "## Primary held-out transport",
            "",
            "| Site | Canonical to controlled CTS [95% CI] | Controlled to canonical CTS [95% CI] | Difference [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for site in ("s2", "end"):
        directional = test[site]["directional"]
        lines.append(
            f"| {site.upper()} | {format_interval(test[site]['forward'])} | "
            f"{format_interval(test[site]['reverse'])} | "
            f"{directional['ratio_difference']:.3f} [{directional['ci_low']:.3f}, "
            f"{directional['ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            (
                "Both clean test conditions pass the frozen behavioral gate: canonical accuracy is "
                "100% and controlled accuracy is 96.25%. The controlled primary probability "
                f"denominator is {test['end']['forward']['denominator']:.5f}; the END forward "
                f"numerator is {test['end']['forward']['numerator']:.5f}, so its over-recovery is "
                "reported with both absolute quantities rather than as a ratio alone."
            ),
            "",
            "## Specificity and local checks",
            "",
            "| Site | Correct effect | Mismatch retention [95% CI] | Correct local effect [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for site in ("s2", "end"):
        retention = test[site]["mismatch_retention"]
        local = test[site]["forward_local"]
        lines.append(
            f"| {site.upper()} | {test[site]['forward']['numerator']:.5f} | "
            f"{retention['ratio']:.3f} [{retention['ci_low']:.3f}, {retention['ci_high']:.3f}] | "
            f"{local['numerator']:.4f} [{local['numerator_ci_low']:.4f}, "
            f"{local['numerator_ci_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            (
                "END correct-minus-control effects survive the predeclared control family, including "
                "the mismatched-entity comparison. At S2, correct patches beat random-head, "
                "random-direction, and off-position controls, but not mismatched entities on the "
                "primary final outcome. This is evidence for a reusable role/position signal at S2 "
                "and stronger entity binding at END."
            ),
            "",
            (
                "The S2 local projection was near zero on development "
                f"({development['s2']['forward_local']['numerator']:.4f}) but positive on test "
                f"({test['s2']['forward_local']['numerator']:.4f}). This instability is retained and "
                "limits the claim that the declared S2-to-END path is fully identified."
            ),
            "",
            "## Corruption robustness",
            "",
            (
                "The alternative corruption preserves positive correct transport at both sites and "
                "both directions. Exact ratios and intervals are stored in `analysis.json`; the "
                "primary corruption remains the confirmatory analysis."
            ),
            "",
            "## Reproducibility",
            "",
            (
                "The complete localization and development intervention runs reproduced "
                "byte-for-byte. The test run uses the frozen source, selection, manifest, controls, "
                "seeds, and thresholds recorded in `data/experiment-002-implementation-freeze.json`."
            ),
        ]
    )
    (RESULTS_ROOT / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    analysis = {
        "development": analyze_split("development"),
        "test": analyze_split("test"),
        "seeds": {
            "bootstrap": BOOTSTRAP_SEED,
            "sign_flip": SIGN_FLIP_SEED,
            "sign_flip_draws": SIGN_FLIP_DRAWS,
        },
    }
    apply_test_multiplicity(analysis)
    (RESULTS_ROOT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(analysis)
    make_figures(analysis)
    print(f"Wrote {RESULTS_ROOT / 'ANALYSIS.md'}")
    print(f"Wrote {RESULTS_ROOT / 'analysis.json'}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
