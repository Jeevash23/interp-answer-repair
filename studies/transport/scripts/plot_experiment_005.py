"""Plot verified, complete Study 005 estimates without recomputing inference.

The figure uses only the final analyzer's raw probability-gain contrasts and
pointwise whole-template bootstrap intervals. No partial or smoke data are
accepted, and exact plotted estimates and input/output hashes are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "experiment-005"
MODELS = ("gpt2-medium", "pythia-410m-deduped")
LABELS = {
    "gpt2-medium": "GPT-2 Medium",
    "pythia-410m-deduped": "Pythia-410M-deduped",
}
COLORS = {"gpt2-medium": "#0072B2", "pythia-410m-deduped": "#D55E00"}
MARKERS = {"gpt2-medium": "o", "pythia-410m-deduped": "s"}
METRIC = "probability_difference"
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_905
STEM = "study-005-robustness"

CAPTION = (
    "Study 5 post-hoc robustness on eight previously examined templates per model "
    "(160 examples; all retained). Points show mean raw gain contrasts in the answer "
    "score P(IO) − P(subject); horizontal bars show pointwise 95% intervals from "
    "5,000 whole-template bootstrap resamples. (a) END gain for canonical→controlled "
    "minus controlled→canonical transfer. Native runs use source-specific groups; "
    "union runs use the same head union in both directions, with concatenated "
    "per-head result-delta norms matched to their geometric mean and multiplied by "
    "α. Native-versus-union differences combine changed group membership and norm "
    "equalization, so they do not isolate dose. (b) Canonical→controlled correct gain "
    "minus the mean gain from five "
    "layer-matched sampled-head sets or five per-head norm-matched random residual "
    "directions, averaged within example before resampling. (c) Attenuation of "
    "canonical S2 transfer under the selected controlled END-node clamp minus "
    "mean attenuation under five layer-matched random clamps. Positive values in "
    "(c) indicate greater attenuation under the selected clamp; this is a "
    "controlled node-dependence diagnostic, not isolation of a causal path. "
    "Intervals are exploratory, unadjusted for multiplicity, and conditional on "
    "the fixed selected head groups. Panels use separate horizontal scales."
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_estimate(estimate: Mapping[str, Any], *, label: str) -> None:
    if estimate.get("n_examples") != 160 or estimate.get("n_templates") != 8:
        raise ValueError(f"Incomplete estimate: {label}")
    if (
        estimate.get("bootstrap_samples") != BOOTSTRAP_SAMPLES
        or estimate.get("bootstrap_seed") != BOOTSTRAP_SEED
        or estimate.get("valid_bootstrap_draws") != BOOTSTRAP_SAMPLES
        or estimate.get("interval_note") is not None
    ):
        raise ValueError(f"Unexpected or suppressed interval: {label}")
    point, ci = estimate.get("mean"), estimate.get("ci95")
    if (
        not isinstance(point, (int, float))
        or not math.isfinite(point)
        or not isinstance(ci, list)
        or len(ci) != 2
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in ci)
        or ci[0] > ci[1]
    ):
        raise ValueError(f"Nonfinite or malformed estimate: {label}")
    templates = estimate.get("per_template", {})
    if len(templates) != 8 or not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in templates.values()
    ):
        raise ValueError(f"Incomplete template estimates: {label}")
    if not math.isclose(sum(templates.values()) / 8, point, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Point estimate is not the equal-template mean: {label}")
    template_means = np.asarray([templates[key] for key in sorted(templates)], dtype=float)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 8, size=(BOOTSTRAP_SAMPLES, 8))
    recomputed_ci = np.percentile(template_means[draws].mean(axis=1), (2.5, 97.5))
    if not np.allclose(recomputed_ci, ci, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Interval does not reproduce from retained whole-template means: {label}")


def load_verified(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("study") != "005" or data.get("status") != (
        "completed descriptive post-hoc robustness analysis"
    ):
        raise ValueError("A completed Study 005 analysis is required")
    if set(data.get("models", {})) != set(MODELS):
        raise ValueError("Both complete model analyses are required")
    if data.get("analysis_script_sha256") != sha(ROOT / "scripts/analyze_experiment_005.py"):
        raise ValueError("The analysis implementation changed after these estimates were saved")
    inference = data.get("inference", {})
    if (
        inference.get("unit") != "whole template"
        or inference.get("bootstrap_samples") != BOOTSTRAP_SAMPLES
        or inference.get("bootstrap_seed") != BOOTSTRAP_SEED
        or inference.get("confirmatory_pvalues") is not False
    ):
        raise ValueError("Unexpected inferential specification")
    for model in MODELS:
        summary = data["models"][model]
        if (
            summary.get("model") != model
            or summary.get("n_examples") != 160
            or summary.get("n_templates") != 8
            or summary.get("n_specs") != 56
            or summary.get("n_rows") != 8960
        ):
            raise ValueError(f"Incomplete model summary: {model}")
        verification = data.get("run_verification", {}).get(model, {})
        if verification.get("status") != "passed":
            raise ValueError(f"Run has not passed independent verification: {model}")
        entry = data["input_files"][model]
        results_path = Path(entry["path"])
        if sha(results_path) != entry["sha256"]:
            raise ValueError(f"Results changed after analysis: {model}")
        config_path = results_path.parent / "run-config.json"
        config = json.loads(config_path.read_text())
        if (
            config.get("status") != "complete"
            or config.get("smoke") is not False
            or config.get("output_sha256") != entry["sha256"]
            or verification.get("output_sha256") != entry["sha256"]
            or verification.get("run_config_sha256") != sha(config_path)
        ):
            raise ValueError(f"Run config is incomplete or changed after verification: {model}")
    return data


def plotted_estimates(data: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    panels: dict[str, list[dict[str, Any]]] = {"directionality": [], "controls": [], "blocking": []}
    for model in MODELS:
        values = data["models"][model]
        for row, key in enumerate(
            ("native_end", "matched_end_alpha_0.5", "matched_end_alpha_1", "matched_end_alpha_2")
        ):
            estimate = values["directional_contrasts"][key][METRIC]["gain_forward_minus_reverse"]
            validate_estimate(estimate, label=f"{model}/{key}")
            panels["directionality"].append(
                {"model": model, "row": row, "contrast": key, "estimate": estimate}
            )
        for row, (site, control) in enumerate(
            (
                ("s2", "random_heads"),
                ("s2", "random_direction"),
                ("end", "random_heads"),
                ("end", "random_direction"),
            )
        ):
            estimate = values["controls"][site][control][METRIC]["correct_minus_control_gain"]
            validate_estimate(estimate, label=f"{model}/{site}/{control}")
            panels["controls"].append(
                {"model": model, "row": row, "site": site, "control": control, "estimate": estimate}
            )
        estimate = values["blocking"][METRIC]["selected_minus_mean_random_attenuation"]
        validate_estimate(estimate, label=f"{model}/blocking")
        panels["blocking"].append({"model": model, "estimate": estimate})
    return panels


def draw_interval(ax: Any, *, y: float, model: str, estimate: Mapping[str, Any]) -> None:
    low, high = estimate["ci95"]
    # Draw bars directly: a percentile interval need not straddle the point estimate.
    ax.hlines(y, low, high, color=COLORS[model], linewidth=1.8, zorder=3)
    ax.vlines((low, high), y - 0.055, y + 0.055, color=COLORS[model], linewidth=1.35, zorder=3)
    ax.plot(
        estimate["mean"],
        y,
        marker=MARKERS[model],
        color=COLORS[model],
        markersize=6.2,
        markeredgecolor="white",
        markeredgewidth=0.65,
        linestyle="none",
        zorder=4,
    )


def style_axis(ax: Any, *, entries: list[dict[str, Any]]) -> None:
    endpoints = [0.0]
    for entry in entries:
        endpoints.extend(entry["estimate"]["ci95"])
        endpoints.append(entry["estimate"]["mean"])
    low, high = min(endpoints), max(endpoints)
    span = max(high - low, 0.01)
    ax.set_xlim(low - 0.11 * span, high + 0.11 * span)
    ax.axvline(0, color="#777777", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9A9A9A")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", color="#9A9A9A", length=3.5, pad=4)
    ax.set_xlabel("Probability-difference units", labelpad=8)


def render(panels: Mapping[str, list[dict[str, Any]]]) -> Any:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.2,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.2,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.75))
    fig.subplots_adjust(left=0.118, right=0.983, top=0.765, bottom=0.245, wspace=0.68)
    offsets = {MODELS[0]: 0.13, MODELS[1]: -0.13}
    for ax, name, labels, title in (
        (
            axes[0],
            "directionality",
            ("Native", "Union, α = 0.5", "Union, α = 1", "Union, α = 2"),
            "a  END directionality\nForward − reverse gain",
        ),
        (
            axes[1],
            "controls",
            ("S2: heads", "S2: direction", "END: heads", "END: direction"),
            "b  Randomized controls\nCorrect − mean control",
        ),
    ):
        for entry in panels[name]:
            draw_interval(
                ax,
                y=3 - entry["row"] + offsets[entry["model"]],
                model=entry["model"],
                estimate=entry["estimate"],
            )
        ax.set_yticks((3, 2, 1, 0), labels)
        ax.set_ylim(-0.48, 3.48)
        ax.set_title(title, loc="left", pad=15, linespacing=1.4, fontweight="normal")
        style_axis(ax, entries=panels[name])
    ax = axes[2]
    for entry in panels["blocking"]:
        draw_interval(
            ax,
            y=2.5 if entry["model"] == MODELS[0] else 0.5,
            model=entry["model"],
            estimate=entry["estimate"],
        )
    ax.set_yticks((2.5, 0.5), ("GPT-2\nMedium", "Pythia\n410M"))
    ax.set_ylim(-0.48, 3.48)
    ax.set_title(
        "c  END-clamp attenuation\nSelected − random",
        loc="left",
        pad=15,
        linespacing=1.4,
        fontweight="normal",
    )
    style_axis(ax, entries=panels["blocking"])
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=1.6,
            markersize=6.5,
            label=LABELS[model],
        )
        for model in MODELS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.998),
        ncol=2,
        frameon=False,
        columnspacing=3.0,
        handlelength=2.2,
        fontsize=11.4,
    )
    fig.text(
        0.5,
        0.104,
        "Pointwise 95% whole-template intervals; n = 8 templates per model. Separate panel scales.",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#333333",
    )
    fig.text(
        0.5,
        0.045,
        "Exploratory, reused templates. Forward: canonical → controlled; reverse: controlled → canonical.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4A4A4A",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=RESULTS / "analysis.json")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    data = load_verified(args.analysis)
    panels = plotted_estimates(data)
    fig = render(panels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in ("pdf", "png", "svg"):
        path = args.output_dir / f"{STEM}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.13, facecolor="white")
        paths[extension] = {"path": str(path.resolve()), "sha256": sha(path)}
    plt.close(fig)
    caption_path = args.output_dir / f"{STEM}-caption.txt"
    caption_path.write_text(CAPTION + "\n", encoding="utf-8")
    metadata = {
        "study": "005",
        "status": "complete figure from verified complete analyses",
        "analysis_path": str(args.analysis.resolve()),
        "analysis_sha256": sha(args.analysis),
        "plot_script_sha256": sha(Path(__file__)),
        "metric": METRIC,
        "units": "Raw differences in gain in P(IO) - P(subject).",
        "intervals": "Pointwise 95% whole-template bootstrap; exploratory and unadjusted.",
        "interval_verification": (
            "Every plotted mean and interval was independently recomputed from its eight "
            "retained template means using the saved bootstrap seed and sample count."
        ),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "models": MODELS,
        "caption": CAPTION,
        "plotted_estimates": panels,
        "outputs": paths,
    }
    (args.output_dir / f"{STEM}-data.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": metadata["status"], "outputs": paths}, indent=2))


if __name__ == "__main__":
    main()
