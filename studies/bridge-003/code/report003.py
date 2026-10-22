"""Post-analysis tables and figures; never selects or changes experiments."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common003 import HERE, MODELS, read


def interval(v, adjusted=True):
    lo, hi = v["simultaneous_ci" if adjusted else "ci95"]
    return f"{v['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def savefig(fig, directory, name):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(directory / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def medium():
    d = HERE / "results/medium-cancellation"
    s = read(d / "analysis/summary.json")
    out = d / "report"
    out.mkdir(exist_ok=True)
    lines = [
        "# Medium: opposing contributions on fresh prompts",
        "",
        (
            "The two discovery-selected groups have opposing causal contributions on the new "
            "384-case test. Both pass all six layer/size-matched control comparisons using "
            "the fixed 20-quantity interval family. Their joint effect fails the zero-equivalence "
            "criterion and is not the sum of their separate effects. This supports opposing "
            "contributions, not a simple cancellation explanation of the earlier near-zero result."
        ),
        "",
        (
            "A = L15H14, L13H13; B = L22H14, L17H12 (zero-based indices). "
            "The groups were selected from Bridge 002 discovery data. The hypothesis was "
            "informed by that study's test results, then tested on new names and blocks."
        ),
        "",
        (
            "Clean pairwise accuracy is 100% in every cell; vocabulary top-1 is 98.96–100%. "
            "There are eight location/object blocks, six name triples, both actor roles, "
            "two grammatical formulations and both mention orders. Repeated factorial "
            "cells are not 384 independent random samples."
        ),
        "",
        "## Primary estimates",
        "",
        (
            "Attenuation = (edited − base) − (edited with clamp − base with clamp), in "
            "IO-minus-subject logit units. Clamp values come from the corrupted recipient. "
            "The table uses the discovery-fixed orientation: wording +1, order −1. "
            "Positive values oppose the expected edit effect when clamped; negative values "
            "mean the clamp strengthens that oriented effect. Intervals are approximate "
            "95% simultaneous Bonferroni intervals across all 20 primary quantities."
        ),
        "",
    ]
    labels = list(s["factors"]["wording"]["primary"])
    lines += [
        table(
            ["Quantity", "Wording", "Order (orientation reversed)"],
            [
                [
                    k,
                    interval(s["factors"]["wording"]["primary"][k]),
                    interval(s["factors"]["order"]["primary"][k]),
                ]
                for k in labels
            ],
        ),
        "",
        (
            "B-versus-control comparisons reverse the subtraction so that a larger opposing "
            "B contribution gives a positive contrast. Three random groups are controls, "
            "not three independent experimental replications."
        ),
        "",
        "## Secondary estimates",
        "",
    ]
    lines += [
        table(
            ["Quantity", "Wording", "Order"],
            [
                [
                    "Signed answer recovery",
                    interval(s["factors"]["wording"]["recovery"], False),
                    interval(s["factors"]["order"]["recovery"], False),
                ],
                [
                    "Oriented union − A − B",
                    interval(s["factors"]["wording"]["nonadditivity"], False),
                    interval(s["factors"]["order"]["nonadditivity"], False),
                ],
            ],
        ),
        "",
        (
            "All following component estimates use the original signed logit direction and "
            "pointwise 95% intervals. Queries are clamped at END; keys and values at every "
            "position. These are different interventions and do not yield additive mediation shares."
        ),
        "",
    ]
    components = s["factors"]["wording"]["signed_components"]
    lines += [
        table(
            ["Clamp", "Wording (signed)", "Order (signed)"],
            [
                [
                    k,
                    interval(components[k], False),
                    interval(s["factors"]["order"]["signed_components"][k], False),
                ]
                for k in components
            ],
        ),
        "",
        (
            "The query clamp closely reproduces B's large opposing effect, while B's key "
            "and value effects are small. A is sensitive to queries and values, with a "
            "smaller key contribution. This localizes an intervention-sensitive difference; "
            "it does not establish unique information content, a complete path, or a new head class."
        ),
        "",
        "## Interpretation and limits",
        "",
        (
            "For wording, the joint clamp increases repair: its attenuation is negative. "
            "The fixed ±0.2-logit equivalence criterion fails for both factors. The previous "
            "near-zero union result therefore does not generalize as an exact cancellation. "
            "Sequential clamps affect overlapping downstream computation, so union − A − B "
            "is an assay-specific nonadditivity statistic, not automatically a particular "
            "path-specific mediation or xINT estimand from another paper."
        ),
        "",
        (
            "Uncertainty uses 10,000 crossed block/name-triple bootstrap draws, conditional "
            "on the fixed discovery selection. Only eight blocks and six triples limit "
            "generalization; adjusted bootstrap tails have finite Monte Carlo precision. "
            "Earlier source-head selection and the follow-up hypothesis were not selected "
            "blind to all historical results. New names exclude Bridge 001/002, not model pretraining."
        ),
        "",
        "![Opposing contributions](medium-opposition.png)",
        "",
        "![Query/key/value clamps](medium-components.png)",
        "",
        (
            "Saved-output algebra and exact statistical replay passed; see ../verification.json. "
            "This replay does not claim a second independent model-inference run."
        ),
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, factor in zip(axes, ["wording", "order"], strict=True):
        keys = ["A", "B", "union"]
        vals = [s["factors"][factor]["primary"][k] for k in keys]
        y = np.array([v["mean"] for v in vals])
        ci = np.array([v["simultaneous_ci"] for v in vals]).T
        ax.bar(keys, y, color=["#2176ae", "#cf5c36", "#666666"], width=0.6)
        ax.errorbar(range(3), y, yerr=[y - ci[0], ci[1] - y], fmt="none", color="black", capsize=4)
        ax.axhline(0, color="black", lw=0.7)
        ax.axhspan(-0.2, 0.2, color="grey", alpha=0.12)
        ax.set_title(factor.capitalize() + (" (orientation reversed)" if factor == "order" else ""))
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Oriented attenuation (logits)")
    fig.suptitle("Opposing effects persist; the union fails the equivalence criterion")
    fig.tight_layout()
    savefig(fig, out, "medium-opposition")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, group in zip(axes, ["A", "B"], strict=True):
        keys = [group, group + "_q", group + "_k", group + "_v"]
        vals = [components[k] for k in keys]
        y = np.array([v["mean"] for v in vals])
        ci = np.array([v["ci95"] for v in vals]).T
        ax.bar(["z", "q", "k", "v"], y, color="#2176ae" if group == "A" else "#cf5c36")
        ax.errorbar(range(4), y, yerr=[y - ci[0], ci[1] - y], fmt="none", color="black", capsize=4)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"Group {group}, wording edit")
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Signed attenuation (logits); pointwise 95% CI")
    fig.tight_layout()
    savefig(fig, out, "medium-components")


def screen():
    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        s = read(HERE / "results" / model / "screen.json")
        x = np.arange(8)
        for order in range(2):
            acc = [s["scores"][str(f)]["cells"][str(order)]["top1"] for f in range(8)]
            ax.bar(
                x + (order - 0.5) * 0.35,
                np.array(acc) * 100,
                width=0.35,
                label=["ABBA", "BABA"][order],
            )
        ax.axhline(85, color="black", ls="--", lw=1, label="85% top-1 requirement")
        ax.set_xticks(x, [f"F{i}" for i in x])
        ax.set_ylim(0, 108)
        ax.set_title(model)
        ax.spines[["right", "top"]].set_visible(False)
        for family, v in s["scores"].items():
            cells = [v["cells"][str(o)] for o in range(2)]
            rows.append(
                [
                    model,
                    "F" + family,
                    " / ".join(f"{c['pairwise'] * 100:.2f}%" for c in cells),
                    " / ".join(f"{c['top1'] * 100:.2f}%" for c in cells),
                    "pass" if v["passes"] else "fail",
                    "yes" if int(family) in s["retained"] else "no",
                ]
            )
    axes[0].set_ylabel("Clean vocabulary top-1 accuracy (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8)
    fig.tight_layout()
    savefig(fig, out, "screen-accuracy")
    (out / "SCREEN.md").write_text(
        "# Eight-family behavior screen\n\n32 cases per family/model, 16 per order. "
        "A pass requires at least 95% pairwise and 85% vocabulary top-1 in both orders. "
        "These are observed-accuracy gates, not lower confidence bounds.\n\n"
        + table(
            ["Model", "Family", "Pairwise ABBA / BABA", "Top-1 ABBA / BABA", "Gate", "Retained"],
            rows,
        )
        + "\n\nMedium has six passing candidates; the fixed first-four rule retains F0/F1/F2/F4. "
        "F6/F7 were scored but not used in discovery. Pythia has no passing candidate; "
        "it retains the predetermined F0/F1 diagnostic pair. F4's 81.25% BABA top-1 "
        "is below the 85% requirement; the threshold was not relaxed.\n\n"
        "![Screen results](screen-accuracy.png)\n"
    )


def broad(model):
    d = HERE / "results" / model
    s = read(d / "analysis/summary.json")
    selection = read(HERE / "data/broad-selection.json")
    out = d / "report"
    out.mkdir(exist_ok=True)
    lines = [
        f"# Broader wording: {model}",
        "",
        (
            f"Selected families: {s['families']}. Discovery qualified: {s['discovery_qualified']}. "
            f"Held-out behavior valid: {s['behavior_valid']}. Held-out signatures separated: "
            f"{s['test_separated']}. All-cell strict wording switch: {s['wording_switch']}."
        ),
        "",
        (
            "A failed gate limits interpretation; it is not evidence that the donor has no "
            "influence. All 384 test cases and all eight edit conditions are retained."
        ),
        "",
        "## Selection and clean held-out performance",
        "",
    ]
    scores = selection["scores"][model]
    lines += [
        table(
            ["Discovery pair", "Worst reference/cell accuracy", "Qualified"],
            [[k, f"{v['worst']:.3f}", v["qualified"]] for k, v in scores.items()],
        )
        if scores
        else "No behavior-eligible pair. F0/F1 is the frozen diagnostic fallback.",
        "",
    ]
    lines += [
        table(
            ["Cell", "Pairwise", "Top-1", "Wording classification zero / mean"],
            [
                [
                    k,
                    f"{v['pairwise'] * 100:.2f}%",
                    f"{v['top1'] * 100:.2f}%",
                    " / ".join(f"{a * 100:.2f}%" for a in v["wording_classification"]),
                ]
                for k, v in s["natural"].items()
            ],
        ),
        "",
        (
            "Cell wording labels w0/w1 refer to the selected families in the displayed "
            "order; o0/o1 are ABBA/BABA. There are 96 cases per cell. Reference means "
            "and scales use discovery data only at test time. Internal selection CV "
            "holds out centroid/scale-fitting blocks but shares a discovery replacement "
            "bank; see [selection notes](../../../docs/DISCOVERY-NOTES.md)."
        ),
        "",
        "## Behavioral interventions",
        "",
    ]
    lines += [
        table(
            ["Edit", "Logit recovery (pointwise 95% CI)", "Pairwise", "Top-1"],
            [
                [
                    k,
                    interval(v["recovery"], False),
                    f"{v['pairwise'] * 100:.2f}%",
                    f"{v['top1'] * 100:.2f}%",
                ]
                for k, v in s["conditions"].items()
            ],
        ),
        "",
        "Wording edit advantage over the mean of the three per-head norm-matched "
        "random controls: " + interval(s["wording_over_random"], False) + " logits.",
        "",
        "## Primary wording tests",
        "",
        (
            "Each model/reference uses an eight-quantity family: four recovery estimates "
            "and four S scores. Intervals below use its Bonferroni adjustment. S=1 "
            "means one full discovery-centroid displacement along the wording axis; "
            "it does not mean a whole circuit was transferred. Orthogonal displacement "
            "and actual donor distance also matter."
        ),
        "",
    ]
    lines += [
        table(
            ["Cell", "Wording recovery (adjusted CI)"],
            [[k, interval(v["recovery"])] for k, v in s["conditions"]["wording"]["cells"].items()],
        ),
        "",
    ]
    lines += [
        table(
            [
                "Reference",
                "Cell",
                "S (adjusted CI)",
                "Orthogonal",
                "Donor margin",
                "Pairwise loss",
                "Switch",
            ],
            [
                [
                    ref,
                    cell,
                    interval(v["score"]),
                    f"{v['orthogonal']:.3f}",
                    interval(v["margin"], False),
                    f"{v['pairwise_loss']:.3f}",
                    v["switch"],
                ]
                for ref, cells in s["primary_wording"].items()
                for cell, v in cells.items()
            ],
        ),
        "",
        (
            "The fixed switch requires valid clean behavior, qualified and held-out "
            "distinguishable families, adjusted S lower bound >0.5, orthogonal mean "
            "no greater than positive S, positive donor-proximity margin, and pairwise "
            "accuracy loss ≤0.1 in every cell/reference. Passing one condition is insufficient."
        ),
        "",
        "## Secondary order/role factorial",
        "",
    ]
    lines += [
        table(
            ["Contrast", "Cell", "Signed estimate (pointwise 95% CI)"],
            [
                [factor, cell, interval(v, False)]
                for factor, cells in s["factorial"].items()
                for cell, v in cells.items()
            ],
        ),
        "",
        (
            "The full order-transfer geometry and probability outcomes remain in "
            "../analysis/summary.json. This report does not elevate those secondary "
            "comparisons to new primary hypotheses after seeing outcomes."
        ),
        "",
        "![Behavioral repair and actual donor proximity](repair-and-proximity.png)",
        "",
        "![Behavior and dependency displacement](broad-outcomes.png)",
        "",
        (
            "The analysis conditions on one checkpoint and discovery fit. Eight test "
            "blocks and six name triples provide limited generalization. Both references "
            "are interventions, not ground-truth complete circuits. Numerical checks "
            "preceded screening; Pythia used a disclosed 0.002-logit intervention "
            "equivalence tolerance, while raw-model comparison stayed at 0.0005."
        ),
        "",
        (
            "Saved-output geometry, factorial algebra, hashes and exact replay are "
            "audited separately in ../verification.json. No second full inference "
            "run is implied by that statistical replay."
        ),
    ]
    cal = read(d / "geometry-calibration.json")
    paired = read(d / "paired-geometry/summary.json")
    lines += [
        "",
        "## Post hoc endpoint calibration and paired-axis diagnostic",
        "",
        (
            "The original off-axis rule can reject the actual clean donor endpoint. "
            "This check was added after inspecting Medium's wording outcomes and does not "
            "change the frozen primary analysis. A failure below means the original gate "
            "cannot be used on its own as evidence against a circuit switch. Actual-donor "
            "proximity and the raw displacement remain separately interpretable measurements."
        ),
        "",
        table(
            ["Reference", "Cell", "True donor S", "True donor orthogonal", "Geometry gate"],
            [
                [
                    ref,
                    k,
                    f"{v['score']['mean']:.3f}",
                    f"{v['orthogonal']:.3f}",
                    v["geometry_passes"],
                ]
                for ref, cells in cal["actual_donor"].items()
                for k, v in cells.items()
            ],
        ),
        "",
        (
            "A secondary paired-axis projection uses each actual clean recipient-to-donor "
            "difference. Its reference endpoints are exactly 0 and 1 with zero orthogonal "
            "error. This uses held-out clean references and is explicitly descriptive, "
            "not a newly substituted primary criterion. Intervals below are pointwise 95%."
        ),
        "",
        table(
            ["Reference", "Cell", "Paired-axis S", "Distance margin", "% closer to donor"],
            [
                [
                    ref,
                    k,
                    interval(v["score"], False),
                    interval(v["distance_margin"], False),
                    f"{100 * v['fraction_closer_to_donor']:.1f}",
                ]
                for ref, cells in paired["cells"].items()
                for k, v in cells.items()
            ],
        ),
        "",
        (
            "See [the dated diagnostic note](../../../docs/GEOMETRY-CALIBRATION-NOTE.md). "
            "The paired-axis comparison is another description of the same saved outputs, "
            "not independent replication or proof of a complete mechanism."
        ),
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    keys = list(s["conditions"])
    y = np.array([s["conditions"][k]["recovery"]["mean"] for k in keys])
    ci = np.array([s["conditions"][k]["recovery"]["ci95"] for k in keys]).T
    axes[0].bar(keys, y, color="#2176ae")
    axes[0].errorbar(
        range(len(keys)), y, yerr=[y - ci[0], ci[1] - y], fmt="none", color="black", capsize=3
    )
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_ylabel("Logit recovery; pointwise 95% CI")
    for i, (ref, cells) in enumerate(s["primary_wording"].items()):
        vals = [v["score"] for v in cells.values()]
        y = np.array([v["mean"] for v in vals])
        ci = np.array([v["simultaneous_ci"] for v in vals]).T
        axes[1].errorbar(
            np.arange(4) + (i - 0.5) * 0.13,
            y,
            yerr=[y - ci[0], ci[1] - y],
            fmt="o",
            capsize=3,
            label=ref,
        )
    axes[1].set_xticks(range(4), list(s["natural"]))
    axes[1].axhline(0.5, color="black", ls="--", lw=1)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Wording-axis S; adjusted CI")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(
        model
        + f" | clean behavior valid: {s['behavior_valid']} | signatures separated: {s['test_separated']}"
    )
    fig.tight_layout()
    savefig(fig, out, "broad-outcomes")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    accuracy = [
        np.mean([v["top1"] for v in s["natural"].values()]),
        np.mean([v["corrupt_top1"] for v in s["natural"].values()]),
        s["conditions"]["wording"]["top1"],
    ]
    bars = axes[0].bar(
        ["Clean", "Corrupted", "Wording edit"],
        np.array(accuracy) * 100,
        color=["#777777", "#cf5c36", "#2176ae"],
        width=0.6,
    )
    axes[0].bar_label(bars, labels=[f"{v * 100:.1f}%" for v in accuracy], padding=3)
    axes[0].set_ylim(0, 113)
    axes[0].set_ylabel("Vocabulary top-1 accuracy (%)")
    for i, (ref, cells) in enumerate(s["primary_wording"].items()):
        vals = [v["margin"] for v in cells.values()]
        y = np.array([v["mean"] for v in vals])
        ci = np.array([v["ci95"] for v in vals]).T
        axes[1].errorbar(
            np.arange(4) + (i - 0.5) * 0.13,
            y,
            yerr=[y - ci[0], ci[1] - y],
            fmt="o",
            capsize=3,
            label=ref.replace("_", " "),
        )
    axes[1].set_xticks(range(4), list(s["natural"]))
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("Recipient distance − donor distance\nPointwise 95% CI; <0 favors recipient")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(model + " | wording transfer on held-out prompts")
    fig.tight_layout()
    savefig(fig, out, "repair-and-proximity")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["medium", "screen", "broad", "all"])
    p.add_argument("--model", choices=MODELS)
    a = p.parse_args()
    if a.arm in ["medium", "all"]:
        medium()
    if a.arm in ["screen", "all"]:
        screen()
    if a.arm == "broad":
        broad(a.model)
    if a.arm == "all":
        for m in MODELS:
            broad(m)
