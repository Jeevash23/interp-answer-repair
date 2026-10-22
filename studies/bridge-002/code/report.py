"""Post-freeze readable tables and scientific figures from the frozen summary."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from design import HERE

from transport_bridge.common import digest, now, read, write


def ci(x, adjusted=False):
    lo, hi = x["simultaneous_ci" if adjusted else "ci95"]
    return f"{x['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def cell_label(key):
    w, o = key.split("_")
    return f"{'walked' if w == 'w0' else 'arrived'} / {'ABBA' if o == 'o0' else 'BABA'}"


def save_figure(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=180)
    fig.savefig(out / (name + ".svg"))
    plt.close(fig)


def model_report(model):
    directory = HERE / "results" / model
    s = read(directory / "analysis/summary.json")
    output = directory / "report"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    keys = list(s["natural"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    for i, key in enumerate(keys):
        for j, field in enumerate(["recovery", "score", "donor_margin"]):
            baselines = ["zero"] if j == 0 else s["baselines"]
            for bi, baseline in enumerate(baselines):
                x = s["primary_wording"][baseline][key][field]
                lo, hi = x["simultaneous_ci" if j < 2 else "ci95"]
                yy = i + (bi - 0.5) * 0.16 if j else i
                color = ["#225d88", "#b85c38"][bi]
                axes[j].plot([lo, hi], [yy, yy], color=color, lw=2)
                axes[j].scatter(
                    x["mean"],
                    yy,
                    color=color,
                    s=25,
                    label=baseline.replace("_", " ") if i == 0 else None,
                )
    for ax, title in zip(
        axes,
        [
            "Answer recovery\nAdjusted intervals",
            "Donor-axis position S\nAdjusted intervals",
            "Closer to donor (+) or recipient (−)\nPointwise intervals",
        ],
        strict=True,
    ):
        ax.axvline(0, color="#697586", ls="--", lw=1)
        ax.grid(axis="x", alpha=0.12)
        ax.set_title(title)
    axes[0].set_yticks(range(4), [cell_label(k) for k in keys])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Recipient family / order; donor is other family")
    axes[0].set_xlabel("IO − subject logit gain")
    axes[1].set_xlabel("Clean recipient = 0; discovery gap = 1")
    axes[1].legend(fontsize=8)
    axes[2].set_xlabel("Normalized distance margin")
    fig.suptitle(f"{model}: transfer between two grammatical families", fontweight="bold")
    save_figure(fig, output, "recovery-and-dependencies")

    donor_names = ["head_self", "head_role", "head_order", "head_order_role"]
    labels = [
        "Same order\nsame roles",
        "Same order\nswapped roles",
        "Opposite order\nsame roles",
        "Opposite order\nswapped roles",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True, layout="constrained")
    for ax, key in zip(axes, keys, strict=True):
        for i, name in enumerate(donor_names):
            x = s["conditions"][name]["cells"][key]["recovery"]
            lo, hi = x["ci95"]
            ax.plot([i, i], [lo, hi], color="#225d88", lw=2)
            ax.scatter(i, x["mean"], s=35, color="#b85c38")
        ax.axhline(0, color="#697586", ls="--", lw=1)
        ax.set_xticks(range(4), labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(cell_label(key))
        ax.grid(axis="y", alpha=0.12)
    axes[0].set_ylabel("IO − subject logit recovery")
    fig.suptitle(
        f"{model}: donor position and answer identity (pointwise intervals)", fontweight="bold"
    )
    save_figure(fig, output, "factorial-donors")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, layout="constrained")
    modes = ["z_selected", "query", "key", "value", "z_control0", "z_control1", "z_control2"]
    labels = [
        "Selected outputs",
        "Selected queries",
        "Selected keys",
        "Selected values",
        "Control outputs 1",
        "Control outputs 2",
        "Control outputs 3",
    ]
    for ax, factor in zip(axes, ["self", "wording", "order"], strict=True):
        for i, mode in enumerate(modes):
            x = s["blocking"][factor]["components"][mode]
            lo, hi = x["ci95"]
            ax.plot([lo, hi], [i, i], color="#225d88", lw=2)
            ax.scatter(x["mean"], i, color="#b85c38", s=28)
        ax.axvline(0, color="#697586", ls="--", lw=1)
        ax.set_title(
            {"self": "Own-clean repair", "wording": "Wording transfer", "order": "Order transfer"}[
                factor
            ]
        )
        ax.set_xlabel("Signed attenuation (logits)")
        ax.grid(axis="x", alpha=0.12)
    axes[0].set_yticks(range(len(modes)), labels)
    axes[0].invert_yaxis()
    fig.suptitle(f"{model}: which component changes carry the edit effect?", fontweight="bold")
    save_figure(fig, output, "component-clamps")

    lines = [
        f"# {model}: Bridge 002 results",
        "",
        "384 paired test cells; eight template blocks; six exclusive name triples. Both actor roles, wordings and orders are retained. Intervals condition on discovery scales and selected heads.",
        "",
        "`walked` means 'After ... walked into ... handed ... to'; `arrived` means 'When ... arrived at ... handed ... to'. These share the transfer verb and are a limited grammatical comparison.",
        "",
        "## Held-out validity",
        "",
        "| Family / order | Pairwise | Top-1 | Corrupted pairwise | Wording classification zero / mean |",
        "|---|---:|---:|---:|---|",
    ]
    for key, v in s["natural"].items():
        a = v["family_accuracy"]["wording"]
        lines.append(
            f"| {cell_label(key)} | {v['clean_pairwise']:.1%} | {v['clean_top1']:.1%} | {v['corrupt_pairwise']:.1%} | {a[0]:.1%} / {a[1]:.1%} |"
        )
    lines += [
        "",
        f"All clean behavioral gates pass: **{s['all_behavior_valid']}**. The thresholds are 95% pairwise and 85% top-1 in every cell. Natural signature classification is a separate >=75% gate; behavior alone does not ensure distinct identifiable dependency families.",
        "",
        "## All transfer conditions",
        "",
        "Recovery is always relative to the unedited corrupted recipient. Intervals in this table are secondary pointwise 95% intervals. The half-dose and full-state edits are fixed comparisons, not new optimized interventions.",
        "",
        "| Edit | Raw logit recovery [95% interval] | Probability recovery [95% interval] | Pairwise | Top-1 |",
        "|---|---|---|---:|---:|",
    ]
    for name, c in s["conditions"].items():
        lines.append(
            f"| {name} | {ci(c['raw_logit_recovery'])} | {ci(c['raw_probability_recovery'])} | {c['pairwise_accuracy']:.1%} | {c['top1_accuracy']:.1%} |"
        )
    lines += [
        "",
        "## Matched wording recovery and dependencies",
        "",
        "The row names identify the recipient; its donor has the other wording with the same names, roles and order. S is a donor-oriented projection relative to the clean recipient. It can change without actual donor proximity. A poorly classified wording axis cannot establish a distinguishable family switch.",
        "",
        "| Recipient | Reference | Recovery [adjusted] | S [adjusted] | Mean O | Donor margin [95%] | Pairwise loss |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for key in keys:
        for baseline in s["baselines"]:
            c = s["primary_wording"][baseline][key]
            lines.append(
                f"| {cell_label(key)} | {baseline} | {ci(c['recovery'], True)} | {ci(c['score'], True)} | {c['orthogonal']:.3f} | {ci(c['donor_margin'])} | {100 * c['pairwise_loss']:.1f} pp |"
            )
    lines += [
        "",
        f"Complete wording-switch gate passes: **{s['wording_switch']}**. Gate failure is not proof that no router or donor influence exists.",
        "",
        "![Recovery and dependency coordinates](recovery-and-dependencies.png)",
        "",
        "## Factorial order and answer-identity test",
        "",
        "The order effect averages over both role assignments; the role effect averages over both order assignments. Interaction measures whether one effect depends on the other. Intervals are adjusted across 12 quantities (three effects in four cells). A role-equivalence result applies only within the fixed ±0.2-logit margin, not to all semantic content.",
        "",
        "| Recipient | Order effect [adjusted] | Role effect [adjusted] | Interaction [adjusted] | Role equivalent ±0.2 |",
        "|---|---|---|---|---|",
    ]
    for key, c in s["factorial"].items():
        lines.append(
            f"| {cell_label(key)} | {ci(c['order_effect'], True)} | {ci(c['role_effect'], True)} | {ci(c['interaction'], True)} | {c['role_equivalent_within_0p2']} |"
        )
    lines += [
        "",
        "![Factorial donor effects](factorial-donors.png)",
        "",
        "## Causal component clamps",
        "",
        "Clamps hold selected head components at their unedited corrupted-recipient values in full forward passes. Signed attenuation is original edit effect minus effect under the clamp. Negative attenuation for an adverse order edit means the clamp reduces that adverse effect. Query, key and value effects need not add to the output-clamp effect, and no unique natural mediation path is identified.",
        "",
        "| Edit | Selected heads | Total effect [95%] | Selected output attenuation [95%] | Adjusted control criterion |",
        "|---|---|---|---|---|",
    ]
    for f, b in s["blocking"].items():
        heads = ", ".join(f"{l}.{h}" for l, h in b["heads"]["selected"])
        lines.append(
            f"| {f} | {heads} | {ci(b['total_effect'])} | {ci(b['selected_signed'])} | {b['adjusted_success']} |"
        )
    lines += [
        "",
        "For wording/order, the primary control test uses the discovery sign to orient the effect and adjusts across eight selected/control contrasts. The self group is an empirical repair control, selected by positive clean head effects; 'reader' is a selection label rather than proof of direct name copying.",
        "",
        "| Edit | Oriented attenuation [adjusted] | Advantage over control 1 [adjusted] | Advantage over control 2 [adjusted] | Advantage over control 3 [adjusted] |",
        "|---|---|---|---|---|",
    ]
    for f in ["wording", "order"]:
        b = s["blocking"][f]
        lines.append(
            f"| {f} | {ci(b['selected_oriented'], True)} | "
            + " | ".join(ci(x, True) for x in b["over_controls"])
            + " |"
        )
    lines += [
        "",
        "![Component clamp effects](component-clamps.png)",
        "",
        "The figure uses secondary pointwise intervals. Full control IDs and cell breakdowns are retained in the machine-readable summary.",
        "",
        "## Joint-head non-additivity",
        "",
        "Each quantity is the joint ablation effect minus the sum of the four individual effects. A change shows how transport alters this non-additivity under the specified replacement; it does not by itself label redundancy or compensation as a unique mechanism. The pooled corrupted signed effects cancel by the paired scoring design; a pooled zero is not evidence of no interactions. See [stratified diagnostics](DIAGNOSTICS.md) for cell breakdowns and a symmetry audit.",
        "",
        "| Edit | Reference | Corrupted context [95%] | Edited context [95%] | Change [95%] |",
        "|---|---|---|---|---|",
    ]
    for f, bs in s["joint"].items():
        for baseline, j in bs.items():
            lines.append(
                f"| {f} | {baseline} | {ci(j['corrupted_nonadditivity'])} | {ci(j['edited_nonadditivity'])} | {ci(j['change'])} |"
            )
    lines += [
        "",
        "## Selected-group attention",
        "",
        "These descriptive group means average heterogeneous heads. Attention alone is not a causal explanation; the component clamps above provide the intervention evidence.",
        "",
        "| Edit | Corrupted IO minus subject attention [95%] | Edited [95%] | Change [95%] |",
        "|---|---|---|---|",
    ]
    for f, a in s["attention"].items():
        lines.append(
            f"| {f} | {ci(a['corrupt_io_minus_subject'])} | {ci(a['edited_io_minus_subject'])} | {ci(a['change'])} |"
        )
    lines += [
        "",
        "## Provenance and scope",
        "",
        "The failed first screen and amended fresh screen are preserved. Scientific source and inputs were frozen before discovery; selected heads, controls and references were frozen before test. No test examples or failed conditions were removed. Both models use the same test prompts and belong to one model family. Shared templates, fixed discovery fits and a small name bank limit generalization.",
        "",
        "The role/position contrast has substantial prior art in IOI. See the study's mechanistic-context note before making novelty claims. This is an empirical follow-up rather than proof of complete circuit equivalence or a universal router.",
        "",
        f"Frozen summary SHA-256: `{digest(directory / 'analysis/summary.json')}`.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write(
        output / "provenance.json",
        {
            "utc": now(),
            "summary_sha256": digest(directory / "analysis/summary.json"),
            "outputs": {p.name: digest(p) for p in output.iterdir() if p.name != "provenance.json"},
        },
    )
    print("Report:", output / "RESULTS.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    model_report(p.parse_args().model)
