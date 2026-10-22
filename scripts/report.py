"""Build readable reports and scientific figures from the frozen analysis."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transport_bridge.common import ROOT, digest, now, read, write


def model_report(model):
    directory = ROOT / "results" / model
    summary = read(directory / "analysis/summary.json")
    output = directory / "report"
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for factor in ["wording", "order"]:
        for key, cell in summary["primary"][factor + "/zero"].items():
            direction, other = key.split("/")
            a = int(direction[0])
            if factor == "wording":
                label = f"{'Altered donor → natural recipient' if a == 0 else 'Natural donor → altered recipient'} / {'ABBA' if other[-1] == '0' else 'BABA'}"
            else:
                label = f"{'BABA donor → ABBA recipient' if a == 0 else 'ABBA donor → BABA recipient'} / {'natural' if other[-1] == '0' else 'altered'}"
            entries.append(
                (factor, key, label, cell, summary["primary"][factor + "/discovery_mean"][key])
            )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, layout="constrained")
    for j, (title, field) in enumerate(
        [
            ("Answer recovery\nIO − subject logit gain", "raw_logit_recovery"),
            ("Dependency position\nZero ablation", "score"),
            ("Dependency position\nDiscovery-mean ablation", "score"),
        ]
    ):
        ax = axes[j]
        for i, (_, _, _, zero, mean) in enumerate(entries):
            item = (mean if j == 2 else zero)[field]
            lo, hi = item["simultaneous_ci"]
            ax.plot([lo, hi], [i, i], color="#245b88", linewidth=1.6)
            ax.scatter(item["mean"], i, color="#c66336", s=27, zorder=3)
        ax.axvline(0, color="#667085", linewidth=1, linestyle="--")
        if j:
            ax.axvline(0.5, color="#a7afb9", linewidth=1, linestyle=":")
            ax.set_xlabel("S: clean-recipient position = 0")
        else:
            ax.set_xlabel("Gain relative to corrupted recipient")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.15)
    axes[0].set_yticks(range(len(entries)), [x[2] for x in entries])
    axes[0].invert_yaxis()
    fig.suptitle(
        f"{model}: the same head-transport edits measured two ways", fontsize=14, fontweight="bold"
    )
    fig.savefig(output / "paired-recovery-dependencies.png", dpi=180)
    fig.savefig(output / "paired-recovery-dependencies.svg")
    plt.close(fig)
    lines = [
        f"# {model}: bridge results",
        "",
        "These are completed measurements of the prespecified bridge, not a submission-ready paper.",
        f"There are {summary['n_test']} test recipient cells, eight paired template blocks, four name pairs, and 27 edits per recipient.",
        "Uncertainty resamples templates and name pairs and conditions on the discovery fit.",
        "Natural means use both role assignments. Primary intervals below are Bonferroni-adjusted; secondary tables are descriptive.",
        "",
        "## Natural task validity",
        "",
        "| Wording | Order | Clean pairwise | Clean vocabulary top-1 | Corrupted pairwise |",
        "|---|---|---:|---:|---:|",
    ]
    for key, v in summary["natural"].items():
        w, o = key.split("_")
        lines.append(
            f"| {'Natural' if w == 'w0' else 'Altered'} | {'ABBA' if o == 'o0' else 'BABA'} | {100 * v['pairwise_clean']:.1f}% | {100 * v['top1_clean']:.1f}% | {100 * v['pairwise_corrupt']:.1f}% |"
        )
    lines += [
        "",
        "The altered wording intentionally retains entity positions with unusual filler text. Pairwise ranking and full-vocabulary generation are different endpoints. Cells failing the frozen 75% validity gates cannot support a robust-switch claim; no such cell or example was removed.",
        "",
        "## Prediction",
        "",
        "| Target | Internal test R² | Prompt-label baseline R² | Family classification, zero / mean |",
        "|---|---:|---:|---:|",
    ]
    for f, v in summary["prediction"].items():
        acc = v["natural_family_accuracy_by_baseline"]
        lines.append(
            f"| {f} | {v['internal']['r2']:.4f} | {v['prompt_labels']['r2']:.4f} | {acc[0]:.3f} / {acc[1]:.3f} |"
        )
    lines += [
        "",
        "The prompt-label baseline was specified before test inference. The two ridge directions are fixed-site predictive directions, not causally trained optimizers.",
        "",
        "## Primary paired comparisons",
        "",
        "Historical early-head transport adds a clean-minus-corrupted source difference to the corrupted recipient. Labels below show donor-to-recipient transfer. Internal JSON keys instead encode recipient-to-donor axis orientation. The same head union is used in both transfer directions.",
        "",
        "| Comparison | Raw logit recovery [adjusted interval] | S zero [adjusted interval] | S mean [adjusted interval] |",
        "|---|---|---|---|",
    ]

    def interval(x):
        return f"{x['mean']:.3f} [{x['simultaneous_ci'][0]:.3f}, {x['simultaneous_ci'][1]:.3f}]"

    for _, _, label, z, m in entries:
        lines.append(
            f"| {label} | {interval(z['raw_logit_recovery'])} | {interval(z['score'])} | {interval(m['score'])} |"
        )
    lines += [
        "",
        "S is referenced to the unedited clean recipient, not the corrupted starting state. Positive answer recovery can therefore coincide with a near-zero S if the recipient’s own dependency pattern is restored. S is projection on a discovery centroid axis; it is not a complete circuit measurement. Neither a large projection nor a failed significance test alone establishes mechanism transfer or its absence.",
        "",
        "![Matched recovery and dependency measurements](paired-recovery-dependencies.png)",
        "",
        "### Probability and normalization checks",
        "",
        "These probability intervals are secondary pointwise 95% intervals. CTS is the ratio of mean recovery to mean clean-minus-corrupted response; it is not a success percentage. A dash means the prespecified denominator threshold was not met.",
        "",
        "| Comparison | Probability difference recovery [95% interval] | Logit CTS | Probability CTS |",
        "|---|---|---:|---:|",
    ]
    for _, _, label, z, _ in entries:
        p = z["raw_probability_recovery"]
        lo, hi = p["ci95"]
        cts = ["—" if z[k] is None else f"{z[k]:.3f}" for k in ["logit_cts", "probability_cts"]]
        lines.append(f"| {label} | {p['mean']:.4f} [{lo:.4f}, {hi:.4f}] | {cts[0]} | {cts[1]} |")
    lines += [
        "",
        "### Geometry and behavior checks",
        "",
        "O is perpendicular movement normalized by the discovery-axis length. A positive donor-proximity margin means closer to the matched clean donor than to the clean recipient; its intervals below are pointwise. The validity column checks the recipient cell, so an invalid donor remains a further limitation. The complete gate also requires all cells to pass.",
        "",
        "| Comparison | Reference | Mean O | Donor-proximity margin [95% interval] | Pairwise loss | Recipient valid |",
        "|---|---|---:|---|---:|---|",
    ]
    for _, _, label, z, m in entries:
        for baseline, item in [("zero", z), ("mean", m)]:
            margin = item["donor_closer_margin"]
            lo, hi = margin["ci95"]
            lines.append(
                f"| {label} | {baseline} | {item['mean_orthogonal']:.3f} | {margin['mean']:.3f} [{lo:.3f}, {hi:.3f}] | {100 * item['pairwise_accuracy_loss']:.1f} pp | {'yes' if item['natural_valid'] else 'no'} |"
            )
    lines += [
        "",
        "## Complete secondary condition summary",
        "",
        "Recovery in this table is always relative to the corrupted recipient. For edits ending in `_clean`, it includes the recipient's already-clean performance and must not be read as repair caused by that edit.",
        "",
        "| Edit | Raw logit recovery | Pairwise accuracy | Top-1 | Alternative > correct | Mean delta norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in summary["conditions"].items():
        lines.append(
            f"| {name} | {v['raw_logit_recovery']['mean']:.4f} | {100 * v['pairwise_accuracy']:.1f}% | {100 * v['top1_accuracy']:.1f}% | {100 * v['alternative_beats_io']:.1f}% | {v['mean_delta_norm']:.3f} |"
        )
    passed = [name for name, value in summary["strict_switch_by_condition"].items() if value]
    lines += [
        "",
        f"Conditions passing the complete operational switch gate: {', '.join(passed) if passed else 'none'}.",
        "",
        "The gate requires validity, donor-oriented movement in every stratum and both baselines, bounded perpendicular movement, donor proximity, and preserved pairwise behavior. Its failure is not proof that no compact router exists.",
        "",
        "## Controls",
        "",
        "| Correct intervention | Logit gain over mean random [95% interval] |",
        "|---|---|",
    ]
    for name, v in summary["control_contrasts"].items():
        a = v["logit_over_random"]
        lo, hi = a["ci95"]
        lines.append(f"| {name} | {a['mean']:.4f} [{lo:.4f}, {hi:.4f}] |")
    lines += [
        "",
        "These are secondary pointwise control intervals, not separately corrected confirmatory discoveries. Random residual controls match norm within intervention type; head and residual interventions do not share an equal functional dose.",
        "",
        "## Limits and provenance",
        "",
        "- All names are a small purposive set of single-token masculine names; neither lexical nor demographic breadth is established.",
        "- Controlled wording changes linguistic validity. Results about pairwise name preference should not be generalized to unrestricted generation.",
        "- Shared templates across model evaluations do not supply independent template replication.",
        "- The measurement uses individual END-head effects, omitting earlier components, joint effects and path structure.",
        "- Zero and discovery-mean ablations share a fixed reference across prompts; this deliberately differs from the previous routing study’s recipient-specific corruption bank.",
        "- Whole-boundary transplant is a computational positive control. It does not demonstrate a compact causal router.",
        "- The source archive, raw arrays, locked environment, selection freeze and complete statistical summary are retained. The earlier studies remain separate.",
        "",
        f"Analysis summary SHA-256: `{digest(directory / 'analysis/summary.json')}`.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write(
        output / "provenance.json",
        {
            "utc": now(),
            "summary_sha256": digest(directory / "analysis/summary.json"),
            "outputs": {f.name: digest(f) for f in output.iterdir() if f.name != "provenance.json"},
        },
    )
    print("Report:", output / "RESULTS.md")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    model_report(parser.parse_args().model)
