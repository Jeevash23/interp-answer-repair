"""Write a readable study report from the immutable analysis, without changing its criteria."""

from causal_circuit_routing.io import ROOT, digest, read_json


def main():
    analysis = ROOT / "results/study-002/analysis"
    s = read_json(analysis / "summary.json")
    provenance = read_json(analysis / "provenance.json")
    assert digest(analysis / "summary.json") == provenance["outputs"]["summary.json"]
    diagnostic = read_json(ROOT / "results/study-002/prediction-diagnostic.json")
    run = read_json(ROOT / "results/study-002/transfer/run.json")
    lines = [
        "# Study 002: completed results",
        "",
        (
            "**The measured state predicts prompt-conditioned causal dependencies, but neither tested "
            "edit demonstrates a robust circuit-family switch under the frozen criteria.** This is a "
            "bounded negative/mixed result for one selected direction and one full-vector comparison; "
            "it does not rule out other compact or distributed mechanisms."
        ),
        "",
        (
            "The study used GPT-2 Small, 64 discovery recipients, 64 development recipients and 128 test "
            "recipients. Names and templates were disjoint across splits. Test data comprise eight "
            "templates and four unordered name pairs, crossed with role assignment and ABBA/BABA order. "
            "All 30 conditions completed. No prompt was removed after observing model outcomes."
        ),
        "",
        "## Natural behavior and prediction",
        "",
        (
            f"Natural test pairwise accuracy was {100 * s['natural_behavior']['pairwise_accuracy']:.2f}%; "
            f"full-vocabulary top-1 accuracy was {100 * s['natural_behavior']['open_vocab_top1']:.2f}%. "
            "These measure different things: ranking the correct name above the repeated subject versus "
            "ranking it above every vocabulary token."
        ),
        "",
        "| Signature baseline | Test family classification | Natural axis separation | 95% block interval |",
        "|---|---:|---:|---|",
    ]
    for baseline, values in s["natural"].items():
        sep = values["natural_axis_separation"]
        lo, hi = sep["ci95"]
        lines.append(
            f"| {baseline} | {100 * values['family_balanced_accuracy']:.2f}% | "
            f"{sep['mean']:.4f} | [{lo:.4f}, {hi:.4f}] |"
        )
    lines += [
        "",
        (
            "The residual state before layer 6 at S2 was selected using development prediction only. "
            f"Its development R² was {s['selection']['development_r2']:.4f} and test R² was "
            f"{s['selection']['test_r2']:.4f}. A post-hoc discovery-fitted prompt-order mean achieved test "
            f"R² {diagnostic['splits']['test']['known_prompt_order_mean']['r2']:.4f}. Thus no predictive "
            "advantage over that simple input baseline is established. The diagnostic was added after "
            "partial test outputs were observed and changed no selection or causal test criterion."
        ),
        "",
        "## Causal transfer and disruption",
        "",
        (
            "The following comparison keeps the recipient's answer and wording fixed while changing "
            "the donor family. S is movement toward the opposite-family centroid along the fixed "
            "discovery axis. O is perpendicular movement, divided by the same centroid distance. "
            "A positive S alone is not a circuit switch."
        ),
        "",
        "| Edit | S, zero | O, zero | S, corruption | O, corruption | Pairwise accuracy | Top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("rank1", "full"):
        values = s["conditions"][mode + "__same_cross_same"]
        z, c = values["baselines"]["zero"], values["baselines"]["s2_corrupt"]
        lines.append(
            f"| {mode} | {z['score']['mean']:.4f} | {z['mean_orthogonal']:.4f} | "
            f"{c['score']['mean']:.4f} | {c['mean_orthogonal']:.4f} | "
            f"{100 * values['pairwise_accuracy']:.2f}% | {100 * values['open_vocab_top1']:.2f}% |"
        )
    lines += [
        "",
        "## Frozen decision criteria",
        "",
        (
            "Primary contrasts compare opposite-family minus same-family edits at fixed answer and "
            "wording. There are 16 per edit mode: two directions × two answers × two wording strata × "
            "two ablation baselines. The table counts positive Bonferroni simultaneous lower bounds. "
            "All components must pass; a few positive comparisons do not establish the proposed result."
        ),
        "",
        "| Edit | Positive primary bounds | Behavior preserved | Orthogonal bounded | All location/random controls | Overall |",
        "|---|---:|---|---|---|---|",
    ]
    for mode, gate in s["strict_gate_by_mode"].items():
        count = sum(
            v["simultaneous_ci"][0] > 0
            for k, v in s["primary_contrasts"].items()
            if k.startswith(mode + "/")
        )
        entries = [
            "Yes" if gate[k] else "No"
            for k in (
                "pairwise_behavior_preserved",
                "orthogonal_movement_bounded",
                "all_control_contrasts_positive",
            )
        ]
        lines.append(
            f"| {mode} | {count}/16 | "
            + " | ".join(entries)
            + f" | {'Pass' if gate['passed'] else 'Fail'} |"
        )
    lines += [
        "",
        (
            "The four positive adjusted contrasts in each mode are all BABA→ABBA under S2-corrupt "
            "replacement, across both answer and wording conditions. No zero-baseline contrast "
            "has a positive adjusted lower bound. This is evidence of a conditional directional "
            "effect, but it fails the stronger bidirectional, baseline-robust switching claim."
        ),
        "",
        (
            "Behavior preservation permits at most ten percentage points of pairwise accuracy loss. "
            "The perpendicular-movement bound compares mean O to mean positive S. Uncertainty uses "
            "10,000 crossed template/name-pair bootstrap draws, keeping paired roles and donor cells "
            "together. It conditions on the discovery fit. Eight templates and four name pairs cannot "
            "support broad population or language generalization claims."
        ),
        "",
        "## Complete condition inventory",
        "",
        (
            "Condition names encode wording / family / answer. `other` means alternate template; "
            "`cross` means opposite family. `same_same_same` is an identity control. Random and "
            "misplaced controls are norm-matched to the same-wording opposite-family same-answer edit "
            "within their own mode. The two edit modes are not matched to each other's norms. "
            "The alternative-name diagnostic always uses the recipient's four-donor alternative C; "
            "it need not be a shuffled donor's actual answer."
        ),
        "",
        "| Condition | Mean S, zero | Mean S, corrupt | Pairwise accuracy | Top-1 | Alternative C > IO |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in run["completed_conditions"]:
        c = s["conditions"][label]
        lines.append(
            f"| {label} | {c['baselines']['zero']['score']['mean']:.4f} | "
            f"{c['baselines']['s2_corrupt']['score']['mean']:.4f} | {100 * c['pairwise_accuracy']:.2f}% | "
            f"{100 * c['open_vocab_top1']:.2f}% | {100 * c['alternative_answer_beats_io']:.2f}% |"
        )
    lines += [
        "",
        "## Scope, evidence and reproduction",
        "",
        (
            "The 72 head coordinates at END, scales, centroids, site-selection rule and test analysis "
            "were fixed before test inference. They measure single-head causal dependencies, not a "
            "complete circuit or all possible routes. No new axes or candidates were selected to improve "
            "these test results. The earlier 64-recipient pilot remains unchanged and exploratory."
        ),
        "",
        (
            f"The transfer run took {run['duration_seconds'] / 60:.2f} minutes on CPU; peak process memory "
            f"was {run['peak_rss_mib'] / 1024:.2f} GiB. GPU checking and public-source retrieval were "
            "blocked by the app's automatic approval-review service error despite the user's approval. "
            "No GPU inference, fresh novelty audit, official-code reproduction or Pythia replication "
            "is claimed. The original protocol made replication conditional on a successful core result."
        ),
        "",
        (
            "See [reproduction instructions](STUDY-002-REPRODUCIBILITY.md), the "
            "[frozen protocol](STUDY-002-PROTOCOL.md), "
            "[complete statistics](../results/study-002/analysis/summary.json), "
            "[verification record](../results/study-002/verification.json) and "
            "[manuscript](../output/pdf/causal-dependencies-research-draft.pdf)."
        ),
        "",
        f"Analysis summary SHA-256: `{digest(analysis / 'summary.json')}`.",
        "",
    ]
    (ROOT / "docs/RESULTS_STUDY_002.md").write_text("\n".join(lines))
    print("Wrote docs/RESULTS_STUDY_002.md from verified analysis.")


if __name__ == "__main__":
    main()
