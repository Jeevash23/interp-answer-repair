"""Render completed, independently verified Bridge 008 results; no inference.

Outputs RESULTS.md and tables/summary.tex. A fresh --output directory permits
report replay without overwriting the canonical report. No placeholder results
are written when observations or verification are incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
BASE = ROOT / "studies/bridge-007"
REFS = ("zero", "fit_mean")
VARIANTS = ("narrow", "broad", "matched")
REF_LABEL = {"zero": "Zero", "fit_mean": "Fit mean"}
VARIANT_LABEL = {"narrow": "Narrow", "broad": "Broad", "matched": "Matched broad"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def number(value, digits=3):
    if value is None:
        return "not measured"
    if value != 0 and (abs(value) < 0.0005 or abs(value) >= 10000):
        return f"{value:.2g}"
    return f"{value:.{digits}f}"


def interval(value, adjusted=False):
    if value is None or not value.get("valid", False):
        return "not measured"
    lo, hi = value["adjusted_ci" if adjusted else "ci95"]
    return f"{number(value['mean'])} [{number(lo)}, {number(hi)}]"


def percentage(value):
    return "not measured" if value is None else f"{100*value:.1f}%"


def markdown_table(headers, rows):
    def escape(value):
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(escape, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(escape, row)) + " |" for row in rows])


def tex_escape(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
               "|": r"\textbar{}", "Δ": r"\(\Delta\)", "×": r"\(\times\)", "−": "-", "–": "--"}
    return "".join(mapping.get(character, character) for character in str(value))


def tex_table(label, caption, headers, rows):
    # The surrounding manuscript supplies booktabs. Six compact columns fit the
    # existing text width at footnotesize; long descriptive notes stay outside.
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{" + tex_escape(caption) + "}", r"\label{tab:b008-" + label + "}",
             r"\begin{tabular}{@{}" + "l"*len(headers) + "@{}}", r"\toprule",
             " & ".join(map(tex_escape, headers)) + r" \\", r"\midrule"]
    lines += [" & ".join(map(tex_escape, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def render(summary, verification):
    families = summary["families"]
    all_positive = summary["all_four_donorward_increases"]
    decisions = [summary["primary_decisions"][variant][ref] for variant in ("broad", "matched") for ref in REFS]
    if all_positive:
        finding = ("All four adjusted lower bounds exceed zero, meeting the specified donorward-increase criterion. "
                   "Wider source coverage increases the average donor-axis wording-versus-self difference under both "
                   "references, including when each layer's total edit norm is matched to the narrow intervention.")
    elif all(decision == "donorward_decrease" for decision in decisions):
        finding = ("All four adjusted intervals indicate a decrease in the donor-axis wording-versus-self difference. "
                   "The specified donorward-increase criterion is not met; the change is in the opposite direction.")
    else:
        finding = ("The specified four-contrast donorward-increase criterion is not met. Raw and norm-matched "
                   "outcomes are reported separately below. An interval spanning zero is unresolved, not evidence "
                   "of invariance or equivalent computations.")
    tables = []
    primary_rows = []
    names = {"donorward_increase": "Increase", "donorward_decrease": "Decrease",
             "unresolved": "Unresolved", "non_evaluable": "Not evaluable"}
    for variant in ("broad", "matched"):
        for ref in REFS:
            primary_rows.append([VARIANT_LABEL[variant], REF_LABEL[ref],
                                 interval(summary["primary"][variant][ref], adjusted=True),
                                 names[summary["primary_decisions"][variant][ref]]])
    tables.append(("primary", "Four primary paired source-coverage interactions. Intervals use the fixed four-quantity Bonferroni family.",
                   ["Coverage", "Reference", "Interaction [adjusted CI]", "Decision"], primary_rows))
    cell_rows = []
    for variant in ("broad", "matched"):
        for ref in REFS:
            cell_data = summary["primary_cellwise"][variant][ref]
            for wording in (0, 1):
                for order in (0, 1):
                    value = None if cell_data is None else cell_data[f"w{wording}_o{order}"]
                    cell_rows.append([VARIANT_LABEL[variant], REF_LABEL[ref], f"F{families[wording]} / {order}", interval(value)])
    assert len(cell_rows) == 16
    tables.append(("cells", "All sixteen prespecified cell interactions, with pointwise 95% intervals. Pooled effects need not hold in every cell.",
                   ["Coverage", "Reference", "Recipient / order", "Interaction [95% CI]"], cell_rows))
    behavior_rows = []
    for variant in VARIANTS:
        for donor in ("wording", "self"):
            value = summary["conditions"].get(variant+"_"+donor)
            behavior_rows.append([VARIANT_LABEL[variant]+" / "+donor,
                                  interval(None if value is None else value["recovery"]),
                                  percentage(None if value is None else value["top1"]),
                                  percentage(None if value is None else value["pairwise"])])
    assert len(behavior_rows) == 6
    tables.append(("behavior", "Recovery in IO-minus-subject logits, with pointwise 95% intervals. Accuracy uses the same retained configurations.",
                   ["Coverage / donor", "Recovery [95% CI]", "Top-1", "Pairwise"], behavior_rows))
    difference_rows = []
    for variant in VARIANTS:
        values = summary["secondary_wording_self"][variant]
        for ref in REFS:
            row = [VARIANT_LABEL[variant], REF_LABEL[ref]]
            for endpoint in ("signed_axis_difference", "absolute_axis_difference", "full_signature_discrepancy"):
                row.append(interval(None if values is None else values[ref][endpoint]["average"]))
            difference_rows.append(row)
    tables.append(("differences", "Wording-versus-self differences at each coverage level, in clean-separation units. All intervals are pointwise 95%.",
                   ["Coverage", "Reference", "Signed ΔS [95% CI]", "Mean |ΔS| [95% CI]", "Full distance [95% CI]"], difference_rows))
    distance_rows = []
    for variant in VARIANTS:
        for donor in ("wording", "self"):
            condition = summary["conditions"].get(variant+"_"+donor)
            for ref in REFS:
                distance_rows.append([VARIANT_LABEL[variant]+" / "+donor, REF_LABEL[ref],
                    interval(None if condition is None else condition["geometry"][ref]["q_recipient"]["average"]),
                    interval(None if condition is None else condition["geometry"][ref]["q_donor"]["average"])])
    tables.append(("endpoints", "Edited distances from the clean recipient (qR) and donor (qD), in clean-separation units; pointwise 95% intervals.",
                   ["Coverage / donor", "Reference", "qR [95% CI]", "qD [95% CI]"], distance_rows))
    dose_rows = []
    degeneracy = []
    for donor in ("wording", "self"):
        matching = summary["matching"][donor]
        degeneracy.append(f"{donor}: {matching['undefined_layers']} undefined layer matches across "
                          f"{matching['undefined_configurations']} configurations; "
                          f"{matching['zero_narrow_layers']} near-zero narrow and {matching['zero_broad_layers']} near-zero broad layer norms")
        for layer in summary["source_layers"]:
            cell = matching["per_layer"][str(layer)]
            quantiles = cell["alpha"]["quantiles"]
            alpha_range = "undefined" if quantiles is None else f"{number(quantiles[0])} to {number(quantiles[-1])}"
            dose_rows.append([donor, str(layer), number(cell["narrow_norm"]["mean"]), number(cell["broad_norm"]["mean"]),
                              alpha_range, str(cell["undefined"])])
    tables.append(("doses", "Layerwise dose diagnostics. Norms are averages of the summed residual-vector norms; alpha is the matched-broad scale, with its observed range.",
                   ["Donor", "Layer", "Narrow norm", "Broad norm", "Alpha range", "Undefined"], dose_rows))
    preamble = (f"This correlated explanatory follow-up reuses Bridge 007's qualified F{families[0]}/F{families[1]} "
                f"benchmark: {summary['n']:,} configurations from {summary['blocks']} context blocks and "
                f"{summary['triples']} disjoint name triples. All 112 broad source heads occupy the same seven "
                "S2 layers (6–12) as the twelve narrow heads. Fitted references, clean endpoints, corruption and "
                "downstream measurement heads are unchanged.")
    metric = ("For each configuration, D is wording-transfer donor-axis position minus self-transfer position. "
              "The interaction is D at broad or matched-broad coverage minus D at narrow coverage. Positive "
              "values mean an increased donorward differential. The primary intervals use 50,000 paired "
              "crossed-bootstrap draws with seed 2026100408 and a four-quantity Bonferroni family. "
              "The criterion requires all four adjusted lower bounds above zero.")
    scope = ("The new interventions were specified before their measurement, but this is an outcome-informed "
             "follow-up on an inspected benchmark, not independent replication. Uncertainty conditions on "
             "the benchmark, fixed references and selected hypotheses. Even positive raw and norm-controlled "
             "interactions establish only coverage-dependent donor-axis sensitivity under this assay. They "
             "do not establish donor algorithm transfer or a complete information bottleneck. A null signed "
             "interaction does not establish scope invariance: opposite effects can cancel, and donor wording "
             "information shared by clean and corrupted source states can cancel in the injected difference. "
             "Recovery and endpoint distances must be considered alongside the signature contrast; disrupted "
             "examples are retained.")
    md = ["# Bridge 008 — source coverage and donor-axis sensitivity", "", finding, "", preamble, "", metric, "",
          f"Clean top-1 accuracy is {percentage(summary['clean_top1'])}; corrupted top-1 is {percentage(summary['corrupt_top1'])}.", ""]
    headings = {"primary": "Primary contrasts", "cells": "Every family/order cell", "behavior": "Behavioral recovery",
                "differences": "Signed, absolute and full signature differences", "endpoints": "Distances to clean endpoints",
                "doses": "Layerwise dose and matching diagnostics"}
    for label, caption, headers, rows in tables:
        md.extend(["## "+headings[label], "", caption, "", markdown_table(headers, rows), ""])
    md.extend(["; ".join(degeneracy)+".", "",
               "Matching uses the actual summed residual vector in each layer, with float64 normalization and float32 injection. "
               "The norm check allows 1e-6 absolute plus 1e-5 relative error. Undefined matching makes the entire corresponding "
               "matched condition unavailable; no configuration is omitted and the primary family remains four.", "",
               "## Interpretation and evidence", "", scope, "",
               "All secondary cell estimates, configuration distributions, recovery changes relative to narrow, and endpoint-distance changes "
               "are retained in [the complete analysis](results/analysis/summary.json). "
               "The [per-example arrays](results/analysis/per-example.npz) permit direct reanalysis.", "",
               "The [independent verification](results/verification.json) rechecks frozen inputs, source vectors, calibration, chunk aggregation, "
               "fresh-engine replay and all four primary intervals. The [protocol](docs/PROTOCOL.md) defines the limits and stopping rules.", ""])
    tex = ["% Generated from completed independently verified Bridge 008 observations.",
           r"\paragraph{Correlated source-coverage follow-up.} "+tex_escape(preamble),
           "", tex_escape(finding), "", tex_escape(metric), ""]
    for label, caption, headers, rows in tables:
        tex += [tex_table(label, caption, headers, rows), ""]
    tex += [tex_escape("; ".join(degeneracy)+"."), "", tex_escape(scope), "",
            tex_escape("The complete saved analysis reports every secondary cell estimate and configuration distribution. "
                       "Independent verification reconstructs all four primary intervals and the measured source operators."), ""]
    return "\n".join(md), "\n".join(tex)


def render_stop(verification):
    assert verification["outcome"] == "ineligible_base" and verification["stopping_branch_verified"]
    text = ("Bridge 008 was not measured because the Bridge 007 benchmark did not meet the required clean-qualification gates. "
            "The eligibility decision was verified independently. No source-coverage, norm-matching or signed-interaction result "
            "was obtained. This is a completed eligibility stop, not evidence for or against a source bottleneck, and does not "
            "change Bridge 007's outcome.")
    md = "# Bridge 008 — clean-eligibility stop\n\n"+text+"\n\nSee the [verified stop](results/verification.json) and [protocol](docs/PROTOCOL.md).\n"
    tex = "% Verified Bridge008 eligibility stop; no unmeasured values imputed.\n\\paragraph{Source-coverage follow-up not measured.} "+tex_escape(text)+"\n"
    return md, tex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Fresh report destination; default is the Bridge008 directory")
    args = parser.parse_args()
    verification_path = HERE / "results/verification.json"
    if not verification_path.exists():
        print(json.dumps(dict(status="not_verified", report_written=False))); return
    verification = read(verification_path)
    assert verification["status"] == "passed"
    assert verification["helper_sha256"] == digest(HERE / "code/verify008.py")
    assert verification["computational_dependency_sha256"] == digest(BASE / "code/verify007.py")
    if verification.get("outcome") == "ineligible_base":
        assert digest(HERE / "results/status.json") == verification["status_sha256"]
        markdown, latex = render_stop(verification)
    else:
        summary_path = HERE / "results/analysis/summary.json"
        assert digest(summary_path) == verification["analysis_summary_sha256"]
        assert digest(HERE / "results/inference-complete.json") == verification["inference_completion_sha256"]
        summary = read(summary_path)
        assert summary["primary_multiplicity"] == 4 and summary["draws"] == 50000
        markdown, latex = render(summary, verification)
    destination = args.output or HERE
    md_path, tex_path = destination / "RESULTS.md", destination / "tables/summary.tex"
    assert not md_path.exists() and not tex_path.exists(), "Report exists; use a fresh --output destination"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown)
    tex_path.write_text(latex)
    print(json.dumps(dict(status="written", markdown=str(md_path), latex=str(tex_path))))


if __name__ == "__main__":
    main()
