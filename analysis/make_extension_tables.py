"""Format manuscript snippets from terminal, canonically verified studies.

No inference, fitting, resampling, or numerical recomputation occurs here.
Each --study can run independently while the other studies remain in progress.
The existing row checker supplies the primary number/interval formatter so
rounding and required table rows cannot drift between generator and checker.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from check_extension_numbers import CELLS, check_distributions, check_primary, distribution_rows, interval, num, table
from describe_confirmation import OUTPUT as DISTRIBUTIONS, load_verified as load_distributions

ROOT = Path(__file__).resolve().parents[1]
REFS = (("zero", "zero"), ("fit_mean", "mean"))
CONDITIONS = ("clean", "corrupt", "wording", "self", "zero", "order", "role", "order_role",
              "residual0", "residual1", "residual2", "projected0", "projected1", "projected2")
VARIANTS = (("narrow", "Narrow"), ("broad", "Broad"), ("matched", "Matched broad"))


class NotReady(RuntimeError):
    pass


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(text))


def pointwise(value, percentage=False):
    if value is None or not value.get("valid", True) or value.get("mean") is None:
        return "Not evaluable"
    if not percentage:
        return interval(value, "ci95")
    lo, hi = value["ci95"]
    return f"{100*value['mean']:.1f}" + r"\ci{" + f"{100*lo:.1f}" + "}{" + f"{100*hi:.1f}" + "}"


def rate(value):
    return "Not evaluable" if value is None else f"{100*value:.1f}"


def cell_label(cell, families):
    return f"F{families[int(cell[1])]} / {cell[-1]}"


def latex_table(label, caption, headers, rows, *, column_spec=None, group_header=None):
    """Cells and captions are trusted LaTeX assembled below; data strings escape."""
    assert rows and all(len(row) == len(headers) for row in rows)
    spec = column_spec or ("l" + "Y" * (len(headers)-1))
    lines = [r"\begin{table}[H]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{3pt}", r"\caption{" + caption + "}",
             r"\label{" + label + "}", r"\begin{tabularx}{\linewidth}{@{}" + spec + "@{}}",
             r"\toprule"]
    if group_header:
        lines.extend(group_header)
    lines += [" & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(lines)


def load_completed(number):
    study = ROOT / f"studies/bridge-{number}"
    audit_path = study / "results/verification.json"
    if not audit_path.is_file():
        raise NotReady(f"Bridge {number}: canonical verification is not available")
    audit = read(audit_path)
    assert audit["status"] == "passed", "Canonical verification did not pass"
    assert digest(study / f"code/verify{number}.py") == audit["helper_sha256"]
    sources = {str(audit_path.relative_to(ROOT)): digest(audit_path)}
    if number in ("007", "009"):
        helper = module(study / f"code/report{number}.py")
        try:
            bundle, used = helper.load_completed()
        except helper.NotReady as exc:
            raise NotReady(str(exc)) from exc
        sources.update(used)
        if bundle["outcome"] == "final_test_completed":
            assert digest(study / "results/analysis/summary.json") == audit["analysis_summary_sha256"]
            if number == "007":
                bundle["distributions"] = load_distributions()
                sources.update(bundle["distributions"]["sources_sha256"])
                sources[str(DISTRIBUTIONS.relative_to(ROOT))] = digest(DISTRIBUTIONS)
                sources["analysis/check_extension_numbers.py"] = digest(ROOT / "analysis/check_extension_numbers.py")
        elif number == "007":
            # The current manuscript checker explicitly requires a B007 final
            # test; refuse to manufacture its primary tables on an early stop.
            raise NotReady("Bridge 007 stopped before final testing; the manuscript/checker need an explicit stop branch")
        return bundle, sources

    assert number == "008"
    assert audit["computational_dependency_sha256"] == digest(ROOT / "studies/bridge-007/code/verify007.py")
    freeze_path = study / "data/input-freeze.json"
    assert digest(freeze_path) == audit["checks"]["freeze"]["sha256"]
    freeze = read(freeze_path)
    for name, expected in freeze["files"].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        assert digest(path) == (expected["sha256"] if isinstance(expected, dict) else expected), name
    sources[str(freeze_path.relative_to(ROOT))] = digest(freeze_path)
    if audit.get("outcome") == "ineligible_base":
        status_path = study / "results/status.json"
        assert digest(status_path) == audit["status_sha256"]
        status = read(status_path)
        assert status["status"] == "completed" and status["outcome"] == "ineligible_base"
        assert not status["test_measured"] and audit["stopping_branch_verified"]
        assert not (study / "results/analysis/summary.json").exists()
        sources[str(status_path.relative_to(ROOT))] = digest(status_path)
        return dict(outcome="ineligible_base", verification=audit), sources
    summary_path = study / "results/analysis/summary.json"
    completion_path = study / "results/inference-complete.json"
    assert digest(summary_path) == audit["analysis_summary_sha256"]
    assert digest(completion_path) == audit["inference_completion_sha256"]
    assert read(completion_path)["status"] == "completed"
    summary = read(summary_path)
    assert summary["n"] == 3200 and summary["blocks"] == summary["triples"] == 20
    assert summary["primary_multiplicity"] == 4 and summary["draws"] == 50000
    sources.update({str(path.relative_to(ROOT)): digest(path) for path in (summary_path, completion_path)})
    return dict(outcome="final_test_completed", primary=summary, verification=audit), sources


def render_primary(summary, prefix, population):
    assert summary["primary_multiplicity"] == 16
    rows = [[cell_label(c, summary["families"]), interval(summary["primary"]["recovery"][c]),
             interval(summary["primary"]["margin"]["zero"][c]),
             interval(summary["primary"]["margin"]["fit_mean"][c])] for c in CELLS]
    result = latex_table(prefix + "-primary",
        population + " Twelve primary quantities: logit recovery relative to corruption and normalized "
        "recipient-minus-donor distance margins. Negative margins favor the recipient. These and the four "
        "average bounds share a sixteen-quantity Bonferroni family; brackets are adjusted intervals "
        "from 50,000 crossed context-block/name-triple bootstrap draws.",
        ["Family / order", "Recovery (logits)", "Margin, zero", "Margin, mean"], rows)
    rows = []
    for metric, short, bound in (("mean_absolute_axis_difference", "A", .1),
                                  ("mean_full_signature_distance", "F", .5)):
        assert summary["bounds"][metric] == bound
        for ref, label in REFS:
            value = summary["primary"][metric][ref]
            decision = "Supported" if value.get("valid", True) and value["adjusted_ci"][1] < bound else "Not established"
            rows.append([f"${short}$, {label}", interval(value), f"{bound:.2f}", decision])
    result += latex_table(prefix + "-bounds",
        population + " Four primary bounds on the means of the per-configuration absolute donor-axis "
        "discrepancy $A$ and full-signature discrepancy $F$, with absolute values and vector norms taken before averaging. "
        "Both use matched clean-wording separation units and equal weights for the four family/order cells. "
        "Intervals use the same sixteen-quantity adjustment; support requires the adjusted upper bound below "
        "the stated resolution. These are average assay bounds, not every-cell or algorithmic equivalence.",
        ["Quantity / reference", "Mean [adjusted CI]", "Bound", "Outcome"], rows)
    return result


def render_gates(summary, prefix, population):
    rows = []
    for cell in CELLS:
        values = []
        for stage in ("qualification", "test_qualification"):
            c = summary[stage]["cells"][cell]
            values += [c["pairwise"]["mean"], c["top1"]["mean"], c["classification"]["zero"]["mean"],
                       c["classification"]["fit_mean"]["mean"]]
        rows.append([cell_label(cell, summary["families"])] + [rate(v) for v in values])
    return latex_table(prefix + "-gates",
        population + " Observed clean gate rates (percent). Independent qualification uses 1,152 configurations "
        "from 12 context blocks and 12 disjoint name triples; final checks use 3,200 from 20 by 20. "
        "PW is pairwise answer accuracy; Z/M are clean-signature classification under zero/fit-mean references. "
        "Thresholds are 95\\% PW, 85\\% top-1 and 80\\% classification in every cell/reference; these are "
        "observed rates, not interval lower bounds. All clean paired endpoints must also be resolvable.",
        ["Family / order", "PW", "Top-1", "Z", "M", "PW", "Top-1", "Z", "M"], rows,
        column_spec="l"+"Y"*8,
        group_header=[r" & \multicolumn{4}{c}{Independent qualification} & \multicolumn{4}{c}{Final clean check} \\",
                      r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}"])


def render_heterogeneity(summary, prefix, population):
    rows = []
    for ref, label in REFS:
        for cell in CELLS:
            values = summary["per_cell_heterogeneity"][ref][cell]
            rows.append([cell_label(cell, summary["families"]), label, pointwise(values["signed_axis_difference"]),
                         pointwise(values["absolute_axis_difference"]), pointwise(values["full_signature_distance"])])
    return latex_table(prefix + "-heterogeneity",
        population + " Mandatory cellwise mean signed donor-axis difference, absolute-axis discrepancy $A$ "
        "and full-signature discrepancy $F$ between wording and self transfer, in matched clean-wording "
        "separation units. Signed differences are wording minus self. Every family/order "
        "cell and both references are retained. Brackets are pointwise 95\\% crossed-bootstrap intervals, "
        "without multiplicity adjustment; the pooled primary bounds are not all-cell equivalence claims.",
        ["Family / order", "Reference", r"Signed $\Delta S$ [95\% CI]", "$A$ [95\\% CI]", "$F$ [95\\% CI]"], rows)


def render_controls(secondary, prefix, population):
    rows = []
    labels = {"clean": "Clean", "corrupt": "Corrupted", "wording": "Wording", "self": "Self", "zero": "Zero edit",
              "order": "Order donor", "role": "Role donor", "order_role": "Order+role donor"}
    for name in CONDITIONS:
        value = secondary["behavior"][name]
        label = labels.get(name, name.replace("residual", "Residual draw ").replace("projected", "Projected draw "))
        rows.append([escape(label), pointwise(value["recovery"]["pooled"]),
                     pointwise(value["top1"]["pooled"], True), pointwise(value["pairwise"]["pooled"], True)])
    return latex_table(prefix + "-behavior",
        population + " All declared behavioral conditions, pooled across the same retained final configurations. "
        "Recovery is the IO-minus-subject logit gain over corruption; accuracies and their intervals are percentages. "
        "Brackets are pointwise 95\\% crossed-bootstrap intervals. Random controls use three fixed draws of each "
        "kind, shown separately; neither these draws nor donor role/order symmetries add independent samples.",
        ["Condition", "Recovery [95\\% CI]", "Top-1 [95\\% CI]", "Pairwise [95\\% CI]"], rows)


def render007(bundle):
    summary = bundle["primary"]
    population = ("Bridge 007, GPT-2 Medium F10/F11: 3,200 final configurations from 20 context blocks "
                  "and 20 disjoint name triples.")
    result = render_primary(summary, "tab:confirmation", population)
    result += render_gates(summary, "tab:confirmation", population)
    result += render_heterogeneity(summary, "tab:confirmation", population)
    result += latex_table("tab:confirmation-distributions",
        population + " Empirical configuration-level distributions of absolute donor-axis discrepancy $A$ "
        "and full-signature discrepancy $F$, in matched clean-wording separation units. Columns give "
        "the minimum, 5th/25th/50th/75th/95th percentiles and maximum; percentiles use linear interpolation. "
        "The final column counts values strictly above 0.10 for $A$ or 0.50 for $F$, out of 3,200. "
        "These are descriptive values, with no confidence intervals or per-case pass rule: the declared "
        "bounds concern averages. Every-cell distributions, including signed differences, are retained in "
        r"\protect\path{analysis/results/confirmation-distributions.json}.",
        ["Quantity / ref.", "Min", "P05", "P25", "Median", "P75", "P95", "Max", "$n$ above"],
        distribution_rows(bundle["distributions"]), column_spec="l"+"Y"*8)
    result += render_controls(bundle["secondary"], "tab:confirmation", population)
    return result


def render008(bundle):
    if bundle["outcome"] == "ineligible_base":
        return ("% Verified terminal clean-eligibility stop; no unmeasured outcomes imputed.\n"
                "Bridge 008 was not measured because Bridge 007 did not pass the final clean gate. "
                "No source-coverage interaction was observed; this stopping outcome does not test the "
                "source-coverage explanation.\n")
    p = bundle["primary"]
    decisions = {"donorward_increase": "Increase", "donorward_decrease": "Decrease",
                 "unresolved": "Inconclusive", "non_evaluable": "Not evaluable"}
    population = "Bridge 008 reuses the inspected Bridge 007 F10/F11 benchmark (3,200 configurations, 20 context blocks by 20 disjoint name triples)."
    rows = [[f"{label} / {rl}", interval(p["primary"][variant][ref]), decisions[p["primary_decisions"][variant][ref]]]
            for variant, label in VARIANTS[1:] for ref, rl in REFS]
    result = latex_table("tab:coverage-primary", population +
        r" Four primary interactions $(S_{\rm word}-S_{\rm self})_{\rm coverage}-(S_{\rm word}-S_{\rm self})_{\rm narrow}$, "
        "in clean-wording separation units. Positive values mean an increased donorward differential. "
        "Brackets are four-quantity Bonferroni-adjusted intervals from 50,000 paired crossed-bootstrap draws. "
        "The operational criterion requires all four lower bounds above zero. This is an outcome-informed "
        "correlated follow-up, not independent replication or a test of algorithmic identity.",
        ["Coverage / reference", "Interaction [adjusted CI]", "Decision"], rows)
    rows = []
    for variant, label in VARIANTS[1:]:
        for ref, rl in REFS:
            cells = p["primary_cellwise"][variant][ref]
            for cell in CELLS:
                rows.append([f"{label} / {rl}", cell_label(cell, p["families"]),
                             pointwise(None if cells is None else cells[cell])])
    result += latex_table("tab:coverage-cells", population +
        " All sixteen prespecified source-coverage interactions by recipient family/order and reference. "
        "Units and sign match the primary interaction; brackets are pointwise 95\\% intervals, without "
        "multiplicity adjustment. A pooled effect need not hold in every cell; a signed null does not establish invariance.",
        ["Coverage / reference", "Family / order", "Interaction [95\\% CI]"], rows)
    rows = []
    for variant, label in VARIANTS:
        for donor in ("wording", "self"):
            c = p["conditions"].get(variant+"_"+donor)
            rows.append([label+" / "+donor, pointwise(None if c is None else c["recovery"]),
                         rate(None if c is None else c["top1"]), rate(None if c is None else c["pairwise"])])
    result += latex_table("tab:coverage-behavior", population +
        " Behavioral outcomes for all six coverage/donor conditions. Recovery is the IO-minus-subject logit "
        "gain over corruption, with pointwise 95\\% intervals. Top-1 and pairwise accuracy are percentages. "
        "Broad coverage uses all 112 heads in the same seven S2 layers as the twelve narrow heads. "
        "Matched broad coverage matches the norm of each layer's summed residual edit; disrupted cases remain included.",
        ["Coverage / donor", "Recovery [95\\% CI]", "Top-1", "Pairwise"], rows)
    rows = []
    for variant, label in VARIANTS:
        values = p["secondary_wording_self"][variant]
        for ref, rl in REFS:
            rows.append([label+" / "+rl] + [pointwise(None if values is None else values[ref][metric]["average"])
                for metric in ("signed_axis_difference", "absolute_axis_difference", "full_signature_discrepancy")])
    result += latex_table("tab:coverage-discrepancy", population +
        " Pooled wording-minus-self signed donor-axis differences, mean absolute axis differences and full-signature "
        "discrepancies, in clean-wording separation units. Brackets are pointwise 95\\% intervals. Absolute values "
        "and vector norms are taken before averaging, so signed cancellation is visible.",
        ["Coverage / reference", r"Signed $\Delta S$ [95\% CI]", "$A$ [95\\% CI]", "$F$ [95\\% CI]"], rows)
    rows = []
    for variant, label in VARIANTS:
        for donor in ("wording", "self"):
            c = p["conditions"].get(variant+"_"+donor)
            for ref, rl in REFS:
                rows.append([label+" / "+donor, rl] + [pointwise(None if c is None else c["geometry"][ref][metric]["average"])
                    for metric in ("q_recipient", "q_donor")])
    result += latex_table("tab:coverage-endpoints", population +
        " Pooled edited distances from the clean recipient ($q_R$) and clean donor ($q_D$), each normalized "
        "by their matched clean separation. Brackets are pointwise 95\\% intervals. These endpoint distances "
        "and behavioral recovery constrain interpretation of a donor-axis change; closeness is not algorithm identity.",
        ["Coverage / donor", "Reference", "$q_R$ [95\\% CI]", "$q_D$ [95\\% CI]"], rows)
    diagnostics = []
    for donor in ("wording", "self"):
        match = p["matching"][donor]
        layer_values = list(match["per_layer"].values())
        alpha = [v["alpha"]["quantiles"] for v in layer_values if v["alpha"]["quantiles"] is not None]
        alpha_text = (num(min(v[0] for v in alpha))+"--"+num(max(v[-1] for v in alpha))) if alpha else "undefined"
        diagnostics.append(escape(donor)+": "+str(match["undefined_layers"])+" undefined layer matches across "+
            str(match["undefined_configurations"])+" configurations; observed finite scaling range "+alpha_text)
    result += (r"\noindent\textit{Matching diagnostics.} " + "; ".join(diagnostics) +
        ". Near-zero layers use the frozen $10^{-12}$ rule, and the norm check allows $10^{-6}+10^{-5}\\|\\delta_{\\rm narrow}\\|$. "
        "An undefined match makes its entire matched condition unavailable, with no deletion of cases or reduction of "
        "the four-quantity family. Full layerwise norms, scale distributions, all secondary cells and changes relative "
        "to narrow coverage remain in the saved Bridge 008 analysis.\n")
    return result


def render009(bundle):
    if bundle["outcome"] == "final_test_completed":
        p = bundle["primary"]
        population = ("Bridge 009, GPT-2 Small on fixed F10/F11: the same 3,200 cases as Bridge 007, "
                      "from 20 context blocks and 20 disjoint name triples.")
        return (render_primary(p, "tab:small", population) + render_gates(p, "tab:small", population) +
                render_heterogeneity(p, "tab:small", population) + render_controls(bundle["secondary"], "tab:small", population))
    screen = bundle["screen"]
    rows = [[f"F{family} / {order}", rate(screen["cells"][str(family)][str(order)]["pairwise"]),
             rate(screen["cells"][str(family)][str(order)]["top1"])] for family in (10, 11) for order in (0, 1)]
    result = latex_table("tab:small-gates",
        "Bridge 009: observed GPT-2 Small behavioral screen rates (percent) for the fixed F10/F11 pair, "
        "reusing 96 Bridge 007 configurations from three context blocks and four disjoint name triples. "
        "Each family must reach 95\\% pairwise and 85\\% top-1 in both orders. "
        "No model-specific replacement pair is permitted. These are screening rates, not final transfer outcomes.",
        ["Family / order", "Pairwise", "Top-1"], rows)
    outcome = bundle["outcome"]
    if outcome == "no_selection_pair":
        selected = bundle["selection"]["scores"]["10,11"]
        rows = [[cell_label(cell, [10, 11])] + [rate(v) for v in selected["fit_separation"][cell] + selected["selection_separation"][cell]] for cell in CELLS]
        result += latex_table("tab:small-selection",
            "Bridge 009: all fixed-pair clean-classification rates (percent), with zero and fit-mean references "
            "at fitting and selection, each using 512 configurations from eight contexts and eight disjoint name triples. "
            "Every fitting cell/reference must reach 80\\% and every selection cell/reference "
            "90\\%; behavior gates also apply. This clean-selection stop leaves independent qualification and final "
            "transfer unmeasured.",
            ["Family / order", "Fit, zero", "Fit, mean", "Selection, zero", "Selection, mean"], rows)
    elif outcome == "independent_qualification_failed":
        q = bundle["qualification"]
        rows = []
        for cell in CELLS:
            c = q["cells"][cell]
            values = [c["pairwise"]["mean"], c["top1"]["mean"], c["classification"]["zero"]["mean"], c["classification"]["fit_mean"]["mean"]]
            rows.append([cell_label(cell, [10, 11])] + [rate(v) for v in values])
        result += latex_table("tab:small-qualification",
            "Bridge 009: observed independent clean-qualification rates (percent), from 1,152 configurations "
            "crossing 12 contexts and 12 disjoint name triples. Thresholds are 95\\% pairwise, 85\\% top-1 and "
            "80\\% classification under each reference in every cell, with resolvable paired endpoints also required. "
            "The failed qualification leaves final transfer unmeasured.",
            ["Family / order", "Pairwise", "Top-1", "Classification, zero", "Classification, mean"], rows)
    else:
        assert outcome == "no_screen_pair"
    result += ("The fixed-pair Small attempt stopped at its prespecified clean gate. It supplies no qualified final "
               "transfer comparison and does not test the unmeasured transfer predictions.\n")
    return result


def validate_rows(source, number, bundle):
    """Check generated strings in memory, before any output is written."""
    checks = []
    if number in ("007", "009") and bundle["outcome"] == "final_test_completed":
        check_primary(source, bundle["primary"], "tab:confirmation" if number == "007" else "tab:small", checks)
        if number == "007":
            check_distributions(source, bundle["distributions"], checks)
    elif number == "008" and bundle["outcome"] == "final_test_completed":
        p = bundle["primary"]
        body = table(source, "tab:coverage-primary")
        decisions = {"donorward_increase": "Increase", "donorward_decrease": "Decrease", "unresolved": "Inconclusive", "non_evaluable": "Not evaluable"}
        for variant, label in VARIANTS[1:]:
            for ref, rl in REFS:
                row = " & ".join([f"{label} / {rl}", interval(p["primary"][variant][ref]), decisions[p["primary_decisions"][variant][ref]]])
                checks.append((row in body, "coverage primary", row))
        assert table(source, "tab:coverage-cells").count(r" \\") == 17  # header + 16 cells
    elif number == "008":
        assert "did not pass the final clean gate" in source
    failed = [entry for entry in checks if not entry[0]]
    assert not failed, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("007", "008", "009", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/sections")
    parser.add_argument("--check-ready", action="store_true", help="Validate evidence and render in memory only")
    args = parser.parse_args()
    results = {}
    for number in (("007", "008", "009") if args.study == "all" else (args.study,)):
        try:
            bundle, sources = load_completed(number)
        except NotReady as exc:
            results[number] = dict(status="not_ready", reason=str(exc), written=False)
            continue
        content = globals()["render"+number](bundle)
        validate_rows(content, number, bundle)
        provenance = [f"% Bridge {number}: generated only from completed canonical passed verification.",
                      "% Formatting only; primary intervals adjusted, secondary intervals pointwise as captioned.",
                      "% Requires the manuscript's booktabs/tabularx/Y column and \\ci macro.",
                      "% Generator SHA256: " + digest(Path(__file__))]
        provenance += [f"% Source SHA256 {name}: {value}" for name, value in sorted(sources.items())]
        content = "\n".join(provenance)+"\n\n"+content.rstrip()+"\n"
        target = args.output_dir / f"extension-tables-{number}.tex"
        if args.check_ready:
            state = "ready"
        elif target.exists():
            assert target.read_text() == content, f"Different existing snippet: {target}; use a fresh --output-dir for review"
            state = "unchanged"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".partial")
            temporary.write_text(content)
            temporary.replace(target)
            state = "written"
        results[number] = dict(status=state, outcome=bundle["outcome"], path=str(target), written=state == "written")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
