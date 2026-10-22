"""Render completed Bridge 009 summaries without inference or statistical fitting.

This reporting helper was added after the scientific input freeze. It changes
no frozen file or scientific decision. In-progress and partially verified
studies are refused; genuine early stops receive their own complete reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PAIRS = [[10, 11]]
REFS = ("zero", "fit_mean")
CELLS = ("w0_o0", "w0_o1", "w1_o0", "w1_o1")
FINAL = "final_test_completed"
STOPS = {"no_screen_pair", "no_selection_pair", "independent_qualification_failed"}
OUTCOMES = STOPS | {FINAL}


class NotReady(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equivalent(a, b):
    """Compare existing audit values; no estimator is recalculated here."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equivalent(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equivalent(x, y)
    elif a is None or isinstance(a, (bool, str)):
        assert a == b
    else:
        assert math.isclose(a, b, rel_tol=0, abs_tol=2e-10), (a, b)


def load_completed():
    used = {}

    def get(relative):
        path = HERE / relative
        if not path.is_file():
            raise NotReady(f"Required completed record missing: {relative}")
        value = read(path)
        used[str(path.relative_to(ROOT))] = digest(path)
        return value

    status = get("results/status.json")
    if status.get("status") not in ("completed", "inference_completed"):
        raise NotReady("Pipeline is still running or has no completed stopping branch")
    outcome = status.get("outcome")
    if outcome not in OUTCOMES:
        raise NotReady(f"Unknown or incomplete pipeline outcome: {outcome}")
    assert bool(status["test_measured"]) == (outcome == FINAL)
    verification = get("results/verification.json")
    assert verification["status"] == "passed"
    assert verification["outcome"] == outcome and verification["stopping_branch_verified"]
    assert datetime.fromisoformat(verification["utc"]) >= datetime.fromisoformat(status["utc"])
    # Bind the rendered decisions to the same unchanged freezes checked by the
    # independent verifier. Hashing is the only processing of numerical files.
    for name, record in verification["checks"]["freezes"].items():
        path = HERE / "data" / name
        assert digest(path) == record["sha256"], path
        frozen = get("data/" + name)
        for filename, expected in frozen["files"].items():
            if isinstance(expected, dict):
                expected = expected["sha256"]
            target = Path(filename) if Path(filename).is_absolute() else ROOT / filename
            assert digest(target) == expected, target
    bundle = dict(status=status, outcome=outcome, verification=verification,
                  design=get("data/design.json"), preparation=get("data/preparation-audit.json"),
                  screen=get("results/screen.json"),
                  calibration=get("results/calibration.json"))
    assert bundle["design"]["candidate_pairs"] == PAIRS
    assert bundle["design"]["model"] == "gpt2-small"
    assert bundle["design"]["bank_family_ids"] == [10,11,12,13]
    assert bundle["preparation"]["fixed_pair"] == [10,11]
    assert bundle["preparation"]["fit_mean_bank_rows"] == 1024
    assert bundle["preparation"]["fit_signature_rows"] == 512
    for name, expected in bundle["preparation"]["inherited_source_hashes"].items():
        assert digest(ROOT / name) == expected, name
    assert bundle["calibration"]["passed"]
    if outcome == "no_screen_pair":
        assert not bundle["screen"]["pairs"]
        assert not (HERE / "data/selection.json").exists()
    else:
        selection = bundle["selection"] = get("data/selection.json")
        if outcome == "no_selection_pair":
            assert selection["status"] == "no_qualified_pair" and selection["families"] is None
        else:
            assert selection["status"] == "selected" and selection["families"] in PAIRS
            qualification = bundle["qualification"] = get("results/qualification-summary.json")
            replay = bundle["selected_replay"] = get("results/selected-pair-replay.json")
            assert replay["passed"]
            if outcome == "independent_qualification_failed":
                assert not qualification["qualified"]
            else:
                assert qualification["qualified"]
    if outcome != FINAL:
        assert not list((HERE / "results/test").glob("*.npz")), "Early stop has final-test observations"
        assert not (HERE / "results/analysis/summary.json").exists(), "Early stop has final analysis"
        return bundle, used

    primary = bundle["primary"] = get("results/analysis/summary.json")
    secondary = bundle["secondary"] = get("results/secondary-analysis/summary.json")
    computational = bundle["computational_replay"] = get("results/computational-replay.json")
    completion = get("results/inference-complete.json")
    assert completion["status"] == "completed" and computational["passed"]
    assert primary["families"] == secondary["families"] == bundle["selection"]["families"]
    assert primary["n"] == secondary["n"] == 3200
    assert primary["blocks"] == secondary["blocks"] == 20
    assert primary["triples"] == secondary["triples"] == 20
    assert primary["primary_multiplicity"] == 16
    assert primary["model"] == secondary["model"] == "gpt2-small"
    assert primary["signature_coordinates"] == secondary["signature_coordinates"] == 60
    assert secondary["source_heads"] == 4 and secondary["seed"] == 2026100409
    equivalent(primary["qualification"], bundle["qualification"])
    assert primary["bounds"] == {"mean_absolute_axis_difference": .10,
                                  "mean_full_signature_distance": .50}
    assert verification["checks"]["primary"]["endpoints"] == 16
    for path, checked in verification["checks"]["primary"]["reproduced"].items():
        value = primary["primary"]
        for key in path.split("/"):
            value = value[key]
        for key in ("mean", "ci95", "adjusted_ci", "valid"):
            equivalent(value[key], checked[key])
    equivalent(verification["checks"]["final_decisions"],
               {key: primary[key] for key in ("qualified", "outcomes", "full_conjunction_supported")})
    for summary in (primary, secondary):
        for name, expected in summary["input_hashes"].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            assert digest(path) == expected, path
    secondary_freeze = get("data/secondary-analysis-freeze.json")
    assert secondary["secondary_analysis_freeze_sha256"] == digest(HERE / "data/secondary-analysis-freeze.json")
    assert secondary_freeze["planned_outputs"]["scope"]
    for name in ("clean", "corrupt", "wording", "self", "zero", "order", "role", "order_role",
                 "residual0", "residual1", "residual2", "projected0", "projected1", "projected2"):
        assert name in secondary["behavior"], name
    return bundle, used


def number(value, percent=False, digits=3):
    if value is None:
        return "not estimable"
    assert math.isfinite(value)
    return f"{100*value:.1f}%" if percent else f"{value:.{digits}f}"


def interval(value, adjusted=False, percent=False):
    if not value.get("valid", True) or value["mean"] is None:
        return "not estimable (unresolved values retained)"
    bounds = value["adjusted_ci" if adjusted else "ci95"]
    assert bounds is not None and len(bounds) == 2
    return f"{number(value['mean'], percent)} [{number(bounds[0], percent)}, {number(bounds[1], percent)}]"


def yes(value):
    return "pass" if value else "fail"


def pair_label(pair):
    return "/".join(f"F{x}" for x in pair)


def cell_label(cell, families):
    return f"F{families[int(cell[1])]} / {cell[-1]}"


def ref_label(ref):
    return "zero" if ref == "zero" else "fit mean"


def tex_escape(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(ch, ch) for ch in str(text))


class Document:
    def __init__(self):
        self.md = ["# Bridge 009 results", ""]
        self.tex = ["% Generated by report009.py from completed, verified saved summaries.",
                    "% Fragment for the existing manuscript; requires booktabs, tabularx and array.",
                    r"\subsection*{Bridge 009: fixed-pair GPT-2 Small replication}", ""]
        self.tables = []

    def paragraph(self, text, latex=True):
        self.md += [text, ""]
        if latex:
            self.tex += [tex_escape(text), ""]

    def section(self, text):
        self.md += ["## " + text, ""]
        self.tex += [r"\paragraph{" + tex_escape(text) + "}", ""]

    def table(self, label, title, headers, rows, note=None, latex=True):
        assert rows and all(len(row) == len(headers) for row in rows)
        self.tables.append(dict(label=label, title=title, rows=len(rows)))
        self.md += [f"**{title}**", "", "| " + " | ".join(headers) + " |",
                    "| " + " | ".join("---" for _ in headers) + " |"]
        self.md += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
        self.md.append("")
        if note:
            self.md += [note, ""]
        if not latex:
            return
        caption = title + (". " + note if note else "")
        self.tex += [r"\begin{table}[htbp]", r"\centering\footnotesize",
                     r"\caption{" + tex_escape(caption) + "}",
                     r"\label{tab:b009-" + label + "}",
                     r"\begin{tabularx}{\linewidth}{@{}" + "Y"*len(headers) + "@{}}",
                     r"\toprule", " & ".join(tex_escape(x) for x in headers) + r" \\", r"\midrule"]
        self.tex += [" & ".join(tex_escape(x) for x in row) + r" \\" for row in rows]
        self.tex += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]


def qualification_table(doc, q, families, title, label):
    doc.table(label, title,
              ["Family / order", "Pairwise", "Top-1", "Classification, zero", "Classification, fit mean"],
              [[cell_label(cell, families), interval(q["cells"][cell]["pairwise"], percent=True),
                interval(q["cells"][cell]["top1"], percent=True),
                interval(q["cells"][cell]["classification"]["zero"], percent=True),
                interval(q["cells"][cell]["classification"]["fit_mean"], percent=True)] for cell in CELLS],
              "Observed rates with pointwise 95% crossed-bootstrap intervals. Gates apply to observed rates, not interval bounds.")
    doc.paragraph(f"{title}: {yes(q['qualified'])}. Behavior gate: {yes(q['behavior_pass'])}; "
                  f"clean separation gate: {yes(q['separation_pass'])}; unresolved reference/example endpoints: "
                  f"{q['unresolvable_reference_examples']}. Sampling: {q['blocks']} context blocks by "
                  f"{q['triples']} name triples, {q['n']} configurations.")


def render(bundle):
    doc = Document()
    outcome = bundle["outcome"]
    texts = {
        "no_screen_pair": "Completed bounded study: fixed F10/F11 failed behavioral screening in GPT-2 Small. Fitting, selection, independent qualification and final transfer were not measured.",
        "no_selection_pair": "Completed bounded study: fixed F10/F11 failed the clean fitting or selection requirements in GPT-2 Small. Independent qualification and final transfer were not measured.",
        "independent_qualification_failed": "Completed bounded study: fixed F10/F11 failed independent clean qualification in GPT-2 Small. The final bank and the self-versus-wording transfer predictions were not measured.",
    }
    if outcome == FINAL:
        main = bundle["primary"]
        doc.paragraph("Completed final test: the full prespecified conjunction is " +
                      ("supported." if main["full_conjunction_supported"] else "not supported.") +
                      " The clean gates and four component decisions are reported separately below.")
    else:
        doc.paragraph(texts[outcome])
    doc.paragraph("This is a fixed-pair GPT-2 Small replication attempt using F10/F11 selected in Medium. "
                  "It reuses the exact Bridge 007 cases and permits no Small-specific wording search. "
                  "Its local prospective freeze is not public preregistration. The original self-versus-wording analysis remains post hoc.")
    doc.paragraph("The prespecified assay has four inherited source heads (0.1, 3.0, 5.5, 6.9) and 60 END-head coordinates in layers 7–11. "
                  "The fitting-mean rule specifies all four fixed Medium families and 1,024 clean fitting rows, with 512 F10/F11 signatures in each fitting and selection bank. "
                  "F12/F13 cannot enter Small screening, eligibility or transfer decisions.")
    if outcome == "no_screen_pair":
        doc.paragraph("These are planned scientific fitting counts, not measured observations: only numerical calibration and behavioral screening were reached.")
    doc.section("Behavioral screen and fixed-pair disposition")
    screen = bundle["screen"]
    rows = []
    for family in (10, 11):
        for order in (0, 1):
            c = screen["cells"][str(family)][str(order)]
            rows.append([f"F{family} / {order}", str(c["n"]), number(c["pairwise"], True),
                         number(c["top1"], True), "eligible" if family in screen["eligible"] else "ineligible"])
    doc.table("screen", "Both fixed families at behavioral screening",
              ["Family / order", "n", "Pairwise", "Top-1", "Family decision"], rows,
              "Each family must reach 95% pairwise and 85% top-1 accuracy in both orders.")
    selection = bundle.get("selection")
    rows = []
    for pair in PAIRS:
        key = ",".join(map(str, pair))
        if selection is not None and key in selection["scores"]:
            s = selection["scores"][key]
            minimum_fit = min(value for rates in s["fit_separation"].values() for value in rates)
            decision = "eligible fixed pair" if selection["families"] == pair else ("eligible" if s["eligible"] else "ineligible")
            rows.append([pair_label(pair), number(minimum_fit, True), number(s["worst"], True),
                         number(s["average"], True), decision])
        else:
            assert pair not in screen["pairs"]
            rows.append([pair_label(pair), "not measured", "not measured", "not measured", "screen ineligible"])
    doc.table("selection", "Sole fixed pair and observed clean-selection scores",
              ["Pair", "Worst fitting cell", "Worst selection cell", "Mean selection cell", "Decision"], rows,
              "Unmeasured entries identify actual screen exclusions. Eligibility additionally requires fit/selection behavior gates; separation thresholds are 80% fitting and 90% selection. F10/F11 is the sole permitted pair; no model-specific ranking or replacement is possible.")
    if selection is not None:
        measured = [s for s in selection["scores"].values()]
        if measured:
            rows = []
            for s in measured:
                def passes_behavior(cells):
                    return all(c["pairwise"] >= .95 and c["top1"] >= .85 for family in cells.values() for c in family.values())
                rows.append([pair_label(s["families"]), yes(passes_behavior(s["fit_behavior"])),
                             yes(passes_behavior(s["selection_behavior"])), yes(s["eligible"])])
            doc.table("selection-gates", "Behavior gates and full eligibility for measured pairs",
                      ["Pair", "Fit behavior", "Selection behavior", "All eligibility requirements"], rows)
    if "qualification" in bundle:
        doc.section("Independent qualification")
        families = selection["families"]
        doc.paragraph(f"The Medium-selected fixed pair {pair_label(families)} passed Small fitting/selection eligibility before independent qualification. "
                      "No reference was refitted on qualification observations and no replacement pair was selected.")
        qualification_table(doc, bundle["qualification"], families, "Independent clean qualification", "qualification")
    if outcome == FINAL:
        render_final(doc, bundle)
    doc.section("Interpretation and generality limits")
    if outcome != FINAL:
        doc.paragraph("This completed stopping outcome does not test the unmeasured final transfer hypothesis. "
                      "It supplies no additional qualified final-transfer comparison and does not change the prior Medium results. "
                      "No thresholds, names, contexts, source heads or pairs were replaced to obtain a favorable result.")
    else:
        doc.paragraph("The average bounds describe this assay's specified resolutions. They do not establish every-cell "
                      "or every-example similarity, exact signature identity, equivalent algorithms, or absence of donor influence. "
                      "Failure to establish a bound is not automatically evidence that the population discrepancy exceeds it.")
    doc.paragraph("The evidence concerns a second GPT-2 checkpoint under a Medium-selected wording comparison, model-specific historical source heads "
                  "and fitting references. It does not establish generality across architectures or estimate selection-pipeline uncertainty. "
                  "The Small and Medium cases are identical and correlated; their independent language sample counts cannot be added. "
                  "Source-set size and signature dimensions also differ, so model size alone cannot explain any difference. "
                  "Context blocks and disjoint name triples are the crossed sampling dimensions; paired roles, orders and control draws "
                  "do not supply additional independent samples.")
    verification = bundle["verification"]
    doc.section("Verification and provenance")
    doc.paragraph(f"Independent saved-observation verification passed for the {outcome} branch at {verification['utc']}. "
                  "It checked the stopping branch, unchanged freezes, measurement records, numerical calibration and the applicable "
                  "selection/qualification computations. For a final test it independently reproduces all sixteen primary intervals "
                  "and decisions; secondary calculations are separately identified in their summary and implementation freeze.")
    doc.paragraph("This report only formats saved summaries. Its source-summary hashes and the report-helper hash are recorded in "
                  "results/report-manifest.json. No model inference, resampling, source selection or statistical fitting occurs here.")
    return "\n".join(doc.md), "\n".join(doc.tex), doc.tables


def render_final(doc, bundle):
    p, s = bundle["primary"], bundle["secondary"]
    families, primary = p["families"], p["primary"]
    doc.section("Final clean checks and all primary outcomes")
    qualification_table(doc, p["test_qualification"], families, "Final-bank clean qualification", "test-qualification")
    doc.paragraph("The final test contains 3,200 configurations from 20 context blocks and 20 disjoint name triples. "
                  "All sixteen primary quantities use 50,000 crossed-bootstrap draws (seed 2026100409) and one within-study Bonferroni family (tails 0.05/(2*16)). "
                  "The next two tables show adjusted intervals; all other intervals in this report are pointwise 95% unless stated.")
    doc.table("primary-repair", "Twelve primary recovery and recipient-proximity quantities",
              ["Family / order", "Recovery (logits)", "Margin, zero", "Margin, fit mean"],
              [[cell_label(cell, families), interval(primary["recovery"][cell], adjusted=True),
                interval(primary["margin"]["zero"][cell], adjusted=True),
                interval(primary["margin"]["fit_mean"][cell], adjusted=True)] for cell in CELLS],
              "Recovery lower bounds must exceed zero. Margin upper bounds must be below zero; negative margins favor the recipient.")
    rows = []
    for metric, label in (("mean_absolute_axis_difference", "Mean absolute axis difference"),
                          ("mean_full_signature_distance", "Mean full signature discrepancy")):
        bound = p["bounds"][metric]
        for ref in REFS:
            v = primary[metric][ref]
            passes = bool(v["valid"] and v["adjusted_ci"][1] < bound)
            rows.append([label, ref_label(ref), interval(v, adjusted=True), f"upper < {bound:.2f}",
                         "supported" if passes else "not established"])
    doc.table("primary-similarity", "Four prospective wording-versus-self average-bound quantities",
              ["Quantity", "Reference", "Mean [adjusted interval]", "Criterion", "Outcome"], rows,
              "Distances use matched clean-wording separation units. Absolute differences are taken before averaging; these are pooled average bounds, not all-cell equivalence.")
    labels = {
        "all_recovery_intervals_positive": "All four recovery intervals positive",
        "all_margin_intervals_recipient": "All eight margins favor recipient",
        "average_absolute_axis_bound_supported": "Both absolute-axis bounds supported",
        "average_full_distance_bound_supported": "Both full-discrepancy bounds supported",
    }
    doc.table("decisions", "Separate component decisions",
              ["Component", "Outcome"],
              [[label, "supported" if p["outcomes"][key] else "not established"] for key, label in labels.items()]
              + [["All clean gates and resolvable endpoints", yes(p["qualified"])],
                 ["Complete prespecified conjunction", "supported" if p["full_conjunction_supported"] else "not supported"]])
    doc.section("All-cell wording-versus-self heterogeneity")
    rows = []
    for ref in REFS:
        for cell in CELLS:
            h = p["per_cell_heterogeneity"][ref][cell]
            rows.append([cell_label(cell, families), ref_label(ref), interval(h["signed_axis_difference"]),
                         interval(h["absolute_axis_difference"]), interval(h["full_signature_distance"]),
                         interval(h["wording_minus_self_recovery"])])
    doc.table("heterogeneity", "Every family/order cell under both references",
              ["Family / order", "Reference", "Signed axis", "Absolute axis", "Full discrepancy", "Recovery difference (logits)"],
              rows, "Wording minus self, except that absolute and full discrepancies are nonnegative. All intervals are pointwise 95%.")
    doc.section("Behavioral controls and paired comparisons")
    order = ["clean", "corrupt", "wording", "self", "zero", "order", "role", "order_role",
             "residual0", "residual1", "residual2", "projected0", "projected1", "projected2"]
    doc.table("behavior", "All declared behavioral conditions",
              ["Condition", "Logit recovery", "Top-1", "Pairwise"],
              [[name, interval(s["behavior"][name]["recovery"]["pooled"]),
                interval(s["behavior"][name]["top1"]["pooled"], percent=True),
                interval(s["behavior"][name]["pairwise"]["pooled"], percent=True)] for name in order],
              "Recovery is relative to corrupted behavior. Pointwise 95% intervals; random controls have three fixed draws of each kind. Order/role symmetries are not independent corroborations.")
    doc.table("control-contrasts", "Paired wording-minus-control differences",
              ["Comparator", "Logit difference", "Top-1 difference"],
              [[name.removeprefix("wording_minus_"), interval(value["logit"]["pooled"]),
                interval(value["top1"]["pooled"], percent=True)]
               for name, value in s["paired_behavior_contrasts"].items()],
              "Top-1 differences are percentage points. The two mean controls average their three fixed draws within configuration before comparison; intervals are pointwise 95%.")
    doc.table("restored-fraction", "Fraction of the clean logit gap restored",
              ["Condition", "Restored fraction"],
              [[name, interval(s["recovery_fraction"][name]["pooled"])] for name in ("wording", "self")],
              "Ratio of mean recovery to mean clean-minus-corrupted logit gap; paired bootstrap ratio intervals.")
    doc.section("Secondary distance and movement checks")
    rows = []
    for metric in ("unscaled_paired_l2", "standardized_cosine", "centroid_normalized_l2"):
        for ref in REFS:
            rows.append([metric.replace("_", " "), ref_label(ref),
                         interval(s["alternative_distances"]["wording"][metric][ref]["pooled"]),
                         interval(s["alternative_distances"]["self"][metric][ref]["pooled"])])
    doc.table("alternative-distances", "Three declared alternative distance margins",
              ["Metric", "Reference", "Wording transfer", "Self transfer"], rows,
              "Pooled pointwise 95% intervals; negative margins favor the recipient. Per-cell results remain in the secondary summary. These sensitivities do not replace the primary metric.")
    rows = []
    for metric in ("progress_toward_recipient", "recipient_distance_closed", "cosine_with_ideal_repair", "perpendicular_fraction"):
        for ref in REFS:
            rows.append([metric.replace("_", " "), ref_label(ref),
                         interval(s["movement"]["wording"][metric][ref]["pooled"]),
                         interval(s["movement"]["self"][metric][ref]["pooled"])])
    doc.table("movement", "Movement relative to corruption and full recipient repair",
              ["Quantity", "Reference", "Wording transfer", "Self transfer"], rows,
              "Pooled pointwise 95% intervals. The secondary summary also contains the ideal-repair perpendicular baseline and all cellwise values.")
    doc.table("injected-deltas", "Actual injected wording/self differences",
              ["Quantity", "Pooled estimate [pointwise 95% interval]"],
              [[key.replace("_", " "), interval(s["injected_delta"][key]["pooled"])]
               for key in ("self_norm", "wording_norm", "relative_difference", "cosine")],
              "Norms flatten the four source-head output differences; relative difference is divided by the self-delta norm.")


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text.rstrip() + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-ready", action="store_true", help="Validate completion and hashes without writing reports")
    args = parser.parse_args()
    try:
        bundle, sources = load_completed()
    except NotReady as exc:
        print(json.dumps(dict(status="not_ready", report_written=False, reason=str(exc))))
        return 2
    md, tex, tables = render(bundle)
    if args.check_ready:
        print(json.dumps(dict(status="ready", outcome=bundle["outcome"], report_written=False, tables=len(tables))))
        return 0
    paths = [HERE / "RESULTS.md", HERE / "tables/summary.tex"]
    for path, text in zip(paths, (md, tex)):
        atomic_text(path, text)
    manifest = dict(schema=1, utc=datetime.now(timezone.utc).isoformat(), outcome=bundle["outcome"],
                    status="completed_report", scientific_inference=False, statistical_analysis_rerun=False,
                    helper_sha256=digest(Path(__file__)), source_summaries=sources,
                    output_hashes={str(p.relative_to(HERE)): digest(p) for p in paths},
                    tables=tables,
                    scope="Formatting of completed saved summaries; frozen sources and observations unchanged")
    atomic_text(HERE / "results/report-manifest.json", json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(dict(status="reported", outcome=bundle["outcome"], tables=len(tables),
                          outputs=[str(p) for p in paths])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
