"""Secondary stratified displays and algebraic symmetry audit; no new selection."""

import argparse

import numpy as np
from assay import load_arrays
from design import HERE

from transport_bridge.common import ROOT, digest, now, read, write
from transport_bridge.statistics import block_interval


def main(model):
    directory = HERE / "results" / model
    rows = [r for r in read(HERE / "data/recipients.json") if r["split"] == "test"]
    index = {r["id"]: i for i, r in enumerate(rows)}
    mirror = np.array([index[r["donors"]["order_role"]] for r in rows])
    for i, j in enumerate(mirror):
        assert rows[i]["corrupt_tokens"] == rows[j]["corrupt_tokens"]
        assert rows[i]["io_id"] == rows[j]["subject_id"]
    natural = load_arrays(directory / "test/natural.npz")
    selected = read(directory / "fit/selection.json")
    metrics = load_arrays(directory / "analysis/per-example.npz")
    cfg = read(ROOT / "configs/study.json")["models"][model]
    record = {
        "scope": "Post-freeze secondary breakdowns, no revised axes or selected heads",
        "symmetry": {},
        "joint_by_cell": {},
        "head_changes": {},
    }
    for a, b in [("head_self", "head_order_role"), ("head_role", "head_order")]:
        error = float(
            abs(metrics[a + "__behavior"][mirror, 0] + metrics[b + "__behavior"][:, 0]).max()
        )
        assert error <= 0.0005
        record["symmetry"][a + "_versus_" + b] = error
    baseline_error = float(
        abs(natural["corrupt_behavior"][mirror, 0] + natural["corrupt_behavior"][:, 0]).max()
    )
    assert baseline_error <= 0.0005
    record["symmetry"]["corrupt_label_reversal"] = baseline_error

    def interval(x, mask=None):
        return block_interval(x, rows, mask, multiplicity=1)

    def show(v):
        lo, hi = v["ci95"]
        return f"{v['mean']:.3f} [{lo:.3f}, {hi:.3f}]"

    lines = [
        f"# {model}: secondary diagnostic breakdowns",
        "",
        "These displays were added after the Small test results were available. They use retained per-example measurements and fixed discovery-selected heads. They do not revise the primary analysis or claim additional multiplicity-adjusted discoveries.",
        "",
        "## Structural symmetries",
        "",
        "After the repeated subject is replaced by a third name, opposite-order/role-swapped recipients have identical corrupted input tokens but reversed IO/subject scoring labels. Hence pooled corrupted signed logit effects cancel exactly, and the own/order-plus-role and role/order transfers have paired sign symmetries. These are algebraic properties of the design, not independent confirmations or evidence that corrupted computations have no interactions.",
        "",
        f"Maximum corrupted-label reversal error: {baseline_error:.3g}. Corresponding transfer symmetries were checked against the raw arrays.",
        "",
        "## Joint effects by recipient cell",
        "",
        "The pooled corrupted non-additivity of zero in the primary summary is therefore uninformative about its within-cell magnitude. The table below retains the prespecified wording/order strata. All intervals are secondary pointwise 95% intervals.",
        "",
        "| Edit | Reference | Recipient cell | Corrupted non-additivity [95%] | Edited non-additivity [95%] | Change [95%] |",
        "|---|---|---|---|---|---|",
    ]
    for f in ["self", "wording", "order"]:
        record["joint_by_cell"][f] = {}
        for baseline in ["zero", "discovery_mean"]:
            a = metrics[f + "__joint_nonadditivity__" + baseline]
            cells = {}
            for w in range(2):
                for o in range(2):
                    key = f"w{w}_o{o}"
                    mask = np.array([r["wording"] == w and r["order"] == o for r in rows])
                    cells[key] = {
                        "corrupted": interval(a[0], mask),
                        "edited": interval(a[1], mask),
                        "change": interval(a[1] - a[0], mask),
                    }
                    v = cells[key]
                    lines.append(
                        f"| {f} | {baseline} | {key} | {show(v['corrupted'])} | {show(v['edited'])} | {show(v['change'])} |"
                    )
            record["joint_by_cell"][f][baseline] = cells
    lines += [
        "",
        "## Individual heads within the fixed selected groups",
        "",
        "Attention is descriptive. Each dependency change is edited-minus-corrupted individual-head logit effect under the stated reference. The main study clamps the group jointly; these observations do not assign its query/key/value clamp effects uniquely to individual heads.",
        "",
        "| Edit | Head | Dependency change zero [95%] | Dependency change mean [95%] | Attention change to IO [95%] | Attention change to S2 [95%] |",
        "|---|---|---|---|---|---|",
    ]
    for f in ["self", "wording", "order"]:
        edited = load_arrays(directory / "test" / ("head_" + f + ".npz"))
        trace = load_arrays(directory / "test" / ("trace_" + f + ".npz"))
        changes = {}
        for i, (layer, head) in enumerate(selected["groups"][f]["selected"]):
            hi = cfg["layers"].index(layer) * cfg["heads"] + head
            d = edited["raw"][:, :, hi, 0] - natural["corrupt_raw"][:, :, hi, 0]
            attention = trace["attention"][1, :, i] - trace["attention"][0, :, i]
            entry = {
                "dependency_change": [interval(d[bi]) for bi in range(2)],
                "attention_change": [interval(attention[:, j]) for j in range(3)],
            }
            changes[f"{layer}.{head}"] = entry
            lines.append(
                f"| {f} | {layer}.{head} | {show(entry['dependency_change'][0])} | {show(entry['dependency_change'][1])} | {show(entry['attention_change'][0])} | {show(entry['attention_change'][2])} |"
            )
        record["head_changes"][f] = changes
    output = directory / "report"
    output.mkdir(exist_ok=True)
    (output / "DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")
    write(output / "diagnostics.json", record)
    write(
        output / "diagnostics-provenance.json",
        {
            "utc": now(),
            "helper_sha256": digest(__file__),
            "primary_summary_sha256": digest(directory / "analysis/summary.json"),
            "outputs": {
                name: digest(output / name) for name in ["DIAGNOSTICS.md", "diagnostics.json"]
            },
        },
    )
    print(output / "DIAGNOSTICS.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    main(p.parse_args().model)
