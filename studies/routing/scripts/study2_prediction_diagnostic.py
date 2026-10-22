"""Post-hoc descriptive prediction baselines; never changes selection or causal gates."""

import numpy as np

from causal_circuit_routing.io import ROOT, digest, now, read_json, write_json


def main():
    selection = read_json(ROOT / "results/study-002/family/selection.json")
    all_rows = read_json(ROOT / "data/study-002-recipients.json")
    family_rows = [r for r in all_rows if r["split"] != "test"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    with np.load(ROOT / "results/study-002/family/measurements.npz") as data:
        family = dict(data)
    with np.load(ROOT / "results/study-002/transfer/natural.npz") as data:
        test = dict(data)
    rows = family_rows + test_rows
    raw = np.concatenate([family["raw"], test["raw"]], axis=1)
    features = np.concatenate([family["features"], test["features"]])
    targets = []
    for bi, baseline in enumerate(("zero", "s2_corrupt")):
        fit = {k: np.array(v) for k, v in selection["signature_fits"][baseline].items()}
        targets.append(
            (raw[bi] / fit["scale"] - fit["centroids"].mean(0))
            @ fit["axis"]
            / np.sum(fit["axis"] ** 2)
        )
    target = np.mean(targets, axis=0)
    discovery = np.array([r["split"] == "discovery" for r in rows])
    labels = np.array([r["family"] == "ABBA" for r in rows])
    means = [target[discovery & (labels == c)].mean() for c in (False, True)]
    selected = selection["selected"]
    probe = (features[:, selected["site_index"]] - np.array(selected["mean_x"])) @ np.array(
        selected["weight"]
    ) + selected["mean_y"]
    predictions = {
        "discovery_grand_mean": np.full(len(rows), target[discovery].mean()),
        "known_prompt_order_mean": np.array(means)[labels.astype(int)],
        "frozen_internal_predictor": probe,
    }
    report = {
        "created_utc": now(),
        "post_hoc": True,
        "note": "Added after test behavior/early causal outputs were viewed. Descriptive only; no selection or primary-gate changes.",
        "splits": {},
        "selection_sha256": digest(ROOT / "results/study-002/family/selection.json"),
    }
    for split in ("discovery", "development", "test"):
        ix = np.array([r["split"] == split for r in rows])
        report["splits"][split] = {}
        for name, prediction in predictions.items():
            mse = np.mean((prediction[ix] - target[ix]) ** 2)
            report["splits"][split][name] = {
                "mse": float(mse),
                "r2": float(1 - mse / np.var(target[ix])),
            }
    write_json(ROOT / "results/study-002/prediction-diagnostic.json", report)
    print(report["splits"]["test"])


if __name__ == "__main__":
    main()
