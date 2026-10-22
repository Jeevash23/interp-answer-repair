"""Descriptive heatmaps of the frozen raw effects; no fitted dimensions or selection."""

import os

from causal_circuit_routing.io import ROOT, digest, load_records, read_json

os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    run = ROOT / "results/pilot-001"
    metadata = read_json(run / "run.json")
    if metadata["status"] != "completed":
        raise RuntimeError("No completed signature sweep to plot")
    if digest(run / "signatures.npz") != metadata["result_sha256"]["signatures.npz"]:
        raise RuntimeError("Raw signature hash mismatch")
    with np.load(run / "signatures.npz") as data:
        raw = data["raw"]
    records = load_records()
    matrices = []
    for bi in range(2):
        row = []
        for split in ("discovery", "heldout"):
            a = [i for i, r in enumerate(records) if r["split"] == split and r["family"] == "ABBA"]
            b = [i for i, r in enumerate(records) if r["split"] == split and r["family"] == "BABA"]
            row.append((raw[bi, a].mean(0) - raw[bi, b].mean(0)).reshape(6, 12))
        matrices.append(row)
    limit = max(float(np.abs(m).max()) for row in matrices for m in row)
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 6.6), layout="constrained", sharex=True, sharey=True
    )
    fig.suptitle("Different prompt orders: downstream head-ablation effects", fontsize=15)
    for bi, label in enumerate(("Zero replacement", "S2-corrupt replacement")):
        for si, split in enumerate(("Discovery", "Held-out names and templates")):
            ax = axes[bi, si]
            im = ax.imshow(matrices[bi][si], vmin=-limit, vmax=limit, cmap="RdBu_r", aspect="auto")
            ax.set_title(f"{label} · {split}", fontsize=10)
            ax.set_yticks(range(6), [str(i) for i in range(6, 12)])
            ax.set_xticks(range(12))
            ax.set_ylabel("Layer (zero-indexed)")
            if bi == 1:
                ax.set_xlabel("Head (zero-indexed)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Mean causal effect: ABBA − BABA (logits)")
    fig.supxlabel("72 fixed END heads · 64 prompts · No routing interventions", fontsize=10)
    dest = run / "analysis/raw-effect-differences.png"
    fig.savefig(dest, dpi=180, facecolor="white")
    plt.close(fig)
    print(dest)


if __name__ == "__main__":
    main()
