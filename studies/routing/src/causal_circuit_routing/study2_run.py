"""Shared collection and source verification for study 002."""

from __future__ import annotations

from activation_transfer.paths import frozen_file_matches
import numpy as np
import torch

from causal_circuit_routing.io import ROOT, digest, read_json
from causal_circuit_routing.study2_engine import batches


def verify_inputs(selection=False):
    frozen = read_json(ROOT / "data/study-002-freeze.json")
    for p, sha in frozen["files"].items():
        if not frozen_file_matches(ROOT / p, sha):
            raise RuntimeError(f"Frozen study input changed: {p}")
    for p, sha in frozen["model"]["files"].items():
        if digest(ROOT / frozen["model"]["path"] / p) != sha:
            raise RuntimeError(f"Model file changed: {p}")
    if selection:
        frozen_selection = read_json(ROOT / "data/study-002-selection-freeze.json")
        for p, sha in frozen_selection["files"].items():
            if not frozen_file_matches(ROOT / p, sha):
                raise RuntimeError(f"Selection freeze changed: {p}")


def measure(engine, rows, library, config, label):
    n = len(rows)
    arrays = {
        "raw": np.empty((2, n, 72), np.float32),
        "probability_raw": np.empty((2, n, 72), np.float32),
        "features": np.empty((n, len(engine.sites), 768), np.float32),
        "corrupt_z": np.empty((n, 6, 12, 64), np.float32),
        "clean_z": np.empty((n, 6, 12, 64), np.float32),
    }
    groups = batches(rows, config["batch_size"])
    for batch_number, ix in enumerate(groups):
        clean = [rows[i] for i in ix]
        corrupt = [{**r, "token_ids": library[r["corrupt_id"]]["token_ids"]} for r in clean]
        corrupt_behavior, _, corrupt_z, _ = engine.forward_cache(corrupt)
        raw, prob, behavior, features, clean_z = engine.signatures(clean, corrupt_z)
        arrays["raw"][:, ix] = raw
        arrays["probability_raw"][:, ix] = prob
        arrays["features"][ix] = features.numpy()
        arrays["corrupt_z"][ix] = corrupt_z.numpy()
        arrays["clean_z"][ix] = clean_z.numpy()
        for prefix, data in (("clean", behavior), ("corrupt", corrupt_behavior)):
            for key, value in data.items():
                name = f"{prefix}_{key}"
                if name not in arrays:
                    arrays[name] = np.empty(n, np.float64)
                arrays[name][ix] = value.numpy()
        print(f"{label}: batch {batch_number + 1}/{len(groups)}", flush=True)
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError("Non-finite measurements")
    return arrays


def project(delta, direction, mode):
    if mode == "full":
        return delta
    direction = torch.as_tensor(direction, dtype=delta.dtype)
    return (delta @ direction)[:, None] * direction[None, :]


def matched_random(delta, seed):
    rng = torch.Generator().manual_seed(seed)
    noise = torch.randn(delta.shape, generator=rng)
    return (
        noise * delta.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    )
