"""Synthetic known-answer checks of the frozen confirmatory analysis."""

import importlib.util

import numpy as np

from causal_circuit_routing.io import ROOT

spec = importlib.util.spec_from_file_location("analysis_study2", ROOT / "scripts/analyze_study2.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_vectorized_scores_handle_both_directions_and_orthogonal_damage():
    fit = {"scale": [1.0, 2.0], "axis": [2.0, 0.0], "centroids": [[0.0, 0.0], [2.0, 0.0]]}
    before = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    after = np.array([[2.0, 0.0], [0.0, 0.0], [2.0, 8.0]])
    score, orth, closer = analysis.signature_outcomes(after, before, fit, np.array([0, 1, 0]))
    np.testing.assert_allclose(score, [1, 1, 1])
    np.testing.assert_allclose(orth, [0, 0, 2])
    np.testing.assert_allclose(closer, [2, 2, -2])


def test_constant_paired_contrast_bootstrap_and_cluster_counts():
    rows = [
        {"template_id": f"t{t}", "name_pair_id": f"n{n}"}
        for t in range(8)
        for n in range(4)
        for _ in range(4)
    ]
    result = analysis.block_stats(
        np.full(len(rows), 0.25), rows, np.ones(len(rows), bool), 17, draws=1000
    )
    assert result["mean"] == 0.25 and result["ci95"] == [0.25, 0.25]
    assert result["simultaneous_ci"] == [0.25, 0.25]
    assert result["templates"] == 8 and result["name_pairs"] == 4


def test_simultaneous_interval_contains_pointwise_interval():
    rows = [
        {"template_id": f"t{t}", "name_pair_id": f"n{n}"}
        for t in range(8)
        for n in range(4)
        for _ in range(4)
    ]
    values = np.array([int(r["template_id"][1:]) + int(r["name_pair_id"][1:]) for r in rows])
    result = analysis.block_stats(values, rows, np.ones(len(rows), bool), 17, draws=1000)
    assert result["simultaneous_ci"][0] <= result["ci95"][0]
    assert result["simultaneous_ci"][1] >= result["ci95"][1]
