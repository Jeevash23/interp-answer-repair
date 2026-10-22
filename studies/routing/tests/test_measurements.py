import numpy as np
from activation_transfer.paths import recorded_path

import pytest
import torch

from causal_circuit_routing.interventions import head_hook
from causal_circuit_routing.metrics import fit_signature, hierarchical_interval, movement


def test_switch_score_identity_full_transfer_reverse_and_off_axis():
    fit = {"axis": np.array([2.0, 0.0]), "scale": np.ones(2)}
    assert movement([0, 0], [0, 0], 0, fit) == (0.0, 0.0)
    assert movement([0, 0], [2, 0], 0, fit) == (1.0, 0.0)
    assert movement([2, 0], [0, 0], 1, fit) == (1.0, 0.0)
    assert movement([0, 0], [2, 4], 0, fit) == (1.0, 2.0)
    assert movement([0, 0], [-2, 0], 0, fit) == (-1.0, 0.0)


def test_degenerate_axis_rejected_and_scales_use_within_family_variance():
    with pytest.raises(ValueError):
        fit_signature(np.ones((4, 2)), np.array([0, 0, 1, 1]))
    fit = fit_signature(np.array([[0.0, 0], [0, 0], [10, 0], [10, 0]]), [0, 0, 1, 1])
    np.testing.assert_array_equal(fit["scale"], [0.1, 0.1])


def test_hook_only_changes_selected_head_and_position_without_mutating_cache():
    original = torch.randn(2, 5, 12, 64)
    stored = original.clone()
    (name, fn), calls = head_hook(7, 3, torch.ones(2, 64), torch.tensor([3, 4]))
    result = fn(original, None)
    assert name == "blocks.7.attn.hook_z" and calls["count"] == 1
    torch.testing.assert_close(original, stored)
    for row, pos in enumerate([3, 4]):
        torch.testing.assert_close(result[row, pos, 3], torch.ones(64))
        result[row, pos, 3] = stored[row, pos, 3]
    torch.testing.assert_close(result, stored)


def test_bootstrap_retains_constant_block_statistic():
    assert hierarchical_interval(np.full((2, 4), 0.75), 100, 123) == [0.75, 0.75]


def test_counterbalanced_manifest_has_disjoint_splits_and_answer_family_control():
    from transformers import AutoTokenizer

    from causal_circuit_routing.dataset import make_records
    from causal_circuit_routing.io import ROOT, read_json

    config = read_json(ROOT / "configs/pilot-001.json")
    tokenizer = AutoTokenizer.from_pretrained(recorded_path(ROOT / config["model_path"]), local_files_only=True)
    records = make_records(tokenizer)
    assert len(records) == 64
    vocab = {
        s: {r[k] for r in records if r["split"] == s for k in ("io", "subject", "corrupt_name")}
        for s in ("discovery", "heldout")
    }
    assert not vocab["discovery"] & vocab["heldout"]
    for pair_id in {r["pair_id"] for r in records}:
        a, b = [r for r in records if r["pair_id"] == pair_id]
        assert {a["family"], b["family"]} == {"ABBA", "BABA"}
        assert a["io_token_id"] == b["io_token_id"]
        assert a["subject_token_id"] == b["subject_token_id"]
        assert len(a["clean_token_ids"]) == len(b["clean_token_ids"])
