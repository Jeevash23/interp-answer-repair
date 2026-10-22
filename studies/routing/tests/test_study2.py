import numpy as np
from activation_transfer.paths import recorded_path

import torch

from causal_circuit_routing.io import ROOT, load_records, read_json
from causal_circuit_routing.study2_data import build
from causal_circuit_routing.study2_engine import state_hook
from causal_circuit_routing.study2_run import matched_random, project


def test_study2_splits_and_four_donors():
    from transformers import AutoTokenizer

    cfg = read_json(ROOT / "configs/study-002.json")
    tok = AutoTokenizer.from_pretrained(recorded_path(ROOT / cfg["model_path"]), local_files_only=True)
    rows, library = build(tok)
    old_names = {r[k] for r in load_records() for k in ("io", "subject", "corrupt_name")}
    assert not old_names & {r["io"] for r in rows}
    assert [sum(r["split"] == s for r in rows) for s in ("discovery", "development", "test")] == [
        64,
        64,
        128,
    ]
    for r in rows:
        for key, pid in r["donors"].items():
            wording, family, answer = key.split("_")
            d = library[pid]
            assert d["split"] == r["split"] and d["subject"] == r["subject"]
            assert (d["template_id"] == r["template_id"]) == (wording == "same")
            assert (d["family"] == r["family"]) == (family == "same")
            assert (d["io"] == r["io"]) == (answer == "same")


def test_projection_and_control_norm_are_exact():
    delta = torch.tensor([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0]])
    q = [1.0, 0.0, 0.0]
    torch.testing.assert_close(
        project(delta, q, "rank1"), torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(matched_random(delta, 13).norm(dim=-1), delta.norm(dim=-1))
    assert torch.equal(matched_random(delta, 13), matched_random(delta, 13))


def test_state_edit_changes_only_declared_position():
    x = torch.zeros(2, 4, 768)
    d = torch.ones(2, 768)
    name, fn = state_hook(4, torch.tensor([1, 2]), d)
    y = fn(x, None)
    assert name == "blocks.4.hook_resid_pre"
    assert x.sum().item() == 0
    assert y.sum().item() == 1536
    np.testing.assert_array_equal(y[0, 1].numpy(), np.ones(768))
