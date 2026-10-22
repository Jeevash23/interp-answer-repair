import numpy as np
import torch

from transport_bridge.common import ROOT, read
from transport_bridge.engine import random_matched, state_hook
from transport_bridge.statistics import block_interval, geometry
from transport_bridge.study import conditions, prepare_delta


def test_complete_disjoint_factorial_and_donors():
    rows = read(ROOT / "data/recipients.json")
    library = read(ROOT / "data/prompts.json")
    index = {r["id"]: r for r in rows}
    assert len(index) == 512
    banks = [
        {r["io"] for r in rows if r["split"] == s} for s in ["discovery", "development", "test"]
    ]
    assert all(not banks[i] & banks[j] for i in range(3) for j in range(i))
    for r in rows:
        clean, corrupt = library[r["clean"]], library[r["corrupt"]]
        assert [
            i
            for i, (a, b) in enumerate(zip(clean["token_ids"], corrupt["token_ids"], strict=True))
            if a != b
        ] == [r["s2"]]
        for f, other in [("wording", "order"), ("order", "wording")]:
            d = index[r["donors"][f]]
            assert d[f] != r[f] and d[other] == r[other]
            assert d["io"] == r["io"] and d["subject"] == r["subject"] and d["split"] == r["split"]
            assert library[r["donors"][f + "_changed"]]["io_id"] == r["alt_id"]


def test_geometry_recipient_donor_and_orthogonal_damage_both_directions():
    rows = [{"wording": 0, "order": 0}, {"wording": 1, "order": 0}]
    fit = {"scale": [[1, 1]], "centers": [[[0, 0], [0, 1], [2, 0], [2, 1]]]}
    clean = np.array([[[[0, 0], [0, 0]], [[2, 0], [0, 0]]]], dtype=float)
    donor = clean[:, ::-1].copy()
    m = geometry(donor, clean, donor, rows, fit, "wording")
    np.testing.assert_allclose(m["score"], 1)
    np.testing.assert_allclose(m["orthogonal"], 0)
    np.testing.assert_allclose(m["donor_distance"], 0)
    damaged = donor.copy()
    damaged[:, :, 1, 0] = 4
    m = geometry(damaged, clean, donor, rows, fit, "wording")
    np.testing.assert_allclose(m["score"], 1)
    np.testing.assert_allclose(m["orthogonal"], 2)


def test_crossed_constant_and_paired_cancellation():
    rows = [{"template": t, "pair": p} for t in range(8) for p in range(4) for _ in range(4)]
    result = block_interval(np.full(128, 0.25), rows, draws=1000)
    assert result["ci95"] == [0.25, 0.25] and result["simultaneous_ci"] == [0.25, 0.25]


def test_random_control_batch_independent_and_norm_matched():
    d = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) + 1
    full = random_matched(d, ["a", "b", "c"], "head", 2)
    parts = torch.cat(
        [random_matched(d[:1], ["a"], "head", 2), random_matched(d[1:], ["b", "c"], "head", 2)]
    )
    torch.testing.assert_close(full, parts, rtol=0, atol=0)
    torch.testing.assert_close(full.norm(dim=-1), d.norm(dim=-1))


def test_state_edit_only_target_site_and_full_recovers_donor():
    x = torch.randn(3, 5, 4)
    donor = torch.randn(3, 4)
    pos = torch.tensor([1, 2, 3])
    _, hook = state_hook(2, pos, donor - x[torch.arange(3), pos])
    y = hook(x, None)
    torch.testing.assert_close(y[torch.arange(3), pos], donor)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[torch.arange(3), pos] = False
    torch.testing.assert_close(x[mask], y[mask])


def test_interventions_all_upstream_and_inventory():
    cfg = read(ROOT / "configs/study.json")
    for model in cfg["models"].values():
        assert max(h[0] for h in model["source_heads"]) < min(model["layers"])
        assert model["site"] == min(model["layers"])
    assert len(conditions()) == len({c["name"] for c in conditions()}) == 27


def test_equal_norm_full_and_rank_projection():
    rows = [{"id": "a", "donors": {"wording": "b"}}, {"id": "b", "donors": {"wording": "a"}}]
    a = {
        "clean_state": np.array([[1, 2], [3, 4]], np.float32),
        "corrupt_state": np.zeros((2, 2), np.float32),
    }
    s = {"predictors": {"wording": {"direction": [1.0, 0.0]}}}
    rank, _ = prepare_delta({"kind": "rank1", "factor": "wording", "base": "corrupt"}, rows, a, s)
    matched, _ = prepare_delta(
        {"kind": "matched", "factor": "wording", "base": "corrupt"}, rows, a, s
    )
    torch.testing.assert_close(rank.norm(dim=-1), matched.norm(dim=-1))
