"""Finite-design and intervention checks independent of model outcomes."""

import numpy as np
import torch
from assay import CausalEngine, delta_for
from design import specs
from run import make_rows


def test_disjoint_triples_and_complete_factorial():
    rows = make_rows()
    names_by_split = {}
    for split, n in [("screen", 128), ("discovery", 128), ("test", 384)]:
        rr = [r for r in rows if r["split"] == split]
        assert len(rr) == n
        names_by_split[split] = {r[k] for r in rr for k in ["io", "subject", "alternative"]}
        triples = {
            pair: {r[k] for r in rr if r["pair"] == pair for k in ["io", "subject", "alternative"]}
            for pair in {r["pair"] for r in rr}
        }
        assert all(len(t) == 3 for t in triples.values())
        assert sum(map(len, triples.values())) == len(names_by_split[split])
    assert sum(map(len, names_by_split.values())) == len(set.union(*names_by_split.values()))


def test_donors_exchange_position_and_identity_separately():
    rows = make_rows()
    index = {r["id"]: r for r in rows}
    for r in rows:
        a = index[r["donors"]["role"]]
        b = index[r["donors"]["order"]]
        both = index[r["donors"]["order_role"]]
        assert a["io"] == r["subject"] and a["io_position"] == r["io_position"]
        assert b["io"] == r["io"] and b["io_position"] != r["io_position"]
        assert both["io"] == r["subject"] and both["io_position"] != r["io_position"]
        assert both["token_ids"][2:5] == r["token_ids"][2:5]
        assert sum(a != b for a, b in zip(r["token_ids"], r["corrupt_tokens"], strict=True)) == 1
    assert len(specs()) == 13


def test_clamp_scope_for_query_key_value_and_z():
    engine = CausalEngine.__new__(CausalEngine)
    engine.layers = [2]
    cache = {f"blocks.2.attn.hook_{k}": torch.randn(2, 5, 3, 4) for k in ["q", "k", "v", "z"]}
    for k in ["q", "k", "v", "z"]:
        x = torch.randn(2, 5, 3, 4)
        _, hook = engine.clamp(cache, [[2, 1]], k)[0]
        y = hook(x, None)
        positions = [-1] if k in ["q", "z"] else list(range(5))
        torch.testing.assert_close(
            y[:, positions, 1], cache[f"blocks.2.attn.hook_{k}"][:, positions, 1]
        )
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, positions, 1] = False
        torch.testing.assert_close(y[mask], x[mask])


def test_joint_clamp_uses_reference_bank_per_layer_head():
    engine = CausalEngine.__new__(CausalEngine)
    engine.layers = [2, 3]
    cache = {f"blocks.{l}.attn.hook_z": torch.randn(2, 5, 3, 4) for l in [2, 3]}
    bank = torch.randn(2, 3, 4)
    hooks = engine.clamp(cache, [[2, 1], [3, 2]], replacement=bank)
    for i, (name, hook) in enumerate(hooks):
        original = cache[name]
        actual = hook(original, None)
        h = i + 1
        torch.testing.assert_close(actual[:, -1, h], bank[i, h].expand(2, -1))
        torch.testing.assert_close(actual[:, :-1], original[:, :-1])


def test_role_donor_delta_and_half_dose():
    rows = [{"id": "a", "donors": {"role": "b"}}, {"id": "b", "donors": {"role": "a"}}]
    natural = {
        "clean_head_results": np.arange(16, dtype=np.float32).reshape(2, 2, 4),
        "corrupt_head_results": np.ones((2, 2, 4), np.float32),
    }
    delta, donors = delta_for({"kind": "head", "donor": "role", "dose": 0.5}, rows, natural)
    np.testing.assert_array_equal(donors, [1, 0])
    np.testing.assert_allclose(delta, (natural["clean_head_results"][[1, 0]] - 1) * 0.5)
