"""Design integrity and semantic donor matching for the new studies."""

import functools
import operator

from broad003 import rows_for
from common003 import HERE, MODELS, config, donor_indices, read


def test_names_and_splits_are_exclusive():
    design = read(HERE / "data/design.json")
    banks = list(design["names"].values())
    assert len({n for bank in banks for n in bank}) == sum(map(len, banks)) == 60
    assert not set(functools.reduce(operator.iadd, banks, [])) & set(design["excluded_prior_names"])


def test_semantic_positions_and_donors():
    for model in MODELS:
        rows = rows_for(model, "test", [0, 6])
        assert len(rows) == 384
        for factor in ["self", "wording", "order", "role", "order_role"]:
            ids = donor_indices(rows, factor)
            assert sorted(ids.tolist()) == list(range(len(rows)))
            for r, i in zip(rows, ids, strict=True):
                d = rows[i]
                assert (r["family"] != d["family"]) == ("wording" in factor)
                assert (r["order"] != d["order"]) == ("order" in factor)
                assert (r["io"] != d["io"]) == ("role" in factor)
        for r in rows:
            assert r["token_ids"][r["io_position"]] == r["io_id"]
            assert r["token_ids"][r["subject_position"]] == r["subject_id"]
            assert r["token_ids"][r["s2"]] == r["subject_id"]
            assert r["corrupt_tokens"][r["s2"]] == r["alt_id"]
            assert (
                sum(a != b for a, b in zip(r["token_ids"], r["corrupt_tokens"], strict=True)) == 1
            )


def test_source_heads_precede_signature_heads():
    for model in MODELS:
        spec = config()["models"][model]
        assert max(l for l, h in spec["source_heads"]) < min(spec["layers"])
        assert len(set(map(tuple, spec["source_heads"]))) == len(spec["source_heads"])


def test_architectures_receive_identical_texts():
    a = rows_for(MODELS[0], "test", [0, 6])
    b = rows_for(MODELS[1], "test", [0, 6])
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["text"] for r in a] == [r["text"] for r in b]
    assert all(r["io"] == s["io"] for r, s in zip(a, b, strict=True))
