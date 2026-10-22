import torch

from circuit_transport.experiment_005 import (
    addition_hooks,
    clamp_hooks,
    donor_map,
    random_directions,
    sampled_heads,
    symmetric_scale,
)


def test_random_head_draws_preserve_layers_and_exclusions():
    reference = ((1, 0), (1, 1), (2, 2))
    draws = [sampled_heads(reference, reference, n_heads=16, seed=i) for i in range(5)]
    assert len(set(draws)) > 1
    for draw in draws:
        assert [l for l, _ in draw] == [1, 1, 2]
        assert not set(draw) & set(reference)
        assert len(set(draw)) == 3


def test_donor_mapping_is_independent_of_order_and_preserves_pattern():
    rows = [
        {
            "example_id": f"t-{i:02d}-{p.lower()}",
            "template_id": "t",
            "pattern": p,
            "io_name": f"a{i}",
            "subject_name": f"b{i}",
            "corrupt_name": f"c{i}",
        }
        for i in range(10)
        for p in ("ABBA", "BABA")
    ]
    mapping = donor_map(rows)
    assert mapping == donor_map(list(reversed(rows)))
    for k, v in mapping.items():
        assert k != v and k.rsplit("-", 1)[1] == v.rsplit("-", 1)[1]


def test_random_direction_is_batch_and_order_invariant_with_matching_norms():
    delta = torch.arange(20 * 16, dtype=torch.float32).reshape(20, 16)
    ids = [str(i) for i in range(20)]
    kwargs = {"model": "test", "site": "s2", "head": (2, 3), "draw": 1}
    whole = random_directions(delta, ids, **kwargs)
    batches = torch.cat(
        [random_directions(delta[i : i + 4], ids[i : i + 4], **kwargs) for i in range(0, 20, 4)]
    )
    assert torch.equal(whole, batches)
    assert torch.equal(whole.flip(0), random_directions(delta.flip(0), ids[::-1], **kwargs))
    assert torch.allclose(whole.norm(dim=1), delta.norm(dim=1))


def test_symmetric_norm_matching_and_zero_vectors():
    left, right = torch.ones(3, 2, 4), torch.full((3, 2, 4), 3.0)
    left[0] = 0
    a, b, _, _, shared = symmetric_scale(left, right)
    assert torch.allclose((left * a[:, None, None]).flatten(1).norm(dim=1), shared)
    assert torch.allclose((right * b[:, None, None]).flatten(1).norm(dim=1), shared)
    assert shared[0] == 0


def test_layer_additions_equal_head_result_sum_and_zero_is_noop():
    result = torch.randn(2, 5, 4, 8)
    d = {(0, 0): torch.randn(2, 8), (0, 2): torch.randn(2, 8)}
    pos = torch.tensor([2, 3])
    expected = result.clone()
    for (_, h), delta in d.items():
        expected[torch.arange(2), pos, h] += delta
    _, add = addition_hooks(d, pos)[0]
    assert torch.allclose(add(result.sum(2), None), expected.sum(2), atol=1e-6)
    _, noop = addition_hooks({(0, 0): torch.zeros(2, 8)}, pos)[0]
    assert torch.equal(noop(result.sum(2), None), result.sum(2))


def test_node_clamp_changes_only_selected_end_values():
    baseline = torch.randn(2, 5, 4, 8)
    pos = torch.tensor([2, 3])
    z = baseline[torch.arange(2), pos].unsqueeze(1)
    _, clamp = clamp_hooks(z, ((0, 1),), pos)[0]
    assert torch.equal(clamp(baseline, None), baseline)
    perturbed = baseline.clone()
    perturbed[torch.arange(2), pos, 1] += 2
    assert torch.equal(clamp(perturbed, None), baseline)
