import numpy as np
import pytest
import torch

from circuit_transport.experiment_002 import (
    benjamini_hochberg,
    build_group_patch_hooks,
    choose_random_heads,
    localization_similarity,
    make_layer_sweep_hook,
    result_hook_name,
    select_top_heads,
    sign_flip_pvalue,
)


def _records(batch: int = 2) -> list[dict]:
    return [
        {
            "positions": {"subject_second": 2, "end": 4},
            "io_token_id": 10,
            "subject_token_id": 11,
        }
        for _ in range(batch)
    ]


def _cache(value: float, *, batch: int = 2) -> dict[str, torch.Tensor]:
    return {result_hook_name(layer): torch.full((batch, 5, 4, 3), value) for layer in range(3)}


def test_layer_sweep_patches_one_head_per_expanded_row() -> None:
    delta = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    positions = torch.tensor([1, 2])
    hook = make_layer_sweep_hook(delta, positions)
    value = torch.zeros(8, 5, 4, 3)
    patched = hook(value, None)
    for example in range(2):
        for head in range(4):
            row = example * 4 + head
            assert torch.equal(patched[row, positions[example], head], delta[example, head])
            assert torch.count_nonzero(patched[row]) == torch.count_nonzero(delta[example, head])


def test_selection_similarity_and_random_heads_are_deterministic() -> None:
    canonical = {(layer, head): float(layer * 4 + head) for layer in range(8) for head in range(4)}
    controlled = dict(canonical)
    assert select_top_heads(canonical, count=2) == ((7, 3), (7, 2))
    similarity = localization_similarity(canonical, controlled)
    assert similarity["spearman"] == 1.0
    assert similarity["top_8_jaccard"] == 1.0
    selected = ((0, 0), (1, 1))
    assert choose_random_heads(selected, excluded=selected, n_heads=4) == ((0, 1), (1, 2))


def test_group_patch_controls_change_expected_coordinates() -> None:
    heads = ((0, 0), (1, 1))
    clean = _cache(1.0)
    corrupt = _cache(0.0)
    hooks, active = build_group_patch_hooks(
        clean,
        corrupt,
        _records(),
        heads=heads,
        site="end",
        control="correct",
        random_heads=None,
        random_seed=1,
    )
    assert active == heads
    values = _cache(0.0)
    for name, hook in hooks:
        values[name] = hook(values[name], None)
    assert torch.all(values[result_hook_name(0)][:, 4, 0] == 1)
    assert torch.all(values[result_hook_name(1)][:, 4, 1] == 1)

    hooks, _ = build_group_patch_hooks(
        clean,
        corrupt,
        _records(),
        heads=heads,
        site="end",
        control="off_target",
        random_heads=None,
        random_seed=1,
    )
    values = _cache(0.0)
    for name, hook in hooks:
        values[name] = hook(values[name], None)
    assert torch.all(values[result_hook_name(0)][:, 3, 0] == 1)
    assert torch.all(values[result_hook_name(0)][:, 4, 0] == 0)


def test_randomization_statistics() -> None:
    pvalue = sign_flip_pvalue(np.ones(12), draws=2_000, seed=5)
    assert pvalue < 0.01
    adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] == 0.06
    assert adjusted["c"] == pytest.approx(0.20)
