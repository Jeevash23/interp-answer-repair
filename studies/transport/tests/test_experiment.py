import torch

from circuit_transport.experiment import (
    HEAD_GROUPS,
    build_patch_hooks,
    early_local_projection,
    result_hook_name,
)


def _records(batch: int = 2) -> list[dict]:
    return [
        {
            "positions": {"io": 1, "subject_first": 3, "subject_second": 9, "end": 13},
            "io_token_id": 10,
            "subject_token_id": 11,
        }
        for _ in range(batch)
    ]


def _cache(value: float, batch: int = 2) -> dict[str, torch.Tensor]:
    return {result_hook_name(layer): torch.full((batch, 14, 12, 4), value) for layer in range(12)}


def test_correct_patch_adds_source_difference_at_target_site() -> None:
    clean = _cache(2.0)
    corrupt = _cache(0.5)
    hooks = build_patch_hooks(
        clean,
        corrupt,
        _records(),
        stage="name_mover",
        control="correct",
        random_seed=1,
    )
    values = _cache(0.0)
    for hook_name, hook in hooks:
        values[hook_name] = hook(values[hook_name], None)
    for layer, head in HEAD_GROUPS["name_mover"]:
        assert torch.all(values[result_hook_name(layer)][:, 13, head, :] == 1.5)


def test_off_target_patch_uses_previous_position() -> None:
    hooks = build_patch_hooks(
        _cache(1.0),
        _cache(0.0),
        _records(),
        stage="s_inhibition",
        control="off_target",
        random_seed=1,
    )
    values = _cache(0.0)
    for hook_name, hook in hooks:
        values[hook_name] = hook(values[hook_name], None)
    for layer, head in HEAD_GROUPS["s_inhibition"]:
        assert torch.all(values[result_hook_name(layer)][:, 12, head, :] == 1.0)
        assert torch.all(values[result_hook_name(layer)][:, 13, head, :] == 0.0)


def test_early_local_projection_recovers_exact_clean_direction() -> None:
    corrupt = _cache(0.0)
    clean = _cache(1.0)
    projection = early_local_projection(clean, corrupt, clean, _records())
    assert torch.allclose(projection, torch.ones(2))
