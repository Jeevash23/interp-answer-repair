import torch

from circuit_transport.experiment_002 import result_hook_name
from circuit_transport.experiment_004 import build_group_patch_hooks, condition_positions


def _records() -> list[dict]:
    return [
        {
            "positions": {
                "canonical": {"subject_second": 2, "end": 4},
                "controlled": {"subject_second": 3, "end": 5},
            }
        }
        for _ in range(2)
    ]


def _cache(value: float) -> dict[str, torch.Tensor]:
    return {result_hook_name(0): torch.full((2, 6, 4, 3), value)}


def test_condition_positions_are_resolved_separately() -> None:
    assert condition_positions(
        _records(), "canonical", "end", device=torch.device("cpu")
    ).tolist() == [4, 4]
    assert condition_positions(
        _records(), "controlled", "end", device=torch.device("cpu")
    ).tolist() == [5, 5]


def test_cross_condition_patch_uses_source_and_target_positions() -> None:
    clean = _cache(0.0)
    corrupt = _cache(0.0)
    clean[result_hook_name(0)][:, 4, 0, :] = 2.0
    hooks, active = build_group_patch_hooks(
        clean,
        corrupt,
        _records(),
        source_condition="canonical",
        target_condition="controlled",
        heads=((0, 0),),
        site="end",
        control="correct",
        random_heads=None,
        random_seed=1,
    )
    value = torch.zeros(2, 6, 4, 3)
    patched = hooks[0][1](value, None)
    assert active == ((0, 0),)
    assert torch.all(patched[:, 5, 0, :] == 2)
    assert torch.count_nonzero(patched[:, 4, 0, :]) == 0
