"""Additional integration checks on previously observed pilot prompts only."""

import torch

from causal_circuit_routing.interventions import head_hook
from causal_circuit_routing.io import ROOT, load_records, now, read_json, write_json
from causal_circuit_routing.study2_engine import Engine, state_hook

cfg = read_json(ROOT / "configs/study-002.json")
cfg["threads"] = 1
with torch.inference_mode():
    engine = Engine(cfg)
    old = load_records()[:4]
    rows = [{**r, "token_ids": r["clean_token_ids"]} for r in old]
    corrupt = [{**r, "token_ids": r["corrupt_token_ids"]} for r in old]
    _, f, _, base = engine.forward_cache(rows)
    _, _, corrupt_z, _ = engine.forward_cache(corrupt)
    pos = torch.tensor([r["positions"]["subject_second"] for r in rows])
    donor = f[:, 5].roll(1, 0)
    delta = donor - f[:, 5]
    full = state_hook(6, pos, delta)
    _, fp, _, _ = engine.forward_cache(rows, [full])
    full_error = float((fp[:, 5] - donor).abs().max())
    checks = []
    for layer, position, kind in [(6, pos - 1, "residual"), (3, pos, "residual"), (0, pos, "mlp")]:
        calls = {"count": 0}
        name, fn = state_hook(layer, position, delta, kind)

        def wrapped(value, hook, *, calls=calls, fn=fn):
            calls["count"] += 1
            return fn(value, hook)

        _, _, _, cache = engine.forward_cache(rows, [(name, wrapped)])
        assert calls["count"] == 1
        head, head_calls = head_hook(
            9, 6, corrupt_z[:, 3, 6], torch.full((4,), len(rows[0]["token_ids"]) - 1)
        )
        fast = engine.suffix(cache["blocks.9.hook_resid_pre"].clone(), 9, [head])
        slow = engine.model.run_with_hooks(
            torch.tensor([r["token_ids"] for r in rows]), fwd_hooks=[(name, wrapped), head]
        )[:, -1:]
        err = float((fast - slow).abs().max())
        assert err <= cfg["numerical_tolerance"]
        checks.append(
            {
                "site": name,
                "control_position": position.tolist(),
                "hook_calls_per_forward": 1,
                "nested_corrupt_baseline_max_error": err,
            }
        )
    assert full_error <= cfg["numerical_tolerance"]
    write_json(
        ROOT / "results/study-002/hook-verification.json",
        {
            "verified_utc": now(),
            "technical_examples": "pilot001 first four already observed prompts",
            "full_vector_recovers_donor_site_max_error": full_error,
            "controls": checks,
        },
    )
    print("Technical hook checks passed:", full_error, checks)
