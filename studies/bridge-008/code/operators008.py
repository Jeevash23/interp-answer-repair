"""Seven-layer S2 source operators; no scientific data are loaded on import."""
from __future__ import annotations

import numpy as np
import torch

LAYERS = tuple(range(6, 13))
EPSILON = 1e-12
NORM_ATOL = 1e-6
NORM_RTOL = 1e-5


def layer_sums_from_cache(engine, rows, cache):
    """Project and sum all 16 heads per layer; retain only n x 7 x width."""
    assert engine.heads == 16
    positions = torch.tensor([r["s2"] for r in rows], device=engine.device)
    indices = torch.arange(len(rows), device=engine.device)
    sums = []
    for layer in LAYERS:
        z = cache[f"blocks.{layer}.attn.hook_z"][indices, positions]
        sums.append(torch.einsum("nhd,hdm->nm", z, engine.weights[layer]))
    return torch.stack(sums, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)


def narrow_layer_deltas(clean_heads, corrupt_heads, source_heads):
    """Reproduce the original float32 subtraction and within-layer torch sum."""
    clean, corrupt = np.asarray(clean_heads), np.asarray(corrupt_heads)
    assert clean.dtype == corrupt.dtype == np.float32 and clean.shape == corrupt.shape
    assert len(source_heads) == clean.shape[1] == 12
    assert set(int(h[0]) for h in source_heads) == set(LAYERS)
    head_delta = torch.from_numpy(clean - corrupt)
    sums = [head_delta[:, [i for i, (l, _) in enumerate(source_heads) if l == layer]].sum(1)
            for layer in LAYERS]
    return torch.stack(sums, dim=1).numpy()


def match_broad_to_narrow(broad, narrow):
    """Normalize in float64, inject float32; retain undefined matches as NaN."""
    broad, narrow = np.asarray(broad), np.asarray(narrow)
    assert broad.dtype == narrow.dtype == np.float32
    assert broad.shape == narrow.shape and broad.ndim == 3 and broad.shape[1] == len(LAYERS)
    assert np.isfinite(broad).all() and np.isfinite(narrow).all()
    b, n = broad.astype(np.float64), narrow.astype(np.float64)
    bn, nn = np.linalg.norm(b, axis=-1), np.linalg.norm(n, axis=-1)
    valid = (bn > EPSILON) | (nn <= EPSILON)
    alpha = np.full(bn.shape, np.nan)
    alpha[(bn <= EPSILON) & (nn <= EPSILON)] = 0.0
    ordinary = bn > EPSILON
    alpha[ordinary] = nn[ordinary] / bn[ordinary]
    # A numerically zero narrow norm has an explicitly zero matched injection.
    alpha[ordinary & (nn <= EPSILON)] = 0.0
    matched = (b * alpha[..., None]).astype(np.float32)
    mn = np.linalg.norm(matched.astype(np.float64), axis=-1)
    error = np.abs(mn - nn)
    assert np.all(error[valid] <= NORM_ATOL + NORM_RTOL * nn[valid])
    return dict(layer_deltas=matched, narrow_layer_deltas=narrow,
                broad_layer_deltas=broad, narrow_norms=nn, broad_norms=bn,
                alpha=alpha, matched_valid=valid, norm_error=error,
                layer_ids=np.asarray(LAYERS, dtype=np.int64))


def make_layer_hooks(engine, layer_deltas, rows):
    """Add a seven-layer tensor to live attention outputs, at S2 only."""
    delta = torch.as_tensor(layer_deltas, dtype=torch.float32, device=engine.device)
    assert tuple(delta.shape) == (len(rows), len(LAYERS), engine.spec["width"])
    assert torch.isfinite(delta).all(), "Undefined layer delta must never be injected"
    positions = torch.tensor([r["s2"] for r in rows], device=engine.device)
    hooks = []
    for index, layer in enumerate(LAYERS):
        injection = delta[:, index]
        def add(value, hook, *, injection=injection):
            out = value.clone()
            out[torch.arange(len(out), device=out.device), positions] += injection
            return out
        hooks.append((f"blocks.{layer}.hook_attn_out", add))
    return hooks


def matching_summary(arrays, rows):
    invalid = np.argwhere(~arrays["matched_valid"])
    valid_errors = arrays["norm_error"][arrays["matched_valid"]]
    return dict(n=len(rows), layer_ids=list(LAYERS),
                epsilon=EPSILON, norm_atol=NORM_ATOL, norm_rtol=NORM_RTOL,
                matched_evaluable=bool(arrays["matched_valid"].all()),
                invalid_layer_count=len(invalid),
                invalid_example_count=len(set(int(i) for i, _ in invalid)),
                invalid=[dict(id=rows[int(i)]["id"], row=int(i), layer=LAYERS[int(j)]) for i,j in invalid],
                maximum_valid_norm_error=float(valid_errors.max()) if len(valid_errors) else None,
                both_zero_count=int(((arrays["broad_norms"] <= EPSILON) &
                                     (arrays["narrow_norms"] <= EPSILON)).sum()),
                zero_narrow_nonzero_broad_count=int(((arrays["broad_norms"] > EPSILON) &
                                                   (arrays["narrow_norms"] <= EPSILON)).sum()),
                rule="Any undefined layer makes that complete matched condition non-evaluable; no rows omitted")


def self_test():
    from types import SimpleNamespace
    rng = np.random.default_rng(104008)
    broad = rng.normal(size=(3,7,5)).astype(np.float32)
    narrow = rng.normal(size=(3,7,5)).astype(np.float32)
    broad[0,0] = 0; narrow[0,0] = 0
    narrow[0,1] = 0
    broad[1,2] = 0
    out = match_broad_to_narrow(broad,narrow)
    assert not out["matched_valid"][1,2] and np.isnan(out["layer_deltas"][1,2]).all()
    assert np.array_equal(out["layer_deltas"][0,0],np.zeros(5))
    assert np.array_equal(out["layer_deltas"][0,1],np.zeros(5))
    valid = out["matched_valid"]
    np.testing.assert_allclose(np.linalg.norm(out["layer_deltas"].astype(float),axis=-1)[valid],
                               out["narrow_norms"][valid],atol=NORM_ATOL,rtol=NORM_RTOL)
    # Layer-sum magnitude, rather than sum of per-head magnitudes, is the target.
    source=[(l,0) for l in LAYERS]+[(l,1) for l in LAYERS[:5]]
    a=np.zeros((1,12,5),np.float32);a[0,0,:2]=[100,0];a[0,7,:2]=[-99,1]
    sums=narrow_layer_deltas(a,np.zeros_like(a),source)
    np.testing.assert_array_equal(sums[0,0,:2],[1,1])
    np.testing.assert_allclose(np.linalg.norm(sums[0,0]),np.sqrt(2),rtol=1e-6)
    engine=SimpleNamespace(device=torch.device("cpu"),spec={"width":5},heads=16,
                           weights=torch.tensor(rng.normal(size=(24,16,3,5)).astype(np.float32)))
    rows=[dict(id="a",s2=1),dict(id="b",s2=2)]
    values=torch.tensor(rng.normal(size=(2,4,5)).astype(np.float32))
    delta=np.ones((2,7,5),np.float32)
    for li,(name,fn) in enumerate(make_layer_hooks(engine,delta,rows)):
        assert name==f"blocks.{LAYERS[li]}.hook_attn_out"
        changed=fn(values,None)-values
        expected=torch.zeros_like(values);expected[0,1]=1;expected[1,2]=1
        torch.testing.assert_close(changed,expected,rtol=0,atol=2e-7)
    cache={f"blocks.{l}.attn.hook_z":torch.tensor(rng.normal(size=(2,4,16,3)).astype(np.float32)) for l in LAYERS}
    efficient=layer_sums_from_cache(engine,rows,cache)
    explicit=np.empty_like(efficient)
    for i,row in enumerate(rows):
        for j,l in enumerate(LAYERS):
            explicit[i,j]=torch.stack([cache[f"blocks.{l}.attn.hook_z"][i,row["s2"],h] @ engine.weights[l,h]
                                      for h in range(16)]).sum(0).numpy()
    np.testing.assert_allclose(efficient,explicit,atol=5e-6,rtol=1e-6)
    try:make_layer_hooks(engine,np.zeros((2,12,5),np.float32),rows)
    except AssertionError:pass
    else:raise AssertionError("A twelve-head tensor was accepted as seven layer sums")
    return dict(status="passed",source="synthetic only",checks=["sum_norm_not_sum_of_head_norms",
        "float64_matching_float32_injection","both_zero_and_undefined_cases",
        "seven_layer_identity_and_s2_only_additions","efficient_contraction_vs_explicit_heads",
        "reject_fake_head_shape"])
