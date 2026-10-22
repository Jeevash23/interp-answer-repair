# Wider-source implementation (completed and verified)

All four conditions and fresh-engine singleton replay completed on 4 October
2026 by 21:30:51 UTC. The independent audit passed at 21:32:10 UTC. The full
four-contrast scientific criterion was not met because the matched fit-mean
interval spans zero. See the [results](../RESULTS.md),
[study overview](../README.md) and [reproduction instructions](../REPRODUCIBILITY.md).

`operators008.py` defines source operators with an explicit **seven-layer axis**
`[6,7,8,9,10,11,12]`. It never relabels layer sums as attention heads and never
changes the original engine's twelve-head source list.

`run008.py` supplies resumable inference using the validated END-only prefix
engine. It does not create a protocol freeze. Before any model stage, it requires
an externally created Bridge 008 `data/input-freeze.json`, complete Bridge 007
final inference/analysis and independent verification, successful independent/final clean qualification, and
the existing successful full-boundary donor replay. Bridge 007 primary scientific
predictions need not succeed. The protocol and analysis describe this as an
exploratory follow-up on reused configurations.

## Invocation and preservation

From the AI-01 project root, this performs **synthetic checks only** and writes
no retained report:

```
.venv/bin/python studies/bridge-008/code/run008.py self-test
```

The runner exposes `calibration`, `prepare`, `inference`,
`computational_replay` and the ordered `pipeline` stage. Each model stage
accepts `--freeze /absolute/path/to/input-freeze.json`; the default is Bridge
008 `data/input-freeze.json`. The input freeze must already exist. Only one
model process should run at once in any new execution copy. This retained run
is complete; do not invoke its model stages again.

An interrupted, unfinished stage can recover verified chunks after its prior
process stops. Once complete, do not rerun the pipeline in the retained tree:
it can rewrite completion timestamps bound by later provenance. Use fresh
output paths for saved-data analysis/verification, or a separate preserved
project copy and new provenance for full model re-execution. No clean-rerun
recipe has been validated. Preserve Bridge 007's canonical audit, which this
study freezes byte for byte.

## Saved interfaces

- `results/natural/{clean,corrupt}.npz`: `layer_sums` of shape `(n,7,1024)`,
  `behavior` of shape `(n,6)`, and `layer_ids_per_row` of shape `(n,7)`.
- `results/deltas/{wording,self}.npz`: `layer_deltas` (the **matched** float32
  injection), `broad_layer_deltas`, and `narrow_layer_deltas`, each `(n,7,1024)`;
  `narrow_norms`, `broad_norms`, `alpha`, `norm_error`, and `matched_valid`, each
  `(n,7)`; `layer_ids` of length seven. Norms and coefficients are float64.
- `results/deltas/matching-summary.json`: degeneracy counts, exact affected
  row IDs/layers, norm tolerances, and immutable delta-artifact hashes.
- `results/test/{broad_wording,broad_self,matched_wording,matched_self}.npz`:
  `raw` of shape `(2,n,176,2)` and `behavior` of shape `(n,6)`, with the same
  reference/score conventions as Bridge 007.
- Every natural/signature aggregate has a `*-record.json` plus atomic chunks.
  Records bind exact rows, freeze, code, layer identity, bank, injected tensor,
  numerical settings and packages. Chunks embed their input key and row indices
  for safe recovery if interruption happens before the record update.
- If any norm match is undefined, the corresponding complete matched condition
  is not measured. `results/test/matched_*-non-evaluable.json` records why, and
  that condition's aggregate NPZ is absent. No example is removed. The other
  conditions proceed, and the analysis must report unavailable interactions.

## Numerical semantics and computational checks

Natural broad outputs contract all sixteen source-head outputs with their output
matrices at S2, separately in each of the original seven layers. Only layer sums
are retained; the full final bank of 112 projected head outputs is never stored.
Broad clean-minus-corrupted subtraction is float32. Narrow sums reproduce the
original float32 per-head subtraction followed by the original PyTorch
within-layer sum. Word/self donor indexing follows the unchanged matched rows.

For matching, cast the actual narrow/broad layer injections to float64, compute
the norm ratio, multiply the broad vector and cast the final injection to
float32. Tolerance is `1e-6 + 1e-5 * narrow_norm`. Broad norm at most `1e-12` with
nonzero narrow norm is undefined; both-zero and zero-narrow cases inject zero.
Undefined tensors contain NaN and cannot pass the injection guard.

Real calibration passed at 20:11:06 UTC on eight fixed calibration
configurations covering both wordings, mention orders and roles. It compared
the efficient broad sum with explicit per-head projection and all-112-head
delta summation, original versus layer-summed narrow hooks, explicit versus
efficient broad-addition signatures/behavior, and full-sequence versus prefix
signatures for natural and matched broad wording edits. All checks were below
the inherited `0.0005` tolerance. The retained records are
`results/calibration.json`, technical source fixtures in
`results/calibration/source-validation.npz`, and paired validation arrays.

After inference, a fresh model instance remeasured a fixed eight-row subset
(the first two manifest rows in each wording/order cell) for all four new
conditions, using singleton rather than grouped head ablations. Behavior
matched exactly; maximum signature error was 0.0000143051, below 0.0005.
`results/computational-replay.json` and
`results/verification/*-singleton.npz` retain those checks. No additional
full-bank inference was performed; this is computational verification,
not an independent sample.

This runner does not introduce layer-13 residual addition or duplicate the
all-token donor-state replacement as a substantive experimental arm.
