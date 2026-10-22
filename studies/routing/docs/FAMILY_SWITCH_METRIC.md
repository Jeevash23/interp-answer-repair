# Frozen downstream family-switch metric

Plan version: 1. Coordinate selection is fixed before pilot inference and before any routing edit.
Numerical scales and centroids are fit from discovery data only and then stored and hashed.

## Intervention tuple

- Component: individual attention head, `hook_z` before the output projection.
- Coordinates: all `(layer, head)` with layer 6–11 and head 0–11; lexicographic order.
- Position: final input token END, with explicit BOS and no padding within a batch.
- Direction: clean-to-ablated necessity effect.
- Ablated set: one node at a time, downstream computation otherwise live.
- Replacement: zero (primary) and recipient-matched S2-corrupt activation (sensitivity), analyzed
  separately. Neither replacement is a universal notion of removing information.
- Endpoint: IO-minus-subject logit difference; probability difference is a secondary sensitivity.

Let `g(x; r)` be that endpoint with routing intervention r, where r=0 means no routing edit.
Let `A_i` set node i to the frozen replacement. Define the raw signature

`a_i(x; r) = g(x; r) - g(x; r, A_i)`.

The router remains active in *both* terms. Recomputing ablations only on the original clean run,
or replacing the signature with activation distances, would not measure dependency reorganization.
The S2-corrupt replacement bank stays fixed to the unedited recipient so the measurement's
reference distribution does not move with the routing edit. A separate edited-reference sensitivity
may later be declared, but cannot silently replace this estimand.

## Scale and centroids

For each coordinate and each baseline, estimate pooled within-family standard deviation on
discovery data only (pooled residual sum of squares divided by N-2). Set

`scale_i = max(discovery_within_family_sd_i, 0.10 logits)` and `s_i = a_i / scale_i`.

The floor prevents almost inert coordinates from dominating through division by numerical noise.
Record its effect and retain raw signatures. Do not normalize each prompt by its own clean–corrupt
gap, since near-zero gaps create unstable ratios and output differences can confound selection.

Compute `mu_ABBA` and `mu_BABA` as discovery means of s. Fit no dimensions, weights, thresholds,
or rotations on held-out data. Keep both 72-dimensional signatures intact.

## Donorward movement

For recipient family B and donor family A, let `v = mu_A - mu_B` and
`delta = s(x; routing_from_A) - s(x; 0)`. Define

`S(B→A) = dot(delta, v) / dot(v, v)`.

Reverse v for the other direction so positive always means donorward movement. Do not clip S:
negative values, overshoot above 1, and harmful changes are diagnostically useful.

`orthogonal = norm(delta - S*v) / norm(v)`

also report `norm(delta)`, distance to donor centroid before/after, and endpoint/answer effects.
S≈1 denotes a projection comparable to the mean family gap. It does not imply that the patched
point equals the donor centroid, since the recipient starts away from its own centroid and can
move far off-axis. Decline to interpret switching when family separation fails the pilot gate;
raise an explicit error when `dot(v,v) < 1e-8`.

## Pilot-only axis statistic

Before routing experiments, project matched natural ABBA-minus-BABA differences onto the
discovery `mu_ABBA - mu_BABA` axis, divided by its squared norm. This tests held-out natural
separation. It is **not** a causal transfer score and must never be reported as one.

## Interpretation limits

A changed single-node effect can arise from nonlinear interactions, altered task confidence,
normalization, redundant pathways, or destructive activation mismatch. Require low orthogonal
movement relative to valid within-family patches, retained behavior, four-donor specificity,
random/edit controls, and independent joint/pathway validation. Finite signatures constrain the
claim to measured dependencies. A probe, logit change or post-edit rediscovered circuit alone
cannot establish routing.

