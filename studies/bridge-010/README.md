# Bridge 010 (Study 5): exploratory geometric null on the Study 2 bank

**Completed and independently verified.** All twelve full-bank conditions finished on 5 October 2026 at 09:15:40 UTC; the frozen analysis, fresh-engine singleton replays and the independent reconstruction passed (audit 09:49:30 UTC). This exploratory study reuses the inspected Study 2 (Bridge 007) final bank and is not pooled with Study 2.

**Question.** Holding each wording edit's per-head norm and angle to self fixed, do wording directions change the downstream signature less than same-geometry control directions (ratio below 1), about as much (interval around 1), or more (ratio above 1)?

**Answer.** No pooled ratio is below 1. Wording directions taken from mismatched configurations reproduce the wording/self discrepancy (F ratio 0.997 under both references). Random head-subspace turns change the signature less than wording turns (F ratios 1.279 and 1.320, adjusted intervals above 1). Natural name/context turns give F ratios of 1.067 and 1.104 with adjusted intervals spanning 1. The excess over random and name/context turns comes from the mention-order-1 cells; in order-0 cells name/context turns give larger point estimates than wording. Under the prespecified interpretation, wording directions are not discounted on average: the bounded wording/self discrepancy of Study 2 reflects how far the wording edit departs from self, not wording-specific insensitivity.

Ratios of pooled means, wording/self over control/self. Brackets: Bonferroni-adjusted intervals over the 24 paired contrasts (50,000 crossed bootstrap draws that also resample each control's four replicates). Wording/self values: F = 0.1586 / 0.1737, A = 0.0274 / 0.0344 (zero / fit mean).

| Control | Reference | F ratio [adjusted CI] | A ratio [adjusted CI] |
| --- | --- | --- | --- |
| Mismatched configuration | Zero | 0.997 [0.898, 1.106] | 0.992 [0.840, 1.142] |
| Mismatched configuration | Fit mean | 0.997 [0.898, 1.104] | 0.998 [0.859, 1.145] |
| Natural name/context | Zero | 1.067 [0.944, 1.195] | 1.087 [0.863, 1.366] |
| Natural name/context | Fit mean | 1.104 [0.973, 1.243] | 1.256 [1.013, 1.537] |
| Random head-subspace | Zero | 1.279 [1.124, 1.427] | 1.191 [0.891, 1.518] |
| Random head-subspace | Fit mean | 1.320 [1.156, 1.478] | 1.399 [1.066, 1.793] |

See [RESULTS.md](RESULTS.md) for differences, behaviour, cells and disruptions, and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands. The author-approved protocol is [docs/PROTOCOL.md](docs/PROTOCOL.md); the approval text is [docs/AUTHOR-APPROVAL.txt](docs/AUTHOR-APPROVAL.txt).

## Design

All 3,200 Study 2 final configurations (20 context blocks × 20 disjoint name triples × roles × orders × F10/F11), GPT-2 Medium, the twelve S2 source heads, all 176 END signature heads, both replacement references and the fitted scales are reused without refitting. For configuration i and head h, with s = δ_self, w = δ_wording and ε = w − s, each control injects s + αŝ + βr⊥, where α = ŝ·ε, β = √(‖ε‖² − α²) and r⊥ is a unit vector orthogonal to s. This keeps ‖w‖, s·w and ‖ε‖, changing only the perpendicular direction.

- **Random:** Gaussian z-space vector projected through the head's W_O, orthogonalised against s.
- **Mismatched:** component of ε(j,h) orthogonal to s(i,h), with j from block and triple derangements that keep wording, role and order.
- **Natural:** component of δ_self(j,h) − δ_self(i,h) orthogonal to s(i,h), with the same mapping.

Four replicates per control (12 full-bank conditions) were fixed before the freeze (3.74 h estimated; 7.48 h for eight replicates).

## Calibration and verification

- Calibration passed before inference: zero perturbation reproduced self transfer and the wording difference's own perpendicular direction reproduced wording transfer, both with maximum error 0.0 (signatures and behaviour); full-sequence versus prefix signature error 6.676e-06, tolerance 5e-4.
- All 460,800 realised head injections matched norm, cross term and perturbation norm (largest absolute errors 6.34e-07, 1.30e-05 and 5.88e-07; 0 violations).
- Fresh-engine singleton replays of eight fixed cases in every condition: maximum error 2.193e-05 (tolerance 5e-4).
- Independent reconstruction ([results/verification.json](results/verification.json)): 4,800 measurement chunks, all 460,800 injections (maximum deviation 0.00e+00), metrics and 54 intervals re-derived from raw saved arrays and literal bootstrap indices.

## Runtime

Observed inference 5.52 h for twelve conditions (20.3–43.7 min each), against the frozen estimate of 3.74 h. The run continued unchanged after the controlling [redacted] session stopped; no condition was restarted or re-measured.
