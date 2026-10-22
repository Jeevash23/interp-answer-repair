# Study 6 results: positive controls for the ablation signature

Completed and independently verified on 16 October 2026. Held-out inference ran once, 10:21–10:47 UTC, after the development freeze. Everything below comes from `results/analysis/summary.json` (held-out), `results/analysis-development/summary.json` and `results/development.json`. Distances are in clean wording separations, d(E, E′) = ‖W(E − E′)‖ / ‖W(D − R)‖, over the 173 END heads other than the three held heads. Pairs are zero / fit-mean replacement.

## Outcome

| Control | Compared with | Reading under the frozen rule |
|---|---|---|
| N1: clean, name movers held | clean recipient | **not detected** (intermediate under both references) |
| N2: self transfer, name movers held | self transfer at dose κ = 0.80 | **below AI-01's resolution** under both references |
| O: order donor | clean recipient | **detected** under both references |

The protocol's reading for this outcome (third row of its table): the signature cannot register a known change of route at the scale of the wording separation, even in clean runs. AI-01's recipient proximity and bounded wording/self similarity are limits of the measurement, not statements about the computation; the answer-repair results stand. Because N2 is below 0.5, Study 2's full-discrepancy bound could not have distinguished this route change from self transfer. Because O is detected and N1 is not, the signature follows the computed answer rather than the route. The route-changed runs keep negative margins, so the margin cannot identify a route.

## Primary quantities (held-out bank, 3,200 configurations)

Means with Bonferroni-adjusted crossed-bootstrap intervals (six-quantity family, 50,000 draws).

| Control | Zero replacement | Reading | Fit-mean replacement | Reading |
|---|---|---|---|---|
| N1 vs clean recipient | 0.530 [0.480, 0.582] | intermediate | 0.495 [0.446, 0.547] | intermediate |
| N2 vs self transfer at κ | 0.422 [0.383, 0.462] | below 0.5 | 0.412 [0.377, 0.451] | below 0.5 |
| O vs clean recipient | 1.951 [1.821, 2.069] | detected | 1.874 [1.756, 1.981] | detected |

## Development stage (288 configurations)

The top three candidates passed all three conditions at k = 3: **L16H15, L20H12 and L15H14** (mean direct attribution 0.798, 0.759 and 0.737 logits; END attention to the indirect object 0.53–0.66, to either subject mention at most 0.054). They carried 43.7% of the clean logit gap by direct attribution. Holding them kept 86.9% of the clean gap with 100% pairwise accuracy (N1), and 70.9% of self transfer's recovery with 88.5% pairwise accuracy (N2). κ = 0.80 matched N2's recovery most closely (2.003 against 1.932 logits). Prefix-reuse and full-sequence signatures with the held heads agreed within 2.3 × 10⁻⁵ on eight development rows.

Development primary distances: N1 0.537/0.491, N2 0.428/0.412, O 2.088/1.938, with the same readings as the held-out bank.

## Secondary quantities (held-out; pointwise 95% intervals)

**Distances from the recipient and margins.** Partial repair alone puts a run about one separation from the recipient: self transfer 1.013/0.906, wording transfer 1.026/0.921, the corrupted run 1.514/1.423. N1 sits 0.530/0.495 from the recipient with margins −0.580/−0.608 and donor-axis score 0.026/0.017: the route change is almost orthogonal to the donor axis and leaves the run strongly recipient-favouring. N2's margins are −0.223/−0.259, and the order donor's −0.035/−0.030. The full replay sits exactly 1 from the recipient and 0 from the donor.

| Condition | q_R zero | margin zero | q_R fit-mean | margin fit-mean |
|---|---|---|---|---|
| Corrupted, no edit | 1.514 [1.445, 1.579] | -0.094 [-0.107, -0.082] | 1.423 [1.364, 1.478] | -0.104 [-0.118, -0.091] |
| Wording transfer | 1.026 [0.971, 1.081] | -0.279 [-0.300, -0.258] | 0.921 [0.876, 0.967] | -0.303 [-0.323, -0.283] |
| Self transfer | 1.013 [0.959, 1.068] | -0.299 [-0.320, -0.278] | 0.906 [0.862, 0.951] | -0.334 [-0.354, -0.314] |
| Self transfer, dose 0.80 | 1.118 [1.055, 1.180] | -0.251 [-0.272, -0.231] | 1.020 [0.968, 1.072] | -0.279 [-0.298, -0.259] |
| N1: clean, name movers held | 0.530 [0.492, 0.569] | -0.580 [-0.601, -0.559] | 0.495 [0.458, 0.533] | -0.608 [-0.628, -0.588] |
| N2: self transfer, name movers held | 1.159 [1.101, 1.215] | -0.223 [-0.241, -0.204] | 1.029 [0.983, 1.073] | -0.259 [-0.278, -0.240] |
| O: order donor | 1.951 [1.856, 2.040] | -0.035 [-0.045, -0.026] | 1.874 [1.789, 1.954] | -0.030 [-0.041, -0.020] |
| X: full layer-13 replay | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

**Fresh-data replication of Study 2.** Wording recovery is positive in all four cells and every wording margin is negative. The wording/self discrepancy F is 0.169/0.187 over the retained heads (0.171/0.190 over all 176), and the mean absolute donor-axis difference A is 0.033/0.043 (0.033/0.045): both inside Study 2's bounds of 0.50 and 0.10, and close to Study 2's 0.159/0.174 and 0.027/0.034.

**Ratios to the wording/self discrepancy on the same rows.** N1/F = 3.135 [2.771, 3.538] / 2.646 [2.319, 3.018]; N2/F = 2.494 [2.187, 2.845] / 2.205 [1.921, 2.536]. The signature does separate the known route change from wording versus self transfer, at a scale below one wording separation.

**Amplitude.** Reducing self transfer's dose from 1 to 0.80, with no route change, moves the signature 0.271/0.285, about two thirds of N2's 0.422/0.412. Removing the best multiple of the comparator leaves N1 0.518/0.483 and N2 0.407/0.398 (best multiples 0.988–1.005), so the route changes are not rescalings; the dose change leaves 0.242/0.255 (best multiple 0.93/0.92).

**All 176 heads.** Including the held heads' own coordinates: N1 0.599/0.577, N2 0.484/0.485, O 1.959/1.895.

**Cells.** No route-change distance reaches one separation in any wording/order cell (N1 0.389–0.644, N2 0.303–0.567); O ranges from 1.562 to 2.164.

**Behaviour and direct attribution.**

| Condition | Logit gap | Recovery over corruption | Top-1 | Pairwise |
|---|---|---|---|---|
| Clean recipient | 4.560 [4.272, 4.860] | 4.560 [4.272, 4.860] | 98.7% [96.8, 99.8] | 100.0% [100.0, 100.0] |
| Corrupted, no edit | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 33.6% [29.9, 37.6] | 50.0% [50.0, 50.0] |
| Wording transfer | 2.333 [2.122, 2.545] | 2.333 [2.122, 2.545] | 92.3% [88.2, 95.8] | 95.7% [92.9, 97.9] |
| Self transfer | 2.414 [2.201, 2.627] | 2.414 [2.201, 2.627] | 90.8% [86.4, 94.8] | 94.9% [91.8, 97.4] |
| Self transfer, dose 0.80 | 1.769 [1.573, 1.972] | 1.769 [1.573, 1.972] | 82.1% [76.1, 87.6] | 89.6% [85.3, 93.4] |
| N1: clean, name movers held | 4.138 [3.871, 4.419] | 4.138 [3.871, 4.419] | 98.7% [96.8, 99.9] | 100.0% [99.8, 100.0] |
| N2: self transfer, name movers held | 1.771 [1.598, 1.948] | 1.771 [1.598, 1.948] | 86.4% [81.2, 91.1] | 90.5% [86.2, 94.1] |
| O: order donor | -2.129 [-2.329, -1.927] | -2.129 [-2.329, -1.927] | 5.8% [2.9, 9.2] | 7.5% [4.2, 11.2] |
| X: full layer-13 replay | 4.560 [4.272, 4.860] | 4.560 [4.272, 4.860] | 98.7% [96.8, 99.8] | 100.0% [100.0, 100.0] |

The held heads carry 42% of the clean gap by direct attribution (70% of self transfer's gap). Held at the mean, they contribute nothing input-dependent, and the other heads' summed direct attribution rises by 81% of the loss in N1 and 64% in N2: the answer is carried by backup heads.

Clean gates on the held-out bank (Study 2's): behaviour and classification pass in every cell; no denominator is unresolvable.

## Predictions against outcomes

| Prediction (PREDICTIONS.md) | Outcome |
|---|---|
| k = 3, layers 15–20, 30–60% of the clean gap | k = 3 in layers 15, 16 and 20; 43.7% (development), 42% (held-out) |
| N1 keeps 55–85% of the clean gap, pairwise ≥ 95% | 86.9% (development), 91% (held-out); pairwise 100% |
| N2 keeps 50–85% of self's recovery; κ 0.60–0.90 | 70.9%; κ = 0.80 |
| O 1.8–3.5, detected | 1.951/1.874, detected |
| N1 0.8–2.0, probably detected | 0.530/0.495, not detected (missed) |
| N2 0.4–1.0, most likely intermediate | 0.422/0.412, below resolution |
| Overall: row 2 of the protocol table | Row 3 |
| Margins of N1 and N2 negative | Yes |
| F 0.13–0.22, A 0.02–0.05 | F 0.169/0.187, A 0.033/0.043 |
| Dose-only distance 0.1–0.4 and below N2's | 0.271/0.285, below N2's |
| Shape distance of the dose change below 0.15 | 0.242/0.255 (missed) |
| Anchor exact | Exact |
| Backup heads recover 20–60% of the lost attribution | 81% (missed: more compensation than predicted) |

## Verification

- Calibration on the fresh calibration bank passed every check: prefix-reuse against full-sequence signatures within 1.9 × 10⁻⁵ logits for all condition types; held heads exactly at the bank; full replay within 7.2 × 10⁻⁷ of the donor; self transfer at zero dose exactly the corrupted run; GPU against CPU within 2.4 × 10⁻⁵; the frozen-scale attribution of the final residual within 6.7 × 10⁻⁶ of the logit gap; summed head outputs within 1.3 × 10⁻⁴ of each layer's attention output (up to a shared bias).
- Fresh-process replays of the eight fixed held-out rows, one prompt and one head at a time: GPU, all nine conditions, within 2.7 × 10⁻⁵; CPU, three conditions, within 2.1 × 10⁻⁵.
- Independent audit (`results/verification.json`): the three freezes, all 47 runner records and their chunk hashes, the stage records, and a plain-NumPy reconstruction of every primary mean, interval and reading, F, N1's margins and the anchor from the saved arrays and saved bootstrap indices (largest difference 4.4 × 10⁻¹⁶).

## Runtime and storage

Calibration took 3 min, development 9 min and held-out inference 26 min (1,557 s); another project's job shared the GPU during calibration and development. Replays took 7 min. The study's results take 1.5 GB.

## Disclosures

- The run settings were 8 prompts and 8 heads per pass, within the protocol's measured range, because the other job held 2.8 GB of the GPU.
- Before the input freeze, the complete code was tested on Study 2's 32 calibration rows (`provenance/code-smoke-test/`). That run showed preliminary choices and readings for those rows. The protocol had already been frozen and approved, and nothing was changed.
- The first launch of calibration failed at import, before any computation, because Python's isolated mode left the code folder off the path; it was rerun from the project root.
