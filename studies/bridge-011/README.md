# Bridge 011 (Study 6): can the ablation signature detect a known change of route?

**Completed and independently verified.** Held-out inference ran once on 16 October 2026, 10:21–10:47 UTC, after the protocol (09:24:57 UTC, before the author's approval), input (10:07:13) and development (10:20:54) freezes. The frozen analysis, fresh-process GPU and CPU replays and the independent audit passed. The held-out test is prospective; the held heads and the dose κ were fixed on development data.

**Question.** Studies 1 and 2 found that the repaired run's 176-head ablation signature sits nearer the clean recipient than the clean donor, and that self transfer gives nearly the same signature. Does the signature register a change in the downstream computation that is known in advance? Each positive control should move it by more than one clean wording separation from the condition it is known to differ from.

**Answer.** Only the reversed answer is detected. On 3,200 held-out configurations, the order donor moves the signature 1.951/1.874 separations from the recipient (zero/fit mean; adjusted lower limits 1.821/1.756). Holding the three primary name movers (L16H15, L20H12, L15H14) at their mean, so that backup heads carry the answer, moves it 0.530/0.495 in clean runs (adjusted upper limits 0.582/0.547: not detected) and 0.422/0.412 in repaired runs (upper limits 0.462/0.451: below Study 2's 0.5 bound). The route-changed runs still favour the recipient (N1 margins −0.580/−0.608). Under the frozen reading, the signature cannot register this route change at the scale of the wording separation, so AI-01's recipient proximity and bounded wording/self similarity are limits of the measurement rather than statements about the computation; the answer-repair results stand. On the same rows the route change is still 2.2–3.1 times the wording/self discrepancy (0.169/0.187), and a dose change of self transfer alone moves the signature 0.271/0.285 (secondary, pointwise).

See [RESULTS.md](RESULTS.md) for every result, [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands, [docs/PROTOCOL.md](docs/PROTOCOL.md) and [docs/PREDICTIONS.md](docs/PREDICTIONS.md) for the frozen design, and [docs/AUTHOR-APPROVAL.txt](docs/AUTHOR-APPROVAL.txt) for the approval.

## Design

- GPT-2 Medium, the F10/F11 pair, the twelve S2 source heads, the 176 END heads (layers 13–23), and Study 2's fitted scales and mean bank, without refitting.
- Fresh banks: 84 names unused in any earlier study, and 28 unused location/object pairs (16 from Study 2's declared pool, 12 new). Calibration 2 × 2, development 6 × 6 and held-out 20 × 20 blocks × triples, each crossed with both roles, mention orders and wordings (32, 288 and 3,200 rows).
- Controls: **N1**, the clean recipient with the three primary name movers held at their mean at END in every pass, compared with the clean recipient; **N2**, self transfer with the same heads held, compared with self transfer at a dose κ matched in recovery on development data; **O**, order-donor transfer, compared with the clean recipient. Anchor **X**: full layer-13 residual replay of the donor.
- Distance d(E, E′) = ‖W(E − E′)‖ / ‖W(D − R)‖ over the 173 END heads other than the held ones. A control detects its change when the Bonferroni-adjusted lower limit (six quantities) exceeds 1; an adjusted upper limit below 0.5 is below Study 2's full-discrepancy bound.

## Records

| File | Contents |
|---|---|
| `data/protocol-freeze.json` | Protocol, predictions, runtime benchmark and inherited inputs, hashed before approval (16 Oct, 09:24:57 UTC) |
| `data/input-freeze.json` | Code, banks, docs, bootstrap indices, inherited code and checkpoint (10:07:13 UTC, 167 files) |
| `data/development-freeze.json` | Development record and measurements, before held-out inference (10:20:54 UTC, 1,163 files) |
| `results/calibration.json` | Technical checks on the fresh calibration bank |
| `results/development.json` | Held heads, κ, candidates, trials and development checks |
| `results/test/` | Held-out measurements with resumable records and chunk hashes |
| `results/analysis/`, `results/analysis-development/` | Prespecified analyses (summary and per-example arrays) |
| `results/gpu-replay.json`, `results/cpu-replay.json` | Fresh-process singleton replays |
| `results/verification.json` | Independent audit |
| `provenance/` | Run logs, the design-stage runtime benchmark and the pre-freeze code smoke test |

## Verification

Calibration passed every check (prefix against full-sequence signatures within 1.9 × 10⁻⁵ logits, GPU against CPU within 2.4 × 10⁻⁵, full replay within 7.2 × 10⁻⁷ of the donor, held heads and zero dose exact). Fresh-process replays of eight fixed held-out rows agree within 2.7 × 10⁻⁵ (GPU, nine conditions) and 2.1 × 10⁻⁵ (CPU, three conditions). The independent audit reconstructs every primary mean, interval and reading from the saved arrays and bootstrap indices (largest difference 4.4 × 10⁻¹⁶) and checks all 47 runner records.

**Preserve this evidence tree.** Do not rerun the stages here; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
