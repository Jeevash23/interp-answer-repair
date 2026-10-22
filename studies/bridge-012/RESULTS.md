# Study 7 results: GPT-2 Large stops at pair selection

Run on Modal (L4) on 17 October 2026, 04:36–04:51 UTC, after the protocol freeze (03:53:15 UTC, after the author's approval) and the input freeze (04:35:00 UTC). **The frozen rule stopped the study at pair selection.** No pair qualified, so no independent qualification, positive-control development or final transfer was measured. As with Study 4 in GPT-2 Small, this is a bounded outcome. AI-01's qualified results remain those of one checkpoint, and the stop is not evidence about transfer in GPT-2 Large.

Every number below comes from the records pulled back from the Modal Volume (`results/`).

## Stages

| Stage | Outcome |
|---|---|
| Calibration (technical bank, technical heads) | Passed. The checks: causal-prefix against full-sequence signatures within 3.4 × 10⁻⁵ logits for every condition type, including held heads; held heads exactly at the bank; the full layer-19 replay within 7.2 × 10⁻⁷ of the donor; self transfer at zero dose exactly the corrupted run; frozen-scale attribution within 1.6 × 10⁻⁵ of the logit gap; summed head outputs within tolerance in logits; GPU against a Modal CPU container within 4.8 × 10⁻⁵ |
| Behavioural screen (48 rows per family) | F10 and F13 passed; F11 and F12 failed. One candidate pair, F10/F13, was eligible |
| Source heads (development bank, F10 and F13 rows, 380 candidate heads in layers 0–18) | Passed with 12 heads: L12H13, L15H4, L8H14, L16H0, L0H14, L17H5, L13H6, L8H1, L7H5, L15H6, L16H5, L17H16. Their joint self transfer recovered 1.974 of 5.822 logits (33.9% of the clean gap), with pairwise accuracy 93.4% |
| Fit and selection (256 rows per family in each bank) | **Stopped.** F10/F13 failed two of Study 2's selection criteria (below) |

**Behavioural screen.** Pairwise accuracy (IO above subject) / top-1 accuracy (IO is the top token) / mean logit gap, by mention order:

| Family | Order 0 | Order 1 | Gate (both orders: pairwise ≥ 95%, top-1 ≥ 85%) |
|---|---|---|---|
| F10 | 100% / 100% / 6.52 | 100% / 100% / 7.74 | passed |
| F11 | 100% / 54.2% / 5.83 | 100% / 79.2% / 6.78 | failed (top-1) |
| F12 | 100% / 58.3% / 5.44 | 100% / 79.2% / 5.50 | failed (top-1) |
| F13 | 100% / 87.5% / 5.98 | 100% / 87.5% / 6.90 | passed |

GPT-2 Large always prefers the indirect object to the subject, but in F11 and F12 the indirect object is often not its top token. Study 4's GPT-2 Small attempt stopped on the same criterion in F11 (top-1 83.3% and 58.3%).

**Selection of F10/F13** (Study 2's rule: behavioural gates in the fit and selection banks; classification at least 80% in every fit cell and at least 90% in every selection cell; zero / fit-mean reference):

| | F10, order 0 | F10, order 1 | F13, order 0 | F13, order 1 |
|---|---|---|---|---|
| Fit top-1 (pairwise 100% throughout) | 100% | 100% | 96.9% | 96.1% |
| Selection top-1 (pairwise 100% throughout) | 100% | 100% | **84.4%** | **84.4%** |
| Fit classification | 96.9% / 96.9% | 100% / 100% | 96.1% / 100% | 99.2% / 100% |
| Selection classification | **86.7% / 88.3%** | 100% / 100% | 96.9% / 96.9% | 96.9% / 99.2% |

F13's top-1 accuracy on the selection bank (84.4% in both orders) fell just below 85%. The worst selection classification cell (F10, order 0: 86.7% and 88.3%) fell below 90%. Either failure alone ends selection. No other pair was eligible, and the frozen rule allows no substitution.

## Predictions against outcomes

| Prediction (PREDICTIONS.md) | Outcome |
|---|---|
| At least one pair eligible after the screen (about 75%); all four families pass (60%) | One pair eligible; two families failed |
| Source heads in layers 10–18; joint recovery 35–60% of the clean gap | Eight of twelve in layers 10–18, three in layers 7–8, one in layer 0; 33.9%, just below the range; gate passed |
| A pair is selected (about 85%) | Not selected |
| A qualified final test overall (about 40%) | No |

## Records and costs

- **Results.**
  - The JSON records, freezes, runner records with chunk hashes, the source-scan array (`results/development/source-scan.npz`) and the fitted bank are in `results/`, pulled back from the Modal Volume `ai01-study7`.
  - The larger arrays stay on the Volume: calibration signatures, natural runs with END outputs and the fit/selection signatures. Twenty files listed in the stage freezes are there only; `code/freeze012.py verify` lists them.
- **Stage log.** `provenance/stage-log.jsonl` (one line per stage, with Modal's billing figure); run logs are in `provenance/run-group1.log`.
- **Cost (Modal's billing report).**
  - Study 7 run: $0.30 (L4 $0.20, memory $0.08, CPU $0.02).
  - Code tests: $0.48.
  - Smoke test: $0.10.
  - Total: $0.89; $0.00 billed after the monthly credits (`provenance/billing-2026-10-17.json`).
- **Data transfer.** About 12 MB uploaded (code, the 10.7 MB bank file, lock file); 0.66 MB of records brought back. The model, packages and arrays stayed on Modal.

## Disclosures

- **Code tests before the input freeze.** Five code-test attempts ran the whole pipeline on GPT-2 Medium on Study 7's calibration bank, on separate Volumes (`ai01-study7-codetest`, `-2` to `-5`). They found and fixed three bugs:
  - the code-test input freeze returned a hash instead of a path;
  - Volume paths are symlinks and must not be resolved when hashing;
  - the summed-head-output calibration check compared residual-stream vectors against a logit tolerance. It now converts the discrepancy into logits through the frozen final LayerNorm scale and the answer's unembedding direction, as the protocol's "logits" requires.

  The code test also showed, in GPT-2 Medium, that the source-head rule recovers 10 of AI-01's original 12 source heads and that the name-mover rule finds 2 of Study 6's 3 held heads. No GPT-2 Large task prompt was run before the input freeze.
- **Calibration bank size.** The protocol's bank table lists 32 calibration rows per family; a 2 × 2 bank has 16 per family (64 in all). The bank definition governs.
- **README after the run.** The study README was listed in the protocol freeze as a navigation document and was rewritten after the run to give the final status. The frozen version is preserved byte for byte in `provenance/README-at-protocol-freeze.md`, recovered from the editor's history and checked against the frozen hash. `provenance/navigation-updates.json` records both hashes, and `code/verify012.py` accepts the change only through that record.
- **Code-test Volumes.** The five code-test Volumes hold only throwaway GPT-2 Medium test outputs, a few MB each. They were kept rather than deleted; deleting them is the author's choice.
