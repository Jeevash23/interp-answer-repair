# Findings in brief

All studies are complete and independently verified: the fresh wording confirmation (Bridge 007, Study 2), source-coverage follow-up (Bridge 008, Study 3), fixed-pair GPT-2 Small attempt (Bridge 009, Study 4), exploratory geometric null (Bridge 010, Study 5), prospective positive control (Bridge 011, Study 6) and GPT-2 Large attempt (Bridge 012, Study 7). The [manuscript](paper/manuscript.pdf) (55 pages, version 3.6 of 17 October 2026, the review copy; version 3.7 of 21 October 2026, its arXiv and TMLR builds, is in [`submission-2026-10/`](submission-2026-10/README.md)) includes the successful confirmation, the mixed coverage result, the two second-checkpoint stops, the geometric null, the positive control and post hoc injected-difference analyses.

| Question | Current evidence |
|---|---|
| Does the edit repair the answer? | Yes, in two independently qualified GPT-2 Medium wording comparisons. F6/F7 raises top-1 accuracy from 35.8% to 86.7% (93.1% clean); fresh F10/F11 raises it from 33.9% to 90.7% (96.8% clean). Both recover about half of the mean clean logit gap, and all four adjusted recovery intervals are positive in each study. |
| Are edited dependencies closer to the clean donor or recipient? | Closer to the clean recipient on average: all eight adjusted primary margin intervals favor it in each qualified study. Post hoc, the unedited corrupted signature already favours the recipient in five (Study 1) and six (Study 2) of the eight cells, and the edit moves every cell further towards it. Relative proximity does not mean clean restoration; edited signatures remain about one clean-wording separation from the recipient endpoint. |
| Does recipient proximity identify a donor-wording-specific effect? | No. Self transfer also gives recipient-favoring signatures. The fresh prospective comparison supports explicit average self-versus-wording discrepancy bounds under both references. Small donorward traces remain detectable, so this is neither signature identity nor evidence that wording has no effect. The comparison holds recipient context fixed and does not isolate why the two edits give similar responses. |
| Is the evidence still based on one qualified wording pair? | No. F10/F11 adds a second qualified pair, using new names and context combinations and larger crossed sampling banks. It is a close replication within related template structures, not evidence for arbitrary syntax. |
| Is there a qualified second model? | No. The fixed F10/F11 GPT-2 Small attempt stops at its frozen behavioral screen: F11 top-1 rates are 83.3% and 58.3%, below 85%, despite 100% pairwise accuracy. The prospective GPT-2 Large attempt (Study 7, on rented GPUs) passes calibration; F11 and F12 fail its screen on top-1 accuracy, and the only eligible pair, F10/F13, fails selection: F13's selection-bank top-1 accuracy is 84.4% (gate 85%), and the weakest classification cell 86.7% (gate 90%). Neither attempt measures final transfer, so these eligibility failures do not test the transfer predictions. |
| Are wording directions discounted downstream? | Not on average. Holding each wording edit's per-head norm and angle to self fixed, wording directions change the signature about as much as natural name/context directions (full-discrepancy ratios 1.067 and 1.104, adjusted intervals spanning 1), and wording directions from mismatched configurations do not differ detectably from them (0.997; adjusted intervals 0.898–1.106 and 0.898–1.104). Random head-subspace directions, a weak null, change it less. The excess over random and name/context turns is confined to mention-order-1 cells. Exploratory, on the inspected Study 2 cases. |
| Can the signature detect a known change of route? | Only weakly. Holding the three primary name movers at their mean, so that backup heads carry the answer, moves the signature 0.530/0.495 clean separations in clean runs and 0.422/0.412 in repaired runs (adjusted upper limits 0.582/0.547 and 0.462/0.451); only the order donor, which reverses the answer, passes the prespecified one-separation rule (1.951/1.874). The route-changed runs still favour the recipient. Recipient proximity and bounded wording/self similarity are therefore limits of this measurement, not statements about the computation. Prospective, on fresh names and contexts. |
| Does broader source coverage change the wording-versus-self response? | At natural strength, expanding twelve heads to all 112 in the same seven layers increases the donorward wording-versus-self differential under both references. After matching layerwise injection norms, only the zero-reference adjusted interval excludes zero; the mean-reference interval spans zero. The complete four-part positive criterion is not met. This is a correlated exploratory follow-up on the same cases. |

## Fresh prospective confirmation

Bridge 007 uses 3,200 final configurations crossing **20 context blocks and 20 disjoint name triples**, with paired roles, mention orders and wordings. These are not 3,200 independent samples. All clean qualification stages and all sixteen primary criteria pass. Protocol and primary analysis were locally frozen before new scientific inference; this is not public preregistration.

| Prespecified average discrepancy | Zero replacement | Fitting-mean replacement | Bound |
|---|---:|---:|---:|
| Absolute donor-axis difference | 0.0274 [0.0209, 0.0348] | 0.0344 [0.0264, 0.0444] | < 0.10 |
| Full normalized signature discrepancy | 0.1586 [0.1374, 0.1816] | 0.1737 [0.1497, 0.1997] | < 0.50 |

Brackets are Bonferroni-adjusted intervals from the same sixteen-quantity family and 50,000 crossed-bootstrap draws. Absolute values and vector norms are taken before averaging. These bounds concern average assay discrepancy, not every case or equivalent algorithms.

The manuscript also reports empirical distributions. The individual axial discrepancy exceeds 0.10 in 122/3,200 zero-reference and 214/3,200 mean-reference cases; full discrepancy exceeds 0.50 in 44 and 64 cases. These descriptive counts do not define per-case success or alter the average-bound decisions. All pooled and cellwise distributions were independently reconstructed from verified observations and are retained in the [descriptive report](analysis/results/confirmation-distributions.json).

Wording transfer recovers 2.332 [2.072, 2.608] logits and 50.7% [45.5%, 56.0%] of the mean clean gap (pointwise 95% intervals). All six random-direction recovery intervals include zero. Self transfer recovers 2.414 logits and reaches 90.4% top-1. Secondary cellwise signed wording effects are positive under both references, with the largest mean in F10/order 1; behavioral differences also vary by cell. [Complete Bridge 007 report](studies/bridge-007/RESULTS.md).

## Source-coverage follow-up

Bridge 008 keeps the same 3,200 cases, layers 6–12 at S2, clean endpoints and fitting references. Its interaction is the wording-minus-self donor-axis difference under broad or norm-matched broad coverage minus that under narrow coverage. A positive value need not mean that either edited signature is close to the donor.

| Primary interaction | Mean | Adjusted interval | Decision |
|---|---:|---:|---|
| Broad / zero reference | 0.0155 | [0.0092, 0.0226] | Increase |
| Broad / mean reference | 0.0175 | [0.0091, 0.0274] | Increase |
| Matched broad / zero reference | 0.0069 | [0.0015, 0.0130] | Increase |
| Matched broad / mean reference | 0.0057 | [−0.0008, 0.0128] | Inconclusive |

The separate four-quantity family uses 50,000 crossed-bootstrap draws. Three contrasts support increases, but the specified conjunction of all four does not pass. All eight natural-strength cell intervals are positive pointwise; after matching, only F10/order 1 excludes zero under both references. The other six matched cell intervals span zero. This is not evidence of invariance or of a general norm-independent bottleneck.

Broad wording transfer reaches 92.4% top-1 and 2.439 [2.149, 2.745] logits recovery. Matched broad wording reaches 87.7% and 1.873 [1.607, 2.155] logits; its paired change from narrow is −0.458 [−0.522, −0.393] logits (pointwise 95% intervals). Every case is retained. Matching is defined for every configuration/layer and controls the injected vector norm, not the entire edited state. [Complete Bridge 008 report](studies/bridge-008/RESULTS.md).

## Geometric null (Study 5)

Bridge 010 perturbs self transfer with control directions that keep, per source head, the wording edit's norm and angle to self, so only the perpendicular direction changes. Four frozen replicates per control on all 3,200 Study 2 cases; ratios are wording/self over control/self discrepancy, with 24-contrast Bonferroni-adjusted intervals (50,000 crossed draws that also resample replicates).

| Control | F ratio, zero | F ratio, mean | A ratio, zero | A ratio, mean |
|---|---:|---:|---:|---:|
| Natural name/context | 1.067 [0.944, 1.195] | 1.104 [0.973, 1.243] | 1.087 [0.863, 1.366] | 1.256 [1.013, 1.537] |
| Mismatched configuration | 0.997 [0.898, 1.106] | 0.997 [0.898, 1.104] | 0.992 [0.840, 1.142] | 0.998 [0.859, 1.145] |
| Random head-subspace (weak null) | 1.279 [1.124, 1.427] | 1.320 [1.156, 1.478] | 1.191 [0.891, 1.518] | 1.399 [1.066, 1.793] |

No adjusted ratio interval lies below 1; the only point estimates below 1 are the mismatched-control ratios of 0.992–0.998. Calibration was exact (zero and own-direction perturbations reproduced self and wording transfer with error 0; all 460,800 injections matched), fresh-engine singleton replays agreed within 2.20 × 10⁻⁵ (largest error 2.19 × 10⁻⁵), and an independent reconstruction re-derived every result.

Post hoc, across Study 3's coverage conditions, the injected wording/self cosine falls from 0.925 (twelve heads) to 0.832 (norm-matched broad) and 0.795 (112 heads) as the full discrepancy rises from 0.159/0.174 to 0.204/0.220 and 0.242/0.263. Across Study 2's 3,200 configurations the pooled rank correlation between the relative injected difference and the full discrepancy is negative, −0.305 [−0.421, −0.190] and −0.339 [−0.452, −0.229], because mention order moves the two quantities in opposite directions. Within wording/order cells, a split chosen after seeing that result, the mean correlation is positive under zero replacement, 0.191 [0.025, 0.358]; under mean replacement, 0.165 [0.001, 0.335], the lower limit is zero to within resampling error. For the axial discrepancy the pooled correlations are also negative, −0.127 [−0.216, −0.036] and −0.204 [−0.296, −0.113], while the within-cell means span zero. Intervals in this paragraph are pointwise 95%. [Complete Bridge 010 report](studies/bridge-010/RESULTS.md).

## Positive control (Study 6)

Bridge 011 applies changes whose effect on the downstream computation is known and asks whether the signature moves by more than one clean wording separation from the condition each change is known to differ from. Fresh banks: 3,200 held-out configurations (20 context blocks × 20 name triples) with names and context pairs unused in any earlier study, plus development (6 × 6) and calibration (2 × 2) banks. The protocol and predictions were hashed before any inference on these banks and approved by the author; the development stage fixed only the three held heads (L16H15, L20H12, L15H14) and the matched dose κ = 0.80. Distances exclude the held heads' own coordinates. Brackets are Bonferroni-adjusted (six quantities, 50,000 crossed draws).

| Control | Compared with | Zero replacement | Mean replacement | Reading |
|---|---|---:|---:|---|
| Name movers held, clean run | clean recipient | 0.530 [0.480, 0.582] | 0.495 [0.446, 0.547] | not detected |
| Name movers held, self transfer | self transfer at κ = 0.80 | 0.422 [0.383, 0.462] | 0.412 [0.377, 0.451] | below Study 2's 0.50 bound |
| Order donor | clean recipient | 1.951 [1.821, 2.069] | 1.874 [1.756, 1.981] | detected |

The answer survives the route change: backup heads recover 81% of the held heads' lost direct attribution and the clean gap falls only from 4.56 to 4.14 logits. The route-changed clean runs keep margins of −0.580/−0.608, so they look recipient-like. Secondary (pointwise): on the same rows the route change is 2.2–3.1 times the wording/self discrepancy, which replicates Study 2 at 0.169/0.187, and a 20% dose change of self transfer alone moves the signature 0.271/0.285. Calibration passed, fresh-process replays agree within 2.7 × 10⁻⁵ (GPU) and 2.1 × 10⁻⁵ (CPU) logits, and an independent audit reconstructs every primary result. [Complete Bridge 011 report](studies/bridge-011/RESULTS.md).

## GPT-2 Large attempt (Study 7)

Bridge 012 ran Study 2's pipeline in GPT-2 Large on a rented L4 GPU (Modal). The protocol was approved by the author and hashed before any GPT-2 Large task inference; the banks used fresh names and contexts, and the hardware and image were fixed in advance. A smoke test first reproduced Study 6's eight fixed cases within 2.5 × 10⁻⁵ logits, and the pipeline was code-tested on GPT-2 Medium. Calibration passed on GPT-2 Large. F11 and F12 failed the behavioural screen on top-1 accuracy (54.2–79.2%). Twelve source heads chosen on development data repaired 33.9% of the clean gap. The only eligible pair, F10/F13, failed selection on F13's selection-bank top-1 accuracy (84.4%) and on the F10, order-0 classification (86.7%/88.3%). The frozen rule stopped the study there. Cost by the provider's billing report: $0.30 for the study run, $0.48 for the code tests and $0.10 for the smoke test. [Complete Bridge 012 report](studies/bridge-012/RESULTS.md).

## Historical evidence and scope

The F6/F7 self-transfer and movement analyses remain post hoc. Their signed axial differences (0.018 / 0.020) are a different estimand from the new mean absolute differences. Historical F4/F7 and fresh Medium/TinyLlama comparisons fail their clean-separation requirements; the initial Medium comparison used an earlier design, and Pythia fails behavioral screening. These outcomes are retained in the manuscript and archive.

The fresh confirmation uses related two-sentence and subordinate-clause forms, a fixed source set and fitted references. Individual context words can recur even though the new context combinations are unused in earlier project inputs. Qualification uses observed rates, and uncertainty does not include the selection pipeline. The Small attempt shares the Medium-selected cases and uses a different historical source set and signature dimension. [Complete Bridge 009 stopping report](studies/bridge-009/RESULTS.md).

## Verification

Bridge 007's independent audit reproduces all sixteen primary means and intervals. Fresh-model replay repeats all 3,200 wording and corrupted behaviors exactly; the eight-case singleton-head signature replay has maximum error 0.0000143 logits, below the fixed 0.0005 tolerance. Primary and secondary saved-observation replays reproduce summaries and per-example arrays exactly. Bridge 009's stopping branch, frozen inputs and pre-inference metadata amendment are audited. A passed computational audit is distinct from a supported scientific prediction.

Bridge 008's independent audit reconstructs all four primary and sixteen cell intervals, the actual source vectors, norm matching and all six measurement records. An additional independent arithmetic check reproduces all 630 secondary intervals. The fresh-engine replay's largest signature error is 0.0000143 logits; all behavioral replay errors are zero. The completed follow-up does not change the original confirmation's frozen results or multiplicity family.

Version 3.1 verification (9 October 2026, in the environment rebuilt that day; logs in `provenance/release-validation-2026-10-09/`): all 787 historical evidence files and 42 documented source relocations pass; Bridge 006 records, the three Bridge 007–009 canonical audits and the Study 5 freeze verify; the 93 unit tests pass. The manuscript checker passes 63 historical and 24 extension value checks, the 12 Study 5 headline ratios, 36 Study 5 and post hoc text values, all cross-references and British spelling. All 24 references match their arXiv, Crossref or venue records. The 44-page PDF builds without LaTeX warnings, and every changed page was inspected.

Version 3.2 verification (12 October 2026; logs in `provenance/release-validation-2026-10-12/`): the same integrity checks and 93 unit tests pass; the twelve historical and the Study 2–3 extension replays reproduce exactly, the first replays in the environment rebuilt on 9 October; the plain-NumPy re-implementation reproduces the twelve Study 1 configurations within 4.52 × 10⁻⁵ logits. Two independent audits traced about 1,500 numbers of version 3.1 to the saved outputs with no numeric error; version 3.2 corrects the statements they flagged (the Appendix F provenance, the pooled A correlations and four labels) and adds the post hoc corrupted-signature margins and Study 2's prespecified secondary geometry. The manuscript checker passes 63 historical and 24 extension value checks, 12 Study 5 headline ratios, 35 Study 5 and post hoc text values and 40 version 3.2 values; all 29 references match their records; the 46-page PDF builds without LaTeX warnings or overfull boxes. The paper is ready for review within its stated scope: related wording templates and one qualified model checkpoint.

Version 3.3 verification (16 October 2026; logs and `validation.json` in `provenance/release-validation-2026-10-16/`): the integrity checks pass with the Study 6 freezes (1,348 file hashes) added; the 93 unit tests pass; the manuscript checker passes 63 historical and 24 extension value checks, 12 Study 5 headline ratios, 35 Study 5 and post hoc text values, 40 version 3.2 values and 33 version 3.3 values and tables; all 30 references match their records; the 52-page PDF builds without LaTeX warnings or overfull boxes. Study 6's audit and its GPU and CPU replays passed.

Version 3.4 verification (17 October 2026; logs and `validation.json` in `provenance/release-validation-2026-10-17/`): the integrity checks pass with Study 7's freezes (146 file hashes, one documented navigation update, 20 files held on the provider's storage); the 93 unit tests pass; the manuscript checker adds 21 version 3.4 values; the 55-page PDF builds without LaTeX warnings or overfull boxes.

Version 3.5 verification (17 October 2026; logs and `validation.json` in `provenance/release-validation-2026-10-17-v3.5/`): all 34 references match arXiv and the venues' paper lists; the integrity checks and the 93 unit tests pass; the manuscript checker adds 15 version 3.5 checks; the 55-page PDF builds without LaTeX warnings or overfull boxes.

Version 3.6 verification (17 October 2026, the review copy; logs and `validation.json` in `provenance/release-validation-2026-10-17-v3.6/`): the integrity checks and the 93 unit tests pass; the manuscript checker adds 17 version 3.6 checks (no archive labels in the text, the shared AI-use statement, the PDF details); the 55-page PDF builds without LaTeX or BibTeX warnings and without overfull or underfull boxes.

Version 3.7 verification (21 October 2026, the arXiv and TMLR builds; logs in `submission-2026-10/checks/` and `provenance/release-validation-2026-10-21-v3.7/`): both 50-page builds have no LaTeX or BibTeX warnings and no overfull or underfull boxes; the manuscript checker passes on both builds (`submission-2026-10/tools/check_submission.py`); the arXiv source compiles in an empty folder to identical text; the integrity checks, the 93 unit tests and `make check-paper` on version 3.6 pass. No study file or number changed.
