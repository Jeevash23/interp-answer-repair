# Bridge 008: source coverage and donor-axis sensitivity

**Completed and independently verified.** All four new conditions and fresh-engine
computational replay finished on 4 October 2026 by 21:30:51 UTC. The canonical
independent audit passed at 21:32:10 UTC, checking six measurement records,
source operators, frozen inputs and all four primary intervals. The complete
four-contrast donorward-increase criterion was **not met**. Audit success verifies
the measurements and arithmetic; it is separate from scientific success.

Read the [complete results](RESULTS.md), including all cellwise interactions,
behavioral changes, endpoint distances and matching diagnostics.

| Coverage | Reference | Mean interaction [adjusted interval] |
| --- | --- | --- |
| Broad | Zero | 0.015515 [0.009219, 0.022645] |
| Broad | Fit mean | 0.017512 [0.009069, 0.027367] |
| Matched broad | Zero | 0.006922 [0.001459, 0.013015] |
| Matched broad | Fit mean | 0.005666 [−0.000835, 0.012803] |

Both raw broad interactions were positive under the four-quantity adjusted
interval family. The matched fit-mean interval spans zero, so the joint
norm-controlled claim is unresolved. All eight raw cellwise pointwise intervals
were positive; among matched cells, only F10/order 1 was positive under both
references. Other matched cell intervals span zero.

Broad wording recovery was 2.439 logits with 92.4% top-1 accuracy, compared with
2.332/90.7% for narrow and 1.873/87.7% for matched broad. The matched-minus-narrow
wording recovery difference was −0.458 logits [pointwise −0.522, −0.393].
Thus norm matching reduced behavioral repair while yielding mixed donor-axis
evidence. No undefined norm matches occurred and every planned condition was
measured without dropping configurations.

This is an exploratory, outcome-informed follow-up to Bridge 007, using the
same F10/F11 cases and inspected narrow-transfer baseline. It asks whether
wider source coverage increases the wording-versus-self donor-axis difference,
and whether that increase remains after matching the edit norm in each layer.
It is a correlated intervention comparison, not an independent replication.

## Fixed comparison

The study retains Bridge 007's 3,200 configurations: 20 context blocks crossed
with 20 disjoint name triples, both wordings, mention orders and roles. It
reuses the same checkpoint, clean/corrupt observations, fitted references,
coordinate scales and zero/fit-mean replacement banks. Eligibility required
both independent and final clean qualification with resolvable endpoints;
it did not depend on the success of Bridge 007's primary transfer predictions.

Three coverage conditions each have wording and self donors:

- **Narrow:** the twelve historical S2 source heads; reuse Bridge 007's measurements.
- **Broad:** all 112 heads in the same seven layers, 6–12, at S2.
- **Matched broad:** the broad direction scaled separately in each layer to
  match the norm of that configuration/donor's summed narrow residual addition.

The four new conditions are `broad_wording`, `broad_self`, `matched_wording`
and `matched_self`. They retain all 176 downstream END measurement heads.
This does not expand to every source layer, replace the full donor state or
identify a complete circuit. Undefined norm matching makes the entire affected
matched condition unavailable; examples are not removed or silently repaired.

Four primary signed interactions compare broad versus narrow and matched
versus narrow wording/self differences, each under both references. The
frozen analysis uses 50,000 crossed-bootstrap draws and one four-quantity
adjusted interval family. All four adjusted lower bounds must exceed zero for
the complete positive criterion. Cellwise heterogeneity and behavioral effects
are mandatory. A null signed interaction is not equivalence or invariance;
clean-minus-corrupt subtraction can cancel wording information common to both
source states.

## Evidence and provenance

The input freeze was written at **2026-10-04 20:09:31.428590 UTC**, covers
181 files, and has SHA-256
`418cd9b2cade7911ad18355065c59b526e2e7561baa950fbf203bb0517277339`.
It binds the protocol, scientific implementation, exact cases, reused Bridge
007 observations/references, checkpoint identity and Bridge 007's canonical
audit. This is a local freeze before new broad outcomes, not public
preregistration. The decision to study coverage was informed by earlier results.

Calibration passed on eight fixed cases at 20:11:06 UTC, including explicit
112-head versus efficient layer-sum checks and full-sequence versus prefix
checks, within the 0.0005 tolerance. These technical checks are not evidence
for the four scientific predictions.

Read the frozen [protocol](docs/PROTOCOL.md),
[implementation details](code/README.md) and
[safe reproduction instructions](REPRODUCIBILITY.md).

| Location | Retained contents |
| --- | --- |
| `data/design.json`, `data/input-freeze.json` | Exact reused rows, eligibility, planned comparisons and frozen file hashes |
| `results/calibration.json`, `results/calibration/` | Technical operator and numerical validation |
| `results/natural/`, `results/deltas/` | Broad layer sums, injected vectors and norm-matching diagnostics |
| `results/test/` | New condition signatures/behavior, chunk records or explicit non-evaluable markers |
| `results/execution.log` | Progress only; measurement records govern numerical completion |
| `results/inference-complete.json`, `results/computational-replay.json` | Completed conditions and passed fresh-engine singleton replay |
| `results/analysis/` | Complete primary and secondary saved-observation analysis |
| `results/verification.json` | Independent audit of completed observations and primary statistics |
| `RESULTS.md`, `tables/summary.tex` | Report generated from the terminal, verified study |

Fresh-engine replay checked eight fixed cases under every new condition. The
maximum singleton signature error was 0.0000143051, below 0.0005, with identical
behavior. Independent primary interval reconstruction differed by at most
6.94×10⁻¹⁸. These are computational checks on the same cases, not new samples.

Preserve the canonical Bridge 007 audit and all frozen inputs. Routine
reproduction uses temporary-output audits and saved-observation replays;
rerunning a completed model pipeline can change hash-bound timestamps.
All scientific stages and combined manuscript/release checks are complete. The final manuscript has 33 visually reviewed pages and no LaTeX warnings. These overview documents are unfrozen explanations.
