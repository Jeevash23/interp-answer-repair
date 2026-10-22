# Supplemental controls

Specified 8 September 2026 after the primary Medium results and TinyLlama's discovery/clean behavior were observed, before these supplemental measurements. These are post hoc secondary checks and do not change any primary family, source selection, data, threshold or conclusion rule.

## Nontrivial mean endpoint replay

The initial matched F1/F2 replay used a single donor to form its technical mean bank. Its mean-reference target effects are consequently trivial zeros, although the zero-reference coordinates and separate two-example numerical validation remain informative. Strengthen this control on calibration data only: average the head outputs across all 64 calibration configurations, then replay F2's full downstream boundary into the matched corrupted F1 for all eight available contexts. Require nonzero effects under each reference and retain maximum behavior/signature errors under the original tolerances. This does not address unequal-length F2/F3 whole-state transplantation; that limitation remains explicit.

## Projected random behavior controls

The frozen random deltas are Gaussian in residual space with matched per-head norms. Actual source-head deltas lie in the corresponding output-projection image, whereas these original random vectors need not. Add three behavior-only draws obtained by sampling standard Gaussian head-coordinate vectors, multiplying by that head's output projection, then scaling the projected vector to match the corresponding donor delta's residual-space norm. This preserves membership in the head-output subspace; it does not match the distribution of natural activations or produce uniform directions in that subspace when the projection is anisotropic.

Use the same selected heads, fresh test cases and semantic positions, with deterministic per-model/example/draw labels and seed 2026090804. Report each draw's recovery and the wording effect minus the average draw, with secondary pointwise 95% crossed-bootstrap intervals using the existing routine. Do not replace the original controls or include these post hoc comparisons in the 24-quantity primary family. Re-execute unedited corrupted and wording behavior to compare with the saved primary outcomes. This repeats behavior inference only, not all downstream signatures.
