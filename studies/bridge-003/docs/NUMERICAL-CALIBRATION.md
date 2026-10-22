# Pythia numerical calibration before behavioral screening

8 September 2026. No Pythia behavioral-screen, discovery or test results preceded
this calibration. The original prompt design and scientific questions remain frozen.

The historical unprocessed TransformerLens conversion used mathematically
equivalent but differently ordered float32 operations. Against the raw HF model,
initial maximum last-token logit discrepancies on two screening prompts were
0.000855 (CPU/GPU), 0.001031 (GPU/GPU), and 0.001078 with raw eager attention.
Layerwise checks localized accumulating rounding differences, including large
residual coordinates. The new adapter preserves every weight but matches native
layer normalization, fused QKV projection, NeoX attention, contiguous linear
weights and residual addition order. The aligned CPU comparison reached exact
agreement. The strict raw-model comparison remains 0.0005 on GPU.

After raw-model agreement passed, full-versus-suffix checks were exact, but a
separate precision issue remained: changing batch size for vectorized ablations
changes GPU matrix multiplication rounding. On the calibration prompts the
largest expanded-versus-singleton head-effect discrepancy was 0.0009165; the
equivalent head-result versus summed-output insertion differed by 0.0005989.
These checks exceeded the inherited GPT-2 tolerance of 0.0005.

Before any Pythia scientific outcomes, set its tolerance for these numerical
intervention equivalence checks to **0.002 logits**, retain every observed error,
and additionally validate the differently structured F6 screening formulation.
Raw-model logit agreement still uses 0.0005. Medium's tolerance is unchanged.
This is a disclosed precision calibration, not a passed check under the original
tolerance. The adapter and this note are frozen before behavioral screening.
The precision limit is 2% of the 0.1-logit signature scale floor; near-threshold
mechanistic claims require corresponding caution. No claim depends on treating
sub-0.002 changes as exactly reproducible across different batch sizes.

The custom adapter supports the study's unpadded, no-cache inference and q/k/v/z
and result hooks. It does not provide a general replacement for TransformerLens's
training, normalization-hook, split-QKV-input or padded-generation interfaces.
