# EXPERIMENT-004 feasibility: Pythia cross-architecture behavior and hooks

**Status:** frozen before Pythia IOI behavior evaluation  
**Model:** EleutherAI Pythia-410M-deduped, revision
`c4fc8d586d62df497f1f9b69d66d3ca419992d3e`  
**Purpose:** determine whether a comparably sized GPT-NeoX-family model supports a valid
cross-architecture circuit-transport replication

## Why this gate comes first

Raw activation transfer across architectures is not meaningful because the representation spaces
and components differ. A replication must independently localize Pythia components and compare the
qualitative transport claim. That work is interpretable only if Pythia itself solves both canonical
and controlled conditions. This feasibility phase therefore evaluates behavior and hook fidelity
before any circuit outcomes are computed.

## Frozen data

- Reuse all 320 texts and 16 development/test template families from Experiment 003.
- Retokenize from text with the pinned Pythia tokenizer; never reuse GPT-2 token IDs.
- All entity names must remain single leading-space tokens. Role positions are recovered separately
  for canonical and controlled prompts and recorded explicitly.
- Pythia sequence lengths may be 14 or 15 tokens. Evaluation batches are template-specific, so no
  padding or post-outcome template filtering is used.
- Primary corruption only. No example is filtered by correctness.

## Frozen model path

TransformerLens 3.8's v3 `TransformerBridge` does not currently construct this GPT-NeoX checkpoint
under Transformers 5.16 because the upstream output module was renamed from `embed_out` to
`lm_head`. We use TransformerLens' supported unprocessed `HookedTransformer` Pythia converter and
provide an in-memory `embed_out = lm_head` alias. The alias references the identical weight tensor;
it does not modify installed packages or model values.

Before behavior is accepted, the converted model must:

1. expose 24 attention-result hooks with shape `[batch, position, 16, 1024]`;
2. match raw Hugging Face logits on a frozen prompt with maximum absolute error at most `0.005`;
3. use float32 and no layer-norm folding, weight centering, BOS insertion, or other preprocessing.

## Behavioral advance gate

Advance to independently localized circuit interventions only if all rules pass:

1. aggregate canonical and controlled clean pairwise IO accuracy are each at least 70% and their
   mean `P(IO)-P(subject)` values are positive on both development and test;
2. at least six of eight controlled templates pass the same accuracy/probability gate in each split;
3. the hook/logit fidelity checks above pass.

If the gate fails, report the result as a size-matched architecture-family behavioral boundary. Do
not change the model size, templates, threshold, or examples in response. A larger architecture
would require a separately frozen follow-up, not a substitution inside this experiment.
