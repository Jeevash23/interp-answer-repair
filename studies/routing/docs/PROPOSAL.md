# What Selects a Circuit?

Date: 2026-09-05. Status: feasibility study, with an incomplete fresh literature audit.

## Research question

In a pretrained transformer performing indirect object identification (IOI), can an internal
state transferred between matched prompts change the recipient's downstream causal dependencies
toward those of the donor's prompt family?

This is the Paper A project from the earlier task “Locate existing research environment”
(`01a065e0-a3fa-73b3-b92e-caf7a6c8d7b1`). That discussion ranked training-time handoffs as the
safer project and this project as the higher-risk option. The current user request selects this
project. Folder 02 is reserved; this project is `routing-project`.

## Verdict and scope

The question warrants a bounded GPT-2 Small pilot on this machine. It is not yet a verified novel
contribution or a promised paper. The distinguishing target is a change in a frozen downstream
causal signature, beyond an answer change or a probe that recognizes word order.

The first stage measures ABBA/BABA signatures. It does not fit routers, run a full factorial
dataset, perform model training, or download larger models. The original project is read-only.

The abstract-level contribution remains a hypothesis:

> Prompt structure predicts recurring downstream causal dependencies on held-out inputs, and
> transferring a candidate internal selection state causes those dependencies to shift toward
> the donor family's signature while preserving task competence and controlling answer identity.

No result in this project presently supports the second part of that statement. Signature
separation alone is not evidence for discrete circuits or a localized router.

## What must be demonstrated

1. Controlled changes to prompt structure systematically alter causal signatures.
2. Irrelevant wording, name identity, and formatting do not explain the effect.
3. A discovery-only candidate state can be edited with an effect beyond generic damage.
4. The recipient's *recomputed* downstream signature moves toward the donor family's signature.
5. Effects survive unseen templates, names, answer tokens, and intervention baselines.

The signature is a finite collection of single-node effects. Even complete success would establish
switching in those measured dependencies, not equivalence of every edge or a universal router.
Joint ablations and an independent pathway/sufficiency check are needed for stronger circuit claims.

## Relation to the earlier paper

The completed project measures activation transport, endpoint recovery, normalization, and
specificity. Its current audit explicitly says that local projections do not establish a shared
pathway. This project cannot simply relabel its endpoint transport as circuit switching. It uses
new manifests and freezes a downstream dependency measurement before testing any routing edit.

Names and familiar IOI syntax may have appeared in the earlier research. “Held out” means held out
from fitting and selection in this project's pilot, not globally unseen by the researcher or the
pretrained model. Confirmatory routing experiments need a further unopened test set.

## Resource envelope

The local inspection found an Intel Core 5 210H, 12 logical CPUs, about 15 GiB RAM, about 9 GiB
available RAM, and 89 GiB free storage. The prior project records a 6 GB NVIDIA GPU with
6 GiB VRAM. GPU access could not be verified here: sandbox access failed and automatic approval
review failed before the outside-sandbox check ran. The initial pilot therefore uses CPU,
float32, four compute threads, and small batches. GPT-2 Small weights are already cached.

One isolated virtual environment and dependency lock serve the entire new project. A copy-on-write
snapshot of the package cache permits offline installation without changing the old environment or
the shared cache. Models are read from the existing cache through a recorded path, never edited.

## Publication boundary

Do not claim novelty until the primary papers and newer literature are accessible and reviewed.
Do not claim a transferable routing state until the causal controls pass. A failed compact edit
does not prove that routing is distributed; it only rules out the tested interventions at their
measured power. A negative paper requires explicit coverage and sensitivity analysis.

