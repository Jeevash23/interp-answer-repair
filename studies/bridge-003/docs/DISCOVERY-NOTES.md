# Scope of selection diagnostics

Written before broader-arm discovery inference. The leave-one-template-block-out
selection diagnostic fits centroids and scales without that block. Both replacement
references are shared across candidates: the mean bank is computed over the
retained discovery families, including the held-out selection block. This is an
internal selection diagnostic with a common reference, not an independent estimate
of generalization. Final test prompts are completely separate; its mean bank,
centroids and scales are fixed from discovery only. Test performance, not the
selection diagnostic alone, must pass all behavior and separation criteria.

Candidate selection can differ between architectures if no common pair qualifies.
In that case compare the qualitative result and report the different texts; do
not call different selected pairs an exactly matched cross-architecture replication.
The unconditional F0/F1 fallback retains failure to find a qualified pair as a
scientific result. Its intervention results cannot establish a valid distinct-task
transfer if those validity/separation gates fail.
