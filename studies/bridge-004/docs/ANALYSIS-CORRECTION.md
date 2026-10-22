# Analysis implementation correction

8 September 2026, after fresh Medium raw inference and before a successful primary geometry analysis.

A diagnostic call exposed NumPy advanced-index ordering in the original `analyze004.py`: selecting the donor example array together with the final scalar endpoint index produced shape `[example, reference, head]` instead of `[reference, example, head]`. The paired geometry function raised a broadcasting error. No valid primary paired-margin result was produced by that path.

The preserved original remains covered by the input freeze. `analyze004_corrected.py` first selects the logit endpoint and then reorders the example axis. A direct, explicit-loop comparison checks each matched donor and every reference/head coordinate. The corrected implementation and this note are separately frozen before running the corrected analysis. All earlier source hashes remain valid. No data, pairing definition, measured outcome, geometric formula, qualification gate, multiplicity, bootstrap seed or test threshold changes.

The initial run of the original analysis is retained as a failed technical analysis attempt if invoked by the existing sequential runner. Numerical inference and the raw experiment do not require repetition for this indexing correction. The manuscript reports this correction separately from the historical scientific failure of the strict switch gate.
