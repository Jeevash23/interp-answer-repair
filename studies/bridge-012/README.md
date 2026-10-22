# Bridge 012 (Study 7): AI-01 in GPT-2 Large on Modal

**Completed: the frozen rule stopped the study at pair selection (17 October 2026).** The protocol and predictions were hashed after the author's approval (03:53:15 UTC). Code, banks and inherited inputs were frozen at 04:35:00 UTC, and the stages ran on a Modal L4 from 04:36 to 04:51 UTC.

**Outcome.**
- Calibration passed on GPT-2 Large.
- Only F10 and F13 passed the behavioural screen. F11 and F12 failed on top-1 accuracy (54–79% against 85%), although pairwise accuracy was 100%.
- Twelve source heads were selected on development data; together they recover 34% of the clean gap.
- The one eligible pair, F10/F13, then failed Study 2's selection criteria: F13's top-1 accuracy on the selection bank was 84.4% (gate 85%), and the worst classification cell was 86.7%/88.3% (gate 90%).
- No pair qualified, so no qualification, positive-control development or final transfer was measured.

As with Study 4 in GPT-2 Small, this is a bounded stop: AI-01's qualified results remain those of one checkpoint.

- [RESULTS.md](RESULTS.md): every gate and value, predictions against outcomes, costs and disclosures.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): how to rerun and verify.
- [docs/PROTOCOL.md](docs/PROTOCOL.md), [docs/PREDICTIONS.md](docs/PREDICTIONS.md) and [docs/AUTHOR-APPROVAL.txt](docs/AUTHOR-APPROVAL.txt): the frozen design and the approval.
- [provenance/modal-smoke-test/](provenance/modal-smoke-test/README.md): the Modal smoke test, which reproduced Study 6 within 2.5 × 10⁻⁵ logits.

**Cost:** $0.30 for the study run, $0.48 for the code tests and $0.10 for the smoke test, by Modal's billing report; $0.00 billed after credits.

| Path | Contents |
|---|---|
| `code/` | `prepare012.py` (banks), `common012.py` (engine), `stages012.py` (stages), `analyze012.py`, `audit012.py`, `modal_app012.py` (Modal app), `freeze012.py` |
| `data/` | Banks (`rows.json`, `design.json`), protocol and input freezes, bootstrap indices |
| `results/` | Records pulled back from the Modal Volume `ai01-study7`: stage records, freezes, runner records with chunk hashes, the source scan and the fitted bank. The larger arrays stay on the Volume |
| `provenance/` | Stage and run logs, code-test logs, billing record, smoke test |
