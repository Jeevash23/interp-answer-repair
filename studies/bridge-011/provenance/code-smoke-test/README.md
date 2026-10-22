# Code smoke test before the input freeze (16 October 2026)

`smoke.py` and `smoke2.py` ran the Study 6 calibration, development, held-out, replay, analysis and audit code on Study 2's 32 F10/F11 calibration rows, which earlier served only technical checks. Outputs went to a scratch directory; only `smoke-summary.json` is kept. The test found no bug after one change: calibration's full-sequence checks were batched and the run settings set to 8 prompts and 8 heads per pass (Studies 2–5's settings), because another project's job was using 2.8 GB of the GPU.

The run printed the development choices (held heads L16H15, L15H14, L20H12; κ = 0.85) and primary decisions for those 32 rows (N1 not detected, N2 below AI-01's resolution, O detected). They were seen before the Study 6 input freeze. The protocol, thresholds, conditions, sample sizes and analysis had already been frozen and approved, and nothing was changed after seeing them. They are not evidence: 32 rows from a 2 × 2 grid of an earlier study.
To rerun, copy the two scripts into an empty scratch directory first: they write next to themselves.
