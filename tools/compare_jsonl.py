"""Compare a rerun of analysis/independent_check.py with the saved record.

    python tools/compare_jsonl.py analysis/results/independent-check.jsonl analysis/results/independent-check-rerun.jsonl

The check re-implements GPT-2 Medium in plain NumPy and reports, per configuration and condition, its difference
from the saved behaviour (beh_err) and signature coordinates (sig_err_logit, sig_err_prob) of the original
pipeline. A rerun passes when every configuration is present, every top-1 prediction matches, every difference is
within the bound the manuscript states (4.6e-5 logits; largest saved difference 4.52e-5), and every logit gap agrees
with the saved record within 1e-5 logits. Float32 sums depend on the number of BLAS threads, so the differences
themselves can change between runs in the seventh decimal place.
"""
from __future__ import annotations

import json
import sys

BOUND, GAP_TOLERANCE = 4.6e-5, 1e-5


def main():
    saved = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1]) if l.strip()}
    rerun = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2]) if l.strip()}
    problems = [f"missing {k}" for k in saved if k not in rerun]
    worst_err, worst_gap = 0.0, 0.0
    for k, row in rerun.items():
        for cond, v in row.items():
            if not isinstance(v, dict):
                continue
            s = saved[k][cond]
            if v.get("top1_match") is not True:
                problems.append(f"{k} {cond}: top-1 differs from the original pipeline")
            for e in ("beh_err", "sig_err_logit", "sig_err_prob"):
                worst_err = max(worst_err, v[e])
                if v[e] >= BOUND:
                    problems.append(f"{k} {cond} {e} = {v[e]:.2e} (bound {BOUND})")
            gap = abs(v["logit_gap"] - s["logit_gap"])
            worst_gap = max(worst_gap, gap)
            if gap > GAP_TOLERANCE:
                problems.append(f"{k} {cond} logit gap differs from the saved record by {gap:.2e}")
    print(f"{len(rerun)} configurations: largest difference from the original pipeline {worst_err:.2e} logits "
          f"(bound {BOUND}); largest logit-gap difference from the saved record {worst_gap:.2e}")
    for p in problems[:20]:
        print("PROBLEM", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
