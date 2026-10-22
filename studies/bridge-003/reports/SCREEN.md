# Eight-family behavior screen

32 cases per family/model, 16 per order. A pass requires at least 95% pairwise and 85% vocabulary top-1 in both orders. These are observed-accuracy gates, not lower confidence bounds.

| Model | Family | Pairwise ABBA / BABA | Top-1 ABBA / BABA | Gate | Retained |
| --- | --- | --- | --- | --- | --- |
| gpt2-medium | F0 | 100.00% / 100.00% | 100.00% / 100.00% | pass | yes |
| gpt2-medium | F1 | 100.00% / 100.00% | 87.50% / 100.00% | pass | yes |
| gpt2-medium | F2 | 100.00% / 100.00% | 93.75% / 100.00% | pass | yes |
| gpt2-medium | F3 | 93.75% / 100.00% | 87.50% / 100.00% | fail | no |
| gpt2-medium | F4 | 100.00% / 100.00% | 100.00% / 100.00% | pass | yes |
| gpt2-medium | F5 | 100.00% / 100.00% | 37.50% / 31.25% | fail | no |
| gpt2-medium | F6 | 100.00% / 100.00% | 100.00% / 100.00% | pass | no |
| gpt2-medium | F7 | 100.00% / 100.00% | 87.50% / 93.75% | pass | no |
| pythia-410m-deduped | F0 | 100.00% / 100.00% | 75.00% / 25.00% | fail | yes |
| pythia-410m-deduped | F1 | 100.00% / 93.75% | 43.75% / 43.75% | fail | yes |
| pythia-410m-deduped | F2 | 100.00% / 100.00% | 50.00% / 43.75% | fail | no |
| pythia-410m-deduped | F3 | 100.00% / 93.75% | 75.00% / 75.00% | fail | no |
| pythia-410m-deduped | F4 | 100.00% / 100.00% | 93.75% / 81.25% | fail | no |
| pythia-410m-deduped | F5 | 100.00% / 81.25% | 0.00% / 0.00% | fail | no |
| pythia-410m-deduped | F6 | 100.00% / 93.75% | 75.00% / 62.50% | fail | no |
| pythia-410m-deduped | F7 | 100.00% / 93.75% | 12.50% / 12.50% | fail | no |

Medium has six passing candidates; the fixed first-four rule retains F0/F1/F2/F4. F6/F7 were scored but not used in discovery. Pythia has no passing candidate; it retains the predetermined F0/F1 diagnostic pair. F4's 81.25% BABA top-1 is below the 85% requirement; the threshold was not relaxed.

![Screen results](screen-accuracy.png)
