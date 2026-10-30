# Calibration sweep (run_id 3237f6fb)

All 90 calibration items, temperature 0, one run. Compute is an ESTIMATE (2 × active params × API tokens). Latency of cached calls is the original call's latency.

### math

| configuration | accuracy | compute / success (TFLOPs, est.) | avg latency (s) | format failures | truncated | think-lang followed | avg in / out tokens |
|---|---|---|---|---|---|---|---|
| allam | 8/30 (27%) | 20.91 | 1.09 | 9 | 0 | 8/30 | 239 / 159 |
| allam-noex | 7/30 (23%) | 20.31 | 0.89 | 7 | 0 | 0/30 | 186 / 152 |
| 20b-low | 27/30 (90%) | 2.54 | 1.14 | 0 | 0 | 30/30 | 222 / 95 |
| 20b-medium | 26/30 (87%) | 5.35 | 1.47 | 1 | 0 | 30/30 | 222 / 421 |
| 20b-high | 21/30 (70%) | 13.86 | 2.82 | 9 | 9 | 30/30 | 222 / 1125 |
| 20b-medium-ar | 27/30 (90%) | 4.85 | 1.07 | 1 | 1 | 0/30 | 225 / 381 |
| 120b-low | 28/30 (93%) | 3.56 | 1.09 | 0 | 0 | 30/30 | 222 / 104 |
| 120b-medium | 28/30 (93%) | 5.29 | 1.33 | 1 | 0 | 30/30 | 222 / 262 |
| 120b-high | 27/30 (90%) | 8.01 | 2.04 | 1 | 0 | 30/30 | 222 / 484 |

### knowledge

| configuration | accuracy | compute / success (TFLOPs, est.) | avg latency (s) | format failures | truncated | think-lang followed | avg in / out tokens |
|---|---|---|---|---|---|---|---|
| allam | 16/30 (53%) | 8.15 | 0.95 | 1 | 0 | 22/30 | 232 / 79 |
| allam-noex | 2/30 (7%) | 61.22 | 0.73 | 27 | 0 | 18/30 | 188 / 104 |
| 20b-low | 20/30 (67%) | 3.77 | 1.02 | 0 | 0 | 30/30 | 229 / 120 |
| 20b-medium | 22/30 (73%) | 6.14 | 1.62 | 1 | 1 | 30/30 | 229 / 396 |
| 20b-high | 18/30 (60%) | 14.80 | 1.59 | 8 | 8 | 30/30 | 229 / 1004 |
| 20b-medium-ar | 1/2 (50%) | 16.10 | 0.92 | 0 | 0 | 0/2 | 697 / 421 |
| 120b-low | 19/30 (63%) | 5.00 | 1.24 | 0 | 0 | 30/30 | 229 / 81 |
| 120b-medium | 20/30 (67%) | 7.47 | 1.34 | 0 | 0 | 30/30 | 229 / 259 |
| 120b-high | 18/30 (60%) | 17.07 | 2.74 | 5 | 5 | 30/30 | 229 / 775 |

### intent

| configuration | accuracy | compute / success (TFLOPs, est.) | avg latency (s) | format failures | truncated | think-lang followed | avg in / out tokens |
|---|---|---|---|---|---|---|---|
| allam | 27/30 (90%) | 4.78 | 0.69 | 1 | 0 | 30/30 | 262 / 45 |
| allam-noex | 25/30 (83%) | 4.53 | 0.74 | 2 | 0 | 19/30 | 218 / 52 |
| 20b-low | 28/30 (93%) | 2.13 | 1.17 | 0 | 0 | 30/30 | 226 / 50 |
| 20b-medium | 30/30 (100%) | 3.92 | 1.25 | 0 | 0 | 30/30 | 226 / 317 |
| 20b-high | 28/30 (93%) | 6.26 | 1.28 | 2 | 2 | 30/30 | 226 / 586 |
| 120b-low | 28/30 (93%) | 3.05 | 0.97 | 0 | 0 | 30/30 | 226 / 52 |
| 120b-medium | 29/30 (97%) | 4.01 | 1.10 | 0 | 0 | 30/30 | 226 / 153 |
| 120b-high | 30/30 (100%) | 5.13 | 1.00 | 0 | 0 | 30/30 | 226 / 276 |

**INCOMPLETE** — stopped by daily limits:
- 20b-medium-ar: daily limit after 32/90 — litellm.RateLimitError: RateLimitError: GroqException - {"error":{"message":"Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01kykk7zpsek284fxx0aynse8m` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199429, Requested 726. Please try again in 1m6.95999999
