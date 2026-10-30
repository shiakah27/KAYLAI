# KAYL — Results

**KAYL — Intelligence Without Waste. The right measure of AI for every Arabic task.**

Every number below was measured and comes from the files in `results/`. Nothing was estimated or tuned on the test set.

## Setup

| Item | Value |
|---|---|
| Held-out test set | `data/test.jsonl`: 90 items (30 math, 30 knowledge, 30 intent), never used for tuning |
| Calibration set | `data/calibration.jsonl`: 90 items, the only data KAYL learned from |
| Runs | **One run at temperature 0.** A second run was not possible because the models' daily token limit was used up. |
| Primary metric | **Estimated compute per successful task** = Σ 2 × active params × (input + output tokens) over *all* calls, including failures and escalations, ÷ number correct. This is an **estimate**, based on the token counts the API returned. |
| Secondary metric | Dollar cost per successful task, using Groq prices from `config.yaml`. Shown as n/a when any call used `allam-2-7b`, whose price is unknown. |
| Floors | **Accuracy floor per type** = baseline A's calibration accuracy minus one question (3.3 points at n=30): math ≥ 27/30, knowledge ≥ 19/30, intent ≥ 28/30. **Latency floor:** p95 ≤ 15 s per task. |
| Models | `allam-2-7b` (7B dense), `gpt-oss-20b` (3.6B active), `gpt-oss-120b` (5.1B active) |
| How calls were sent | Calibration: directly to Groq. **Final test: the gpt-oss calls went through OpenRouter, pinned to the Groq provider** (`only: [groq], allow_fallbacks: false`), because Groq's per-organization daily token limit had been used up. All 634 gpt-oss test attempts reported `provider: Groq`. ALLaM (90 calls) stayed on Groq directly. |

## Final held-out test (n = 90, one run)

| Experiment | Correct | Compute / success (TFLOPs, est.) | Cost / success | Avg latency (s) | p95 (s) | Types meeting the floors | Escalations |
|---|---|---|---|---|---|---|---|
| **D KAYL** | 79/90 (88%) | **2.56** | **$0.000050** | 0.46 | 0.79 | 1/3 (knowledge) | 0 |
| A 120b-medium (largest, baseline A) | **83/90 (92%)** | 5.02 | $0.000188 | 0.76 | 1.42 | 3/3 | 0 |
| B allam (fewest tokens) | 62/90 (69%) | 6.83 | n/a | 0.22 | 0.40 | 2/3 | 0 |
| C 20b-medium (cheapest price) | 77/90 (86%) | 5.65 | $0.000178 | 0.87 | 2.56 | 2/3 | 0 |
| E simple difficulty proxy (question length) | 82/90 (91%) | 5.35 | $0.000189 | 0.85 | 1.60 | 3/3 | 0 |
| *Ablation:* D minus verify/escalate | 79/90 | 2.56 | $0.000050 | 0.46 | 0.79 | 1/3 | — |
| *Ablation:* D minus model key (120b only) | 80/90 (89%) | 3.47 | $0.000093 | 0.45 | 0.73 | **3/3** | 0 |
| *Ablation:* D minus effort key (medium only) | 83/90 (92%) | 5.52 | $0.000189 | 0.90 | 1.98 | 3/3 | 4 |
| *Ablation:* D minus thinking-language key | identical to D, not run | — | — | — | — | — | — |

E routes questions longer than 78 tokens (the calibration median, counted by LiteLLM's token counter) to 120b-medium and shorter ones to 20b-medium. It is a simple length proxy, **not** RouteLLM.

### Per task type

| Experiment | math | knowledge | intent |
|---|---|---|---|
| Floor (baseline A on calibration − 1) | ≥ 27/30 | ≥ 19/30 | ≥ 28/30 |
| D KAYL (20b-low for every type) | 25/30 ✗ · 2.76 | 27/30 ✓ · 2.68 | 27/30 ✗ · 2.24 |
| A 120b-medium | 29/30 ✓ · 5.15 | 25/30 ✓ · 5.99 | 29/30 ✓ · 4.04 |
| B allam | 12/30 ✗ · 13.95 | 22/30 ✓ · 5.76 | 28/30 ✓ · 4.62 |
| C 20b-medium | 26/30 ✗ · 5.53 | 23/30 ✓ · 6.46 | 28/30 ✓ · 5.09 |
| E length proxy | 29/30 ✓ · 5.11 | 25/30 ✓ · 5.91 | 28/30 ✓ · 5.09 |
| D minus model key (120b-low) | 27/30 ✓ · 3.68 | 24/30 ✓ · 3.91 | 29/30 ✓ · 2.89 |
| D minus effort key | 29/30 ✓ · 5.15 | 24/30 ✓ · 6.59 | 30/30 ✓ · 5.01 |

Each cell is correct/30 · compute per success in TFLOPs (estimated).

### What this shows

1. **KAYL used the least compute of all experiments.** It needed 2.56 TFLOPs per success, which is **49% less than A** (5.02). It was also the cheapest in dollars, at 27% of A's cost per success.
2. **But KAYL lost 4 questions compared with A** (79 vs 83), and **it did not meet its own accuracy floor on math (25 vs 27) or intent (27 vs 28).** Its calibration choice (`20b-low` for every type) scored 27/30 on math and 28/30 on intent during calibration, but 25/30 and 27/30 on test. These are drops of 1–2 questions, which is within noise at n=30. By KAYL's own rule, though, it does **not** qualify on those two types.
3. **The best configuration on test that met every floor was the "minus model key" ablation** (`120b-low`). It got 80/90, met all 3 floors, and used 3.47 TFLOPs per success (31% less than A). With this data, choosing 120b at low effort beat choosing the smaller model. The model key saved compute, but cost accuracy.
4. **Verify/escalate added nothing on this test.** All 90 KAYL answers passed verification, so there were 0 escalations, and **11 of the verified answers were wrong.** The verifier checks the answer *format* (valid number, letter or label, Arabic ANSWER line, not cut off), not whether the answer is correct. That is why "D minus verify/escalate" is identical to D.
5. **The effort key matters.** Removing it (medium only) raised accuracy to 83/90 but more than doubled compute (5.52).
6. **B (ALLaM) is the fastest but not "small" in compute.** It is a 7B dense model, so it costs more compute per token than gpt-oss-20b (3.6B active). Its math accuracy is very low (12/30).

## Calibration sweep (n = 30 per type, one run, Groq direct)

The full tables are in [results/calibration_sweep.md](results/calibration_sweep.md). Overall:

| Configuration | Correct /90 | Note |
|---|---|---|
| allam | 51 | with the format example |
| allam-noex | 34 | without the format example (measurement only) |
| 20b-low | 75 | |
| 20b-medium | 78 | |
| 20b-high | 67 | **TRUNCATED: max 2,048 output tokens; 19/90 replies were cut off before FINAL.** Not a fair high-effort result. Not rerun, because gpt-oss-20b's daily quota was used up. |
| 120b-low | 75 | |
| 120b-medium | 77 | **baseline A** |
| 120b-high | 77 | rerun with max 8,192 output tokens: 0 cut off (5/90 were cut off at 2,048) |

**Format example for ALLaM.** One short example in the system prompt raised ALLaM from 34/90 to 51/90 (knowledge 2/30 → 16/30). It cost about 44–53 extra input tokens per call.

## Negative result: "think in Arabic" does not control gpt-oss

We asked gpt-oss-20b (medium effort) to do its reasoning in Arabic. **It reasoned in English in 32 of 32 calls**: all 30 math items, plus 2 knowledge items before the daily limit stopped the run.
- We checked this on the reasoning text itself, after removing text copied from the question. It is not a detector error.
- Math accuracy was 27/30 with the Arabic instruction vs 26/30 with English, which is a one-question difference and not meaningful.

We stopped this configuration. As a result, **the thinking-language key had no effect on gpt-oss in our setup**, and the corresponding ablation is identical to D. The 32 rows are kept in `results/think_lang_ar_stopped.jsonl`.

## Robustness scenarios

| Scenario | Status | n | Result | Limitations |
|---|---|---|---|---|
| Different task types (math, knowledge, intent) | done | 30 each | See the per-type table. KAYL meets its floor only on knowledge. | One dataset per type |
| Dialect: MSA vs Saudi (intent) | done (from saved results) | 15 + 15 | KAYL: MSA 13/15, Saudi 14/15. A: 15/15, 14/15. B (ALLaM): 14/15, 14/15. | Synthetic messages. Dialect versions of math were not built. |
| Noisy input (typos, Arabic-Indic digits) | **not run** | — | — | — |
| Model outage with fallback: real | run, **not exercised** | 5 | 3/5 correct. No limit occurred, because gpt-oss-20b's quota had refilled, so the fallback never triggered. | `results/robustness_fallback_drill.json` |
| Model outage with fallback: **simulated** | **not run yet** | — | Only covered by unit tests with a fake model (3 tests). | — |
| Routing change (Groq direct → OpenRouter pinned to Groq) | happened for real | 634 calls | All replies came from Groq. Calibration and test latencies are not directly comparable (test latencies were lower). | Different network path |
| Price change | **not run** | — | — | Can be recomputed offline from logged tokens |
| Stricter accuracy or latency floors | **not run** | — | — | Can be recomputed offline from saved results |
| English input (`robustness_en.jsonl`) | **not run** (budget) | 20 | — | — |

## Limitations

- **Small samples.** n = 30 per type, so one question equals 3.3 points. Most differences between experiments are 1–4 questions.
- **Single run** at temperature 0. We did not measure run-to-run variance.
- **The verifier checks the answer format, not its correctness**, so it cannot catch a well-formed wrong answer. This is why escalation never triggered on the test set.
- **Compute is an estimate**, based on active parameters and API token counts. It is not a hardware measurement, and energy is not reported.
- **ALLaM's price is unknown**, so dollar cost is n/a for any experiment that used it.
- **The intent data is synthetic**, generated by qwen3.8-27b. **0 of 80 rows have been reviewed by a human yet.**
- **Data quality.** Some ArabicMMLU items contain corrupted text, and the models may have seen GSM8K and ArabicMMLU during training.
- **Different routes.** The final test went through OpenRouter (pinned to Groq), while calibration went to Groq directly, so latencies from the two phases should not be compared.

## Reproduce

```bash
python prepare_data.py                 # data splits (no model calls)
python run_sweep.py                    # calibration sweep (from cache; set routing.gpt_oss: groq in config.yaml,
                                       # because the calibration calls were cached on the direct Groq route)
python run_final.py --final-eval       # final test (from cache; routing.gpt_oss: openrouter)
pytest                                 # tests
```
