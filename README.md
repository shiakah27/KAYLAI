# KAYL (كَيْل)

**KAYL — Intelligence Without Waste. The right measure of AI for every Arabic task.**

An agent that completes Arabic AI tasks with the fewest resources while keeping accuracy.

## Results in short
Held-out test: 90 Arabic questions (30 math, 30 knowledge, 30 intent), one run at temperature 0.

| | Correct | Est. compute per correct answer |
|---|---|---|
| **KAYL** | 79/90 | **2.56 TFLOPs** (49% less) |
| Everything to the largest model (gpt-oss-120b, medium) | 83/90 | 5.02 TFLOPs |

- KAYL used about half the compute of the largest model, and got 4 fewer answers right.
- KAYL met its own accuracy floor on knowledge, but missed math by 2 questions and intent by 1. 30 calibration questions per type is a small sample.
- KAYL's choices were made on calibration data only and were not changed after the test.

Full tables, ablations, robustness scenarios and limitations are in [RESULTS.md](RESULTS.md).

## The problem
Sending every request to the largest model gives good answers but wastes compute.
Many Arabic tasks can be solved by a smaller model, or with less thinking.

## What KAYL does
For each Arabic task, KAYL picks three "keys":

1. **Which model**: small or large open-weight models on Groq
   (`allam-2-7b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`).
2. **How much the model thinks**: reasoning effort `low` / `medium` / `high`
   (only on the gpt-oss models).
3. **Which language it thinks in**: Arabic or English reasoning. The answer is always in Arabic.
   (gpt-oss reasoned in English even when asked for Arabic, so this key had no effect; see RESULTS.md.)

Then it **verifies** the answer and **escalates** (more effort or a bigger model) only when the check fails.
If a model hits its limit or is unavailable, it **falls back** to the next-best model.
Every decision is stored with a short human-readable reason, and the demo UI shows it.
KAYL does not train any model. It only uses controls that already exist.

## Core metric
**Primary: estimated compute per successful task** = total estimated compute of all calls (including failed attempts and escalations) / number of correct answers.

- Compute per call ≈ 2 × active_params × (input + output tokens). This is an **estimate**, using the token counts returned by the API. Energy is not reported.
- Compute is the primary metric because `allam-2-7b` has no published price, so dollars can't compare all three models.
- **Secondary:** dollar cost per successful task, reported only when every call used a model with a verified price.
- A configuration only counts if it meets the **accuracy floor** and **latency floor** in [config.yaml](config.yaml).
  The accuracy floor per task type is baseline A's calibration accuracy minus one question; the latency floor is p95 ≤ 15 s.
- Selection rule: among configurations predicted to meet both floors, pick the one with the lowest compute per successful task, and break ties by latency.

## Data
There are three Arabic task types. Each has 30 calibration and 30 held-out test questions, split with seed 42.

| Type | Source | Graded by |
|---|---|---|
| `math` | [Arabic-GSM8K](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic-gsm8k) (MIT) | The final number |
| `knowledge` | [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) (CC-BY-NC-4.0), all 21 subjects | The option letter |
| `intent` | **Synthetic** chatbot messages in MSA and Saudi dialect: 8 intents plus `other`, reviewed by hand | The intent label |

- `data/calibration.jsonl` (90 questions): the only data KAYL learns from.
- `data/test.jsonl` (90 questions): held out and used only for the final evaluation.
- `data/robustness_en.jsonl` (20 questions): English GSM8K, used for testing only.
- Full details, including licenses, how the files are built and limitations, are in [DATA.md](DATA.md).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results (no API key needed)
Every model response is committed in `cache/llm/`, so everything replays from the cache with no API calls.
`--from-cache` stops at the first response that is not recorded, and no script overwrites `results/` if any call failed.
Run from the repository root:

```bash
pytest                                                          # tests, including a replay of recorded test answers
python -m scripts.prepare_data                                  # rebuilds data/ exactly (no model calls)
python -m scripts.run_sweep --from-cache --route groq           # calibration sweep (it used Groq directly)
python -m scripts.run_final --final-eval --from-cache           # final test (OpenRouter pinned to Groq)
python -m scripts.run_fallback_drill --from-cache --route groq  # real-condition fallback drill
python -m scripts.run_simulated_outage --from-cache             # SIMULATED outage
streamlit run app.py                                            # demo UI; Replay mode plays recorded runs
```

`prepare_data` and the two drills download the public datasets from Hugging Face on first use.
A replay gives the same answers, tokens, compute and grades; only `run_id`, the `cached` flags and the "(from cache)" note in decision reasons differ.

To make new calls instead, copy `.env.example` to `.env`, add your `GROQ_API_KEY` (and `OPENROUTER_API_KEY` for the final-test route), and drop `--from-cache`.
`scripts/generate_intents.py` regenerates the synthetic intent data; its output is already committed, so you do not need to run it.

## Experiments
All run once on the same test set at temperature 0. A second run was not possible because Groq's daily token limit was used up.

| ID | Strategy |
|----|----------|
| A | Everything goes to the largest model (gpt-oss-120b, medium effort): baseline A |
| B | The model with the lowest token count (allam-2-7b) |
| C | The lowest verified-price model (gpt-oss-20b, medium effort) |
| E | A simple difficulty proxy (question length): long questions to 120b, short ones to 20b |
| D | KAYL: all three keys plus verify/escalate |
| Ablations | D minus each key, to show what each key adds |

## Repository layout
| Path | What it holds |
|---|---|
| `kayl/` | The agent and its parts: client, cache, prompts, parsing, grading, metrics, profile, SQLite log |
| `scripts/` | Data build, intent generation, calibration sweep, final test, robustness runs |
| `tests/` | pytest tests (no API calls) |
| `data/` | The data splits and the synthetic intent source and review |
| `results/` | Every reported result (the local SQLite call log is not committed) |
| `cache/llm/` | Every raw model response, keyed by model + params + prompt hash |
| `docs/screenshots/` | Demo UI screenshots |
| `app.py` | Streamlit demo UI |
| `config.yaml` | Models, prices, floors, routing and limits: the single source of truth |

## Configuration
Models, prices, floors and generation limits all live in [config.yaml](config.yaml).
The gpt-oss prices were checked against each model's Groq console page, and those links are in the config.
Groq lists the allam-2-7b price as "Pending", so dollar cost is n/a for any experiment that uses it.
(allam-2-7b replaced llama-3.1-8b-instant, which is not available on our Groq account.)
The accuracy floors are set from the baseline A results on calibration data.
