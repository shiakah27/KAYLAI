# KAYL (كَيْل)

**KAYL — Intelligence Without Waste. The right measure of AI for every Arabic task.**

An agent that completes Arabic AI tasks with the fewest resources while keeping accuracy.

> **Status:** project skeleton only. No agent code and no results yet.

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

Then it **verifies** the answer and **escalates** (more effort or a bigger model) only when the check fails.
KAYL does not train any model. It only uses controls that already exist.

## Core metric
**Primary: estimated compute per successful task** = total estimated compute of all calls (including failed attempts and escalations) / number of correct answers.

- Compute per call ≈ 2 × active_params × (input + output tokens). This is an **estimate**, using the token counts returned by the API. Energy is not reported.
- Compute is the primary metric because `allam-2-7b` has no published price, so dollars can't compare all three models.
- **Secondary:** dollar cost per successful task, reported only when every call used a model with a verified price.
- A configuration only counts if it meets the **accuracy floor** and **latency floor** in [config.yaml](config.yaml).
- Selection rule: among configurations predicted to meet both floors, pick the one with the lowest compute per successful task, and break ties by latency.

## Data
There are three Arabic task types. Each has 30 calibration and 30 held-out test questions, split with seed 42.

| Type | Source | Graded by |
|---|---|---|
| `math` | [Arabic-GSM8K](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic-gsm8k) (MIT) | The final number |
| `knowledge` | [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) (CC-BY-NC-4.0), all 21 subjects | The option letter |
| `intent` | **Synthetic** chatbot messages in MSA and Saudi dialect: 8 intents plus `other` | The intent label |

- `data/calibration.jsonl` (90 questions): the only data KAYL learns from.
- `data/test.jsonl` (90 questions): held out and used only for the final evaluation.
- `data/robustness_en.jsonl` (20 questions): English GSM8K, used for testing only.
- Full details, including license, how the files are built and limitations, are in [DATA.md](DATA.md).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your GROQ_API_KEY in .env
python prepare_data.py      # rebuilds the data/ splits (no model calls)
# Optional: python generate_intents.py regenerates the synthetic intent data (calls Groq).
# Its output is already committed, so you do not need to run it.
```

## Planned experiments
All run on the same test set, at temperature 0, twice each.

| ID | Strategy |
|----|----------|
| A | Everything goes to the largest model |
| B | The model with the lowest token count |
| C | The lowest-price model |
| E | Routing by difficulty only (like RouteLLM) |
| D | KAYL: all three keys plus verify/escalate |
| Ablations | D minus each key, to show each key adds value |

## Configuration
Models, prices, floors and generation limits all live in [config.yaml](config.yaml).
The gpt-oss prices were checked against each model's Groq console page, and those links are in the config.
The allam-2-7b price is not known yet. Until it is, we report estimated compute for that model instead of cost.
(It replaced llama-3.1-8b-instant, which is no longer available on our Groq account.)
The floors are starting values. They will be set from the baseline A results on calibration data.
