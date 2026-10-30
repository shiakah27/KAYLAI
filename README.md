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
   (`llama-3.1-8b-instant`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`).
2. **How much the model thinks**: reasoning effort `low` / `medium` / `high`
   (only on the gpt-oss models).
3. **Which language it thinks in**: Arabic or English reasoning. The answer is always in Arabic.

Then it **verifies** the answer and **escalates** (more effort or a bigger model) only when the check fails.
KAYL does not train any model. It only uses controls that already exist.

## Core metric
**Cost per successful task** = total cost (including failed attempts and escalations) / number of correct answers.

- Cost = (input_tokens × input_price + output_tokens × output_price) / 1,000,000, using token counts returned by the provider.
- A configuration only counts if it meets the **accuracy floor** and **latency floor** in [config.yaml](config.yaml).
- Secondary metric: **estimated compute per task** = 2 × active_params × total tokens. This is an estimate, not a measurement. Active parameters come from the official model cards. Energy is not reported.
- Selection rule: among configurations predicted to meet both floors, pick the cheapest, and break ties by latency.

## Data
There are three Arabic task types. Each has 30 calibration and 30 held-out test questions, split with seed 42.

| Type | Source | Graded by |
|---|---|---|
| `math` | [Arabic-GSM8K](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic-gsm8k) (MIT) | The final number |
| `knowledge` | [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) (CC-BY-NC-4.0), all 21 subjects | The option letter |
| `intent` | **Synthetic** citizen service requests in MSA and Saudi dialect, 10 intents | The intent label |

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
The llama-3.1-8b-instant price is still unverified.
The floors are starting values. They will be set from the baseline A results on calibration data.
