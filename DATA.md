# KAYL — Data

This file documents every dataset KAYL uses: where it comes from, its license, how we built our files, how big they are, and what they cannot tell us.

**Rule:** KAYL learns and tunes only on **calibration** data. **Test** data and the **robustness** set are held out and used only for the final evaluation.

## Overview

| File | Items | Contents |
|---|---|---|
| `data/calibration.jsonl` | 90 | 30 math + 30 knowledge + 30 intent |
| `data/test.jsonl` | 90 | 30 math + 30 knowledge + 30 intent (held out) |
| `data/robustness_en.jsonl` | 20 | English GSM8K math (test only) |
| `data/synthetic/intents_raw.jsonl` | 100 | All generated intent requests (the source for the intent splits) |
| `data/synthetic/intents_generation.json` | — | Generation metadata: model, parameters, prompt, date |

Every item has `id`, `type` (`math` / `knowledge` / `intent`), `question` and `answer`.
Knowledge and intent items also have `choices`.
To rebuild everything, run `python prepare_data.py`. It makes no model calls. Intent data is read from the committed synthetic file.

All three task types use seed 42 (`random.Random(42)`). The split code `assert`s that no id appears in both calibration and test, and we also checked that no question text appears in both.

---

## 1. Math — Arabic-GSM8K

| Field | Value |
|---|---|
| Source | [Omartificial-Intelligence-Space/Arabic-gsm8k](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic-gsm8k) |
| Revision | `f59109bf8fef290c582de1b2658867b2a33330e4` (downloaded 2026-10-30) |
| License | MIT (Hugging Face license tag) |
| Origin | Arabic translation of OpenAI's GSM8K. We did not verify how the translation was made or checked. |
| Split used | `main_test` (1,319 problems) |
| Grading | The final number, after normalization (Arabic-Indic digits become 0–9, separators are removed, trailing `.0` is dropped) |
| Synthetic? | Not by us |

**How it is built:**
1. Extract the final answer: the number after the `####` marker.
2. Drop problems with no answer we can extract. That leaves 1,227.
3. Shuffle with seed 42. The first 30 go to calibration and the next 30 to test.

**Decision: accept `###` as well as `####` (2026-10-30, made before any experiment).**
GSM8K marks the final answer with `####`, but 19 problems in the Arabic version use `###`. We treat that as a formatting slip in the translation, not as a hard problem, so we accept both markers.
This decision changed which problems are in the splits. We made it before any model was run, so no result was tuned on it.
After this change, 92 problems are still dropped:
- 89 have no `#` marker at all.
- 3 have a marker but no number after it.

We do not try to guess their answer from the solution text.

**Limitations:**
- 92 of 1,319 problems (7%) are dropped. If those are harder or easier than average, the sample is slightly biased.
- All sampled answers are non-negative integers, so decimals and negatives are not tested.
- Translation quality is unverified.
- GSM8K is very widely used, so the models may have seen it during training (contamination).

## 2. Knowledge — ArabicMMLU

| Field | Value |
|---|---|
| Source | [MBZUAI/ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU), config `All` |
| Revision | `7aa530e2893ac420352b3f5c1a1310c010e9758b` (downloaded 2026-10-30) |
| License | **CC-BY-NC-4.0 (non-commercial)**: fine for research, not for a commercial product |
| Origin | Real school and university exam questions from Arab countries, written natively in Arabic (not translated) |
| Split used | `test` (14,455 questions, 21 subjects) |
| Grading | The correct option letter (`A`–`E`), taken from the `Answer Key` field |
| Synthetic? | No |

**Dataset fields we checked:** `ID`, `Source`, `Country`, `Group`, `Subject`, `Level`, `Question`, `Context`, `Answer Key`, `Option 1`–`Option 5`, `is_few_shot`.
- Questions have between 2 and 5 options. 10,120 have 4 options, 2,121 have 3, 1,874 have 2 and 340 have 5.
- Empty options are `None`.
- 707 questions include a `Context` passage. When there is one, we put it in front of the question.

**Sampling across subjects:**
1. Group the questions by subject and shuffle each group with seed 42.
2. Go through the subjects in alphabetical order, taking one question from each per round, until we have 60.
3. Alternate the picks between calibration and test.

As a result, all 21 subjects appear in both calibration and test, with 1–2 questions each.

**Limitations:**
- With 30 questions per split, we can only report overall knowledge accuracy, not accuracy per subject.
- The correct letters are not balanced. In calibration, A = 12, B = 10, C = 6 and D = 2. In test, A = 7, B = 11, C = 6 and D = 6. A model that always answers "A" would score 40% on calibration, so it is a baseline to keep in mind.
- Questions have different numbers of options (2 to 5), so random-guess accuracy varies from question to question.
- The models may have seen the dataset during training (contamination).

## 3. Intent classification — synthetic citizen service requests

| Field | Value |
|---|---|
| Source | **SYNTHETIC**, generated by us with `generate_intents.py` |
| Generation model | `groq/openai/gpt-oss-120b` (via LiteLLM) |
| Parameters | `temperature=0`, `reasoning_effort="low"`, `max_tokens=4096`, JSON response format |
| Generated on | 2026-10-30 |
| Generation cost | 4,740 input + 3,121 output tokens ≈ $0.0026 |
| Size | 10 intents × 2 dialects × 5 requests = 100 generated, with 0 duplicates |
| Grading | The correct intent label, which must exactly match one of the 10 labels |
| License | Our own data, released with this repo |

**The 10 intents** (label: what the generator was asked to write about):

| Label | Arabic description given to the generator |
|---|---|
| `renew_national_id` | تجديد بطاقة الهوية الوطنية |
| `renew_passport` | تجديد جواز السفر أو إصداره |
| `renew_driving_license` | تجديد رخصة القيادة |
| `traffic_violation_inquiry` | الاستعلام عن المخالفات المرورية أو سدادها أو الاعتراض عليها |
| `exit_reentry_visa` | إصدار تأشيرة خروج وعودة لعامل أو مقيم |
| `vehicle_ownership_transfer` | نقل ملكية مركبة (سيارة) من شخص لآخر |
| `book_medical_appointment` | حجز موعد في مركز صحي أو مستشفى حكومي |
| `report_utility_outage` | الإبلاغ عن انقطاع الكهرباء أو الماء |
| `social_support_inquiry` | الاستفسار عن الضمان الاجتماعي أو الدعم الحكومي |
| `file_complaint` | تقديم شكوى على جهة حكومية أو موظف بسبب سوء خدمة |

**The 2 dialects:**
- `msa`: العربية الفصحى الحديثة
- `saudi`: اللهجة السعودية العامية (كما يكتبها الناس في الواتساب وتويتر)

**Generation prompt.** The same template is used once for each (intent, dialect) pair, which makes 20 calls. Here it is verbatim:

```
اكتب {n} طلبات مختلفة يكتبها مواطن أو مقيم في السعودية لخدمة حكومية إلكترونية.
الهدف من كل الطلبات: {intent_desc}
اكتبها بـ: {dialect_desc}

الشروط:
- كل طلب جملة أو جملتان فقط، كما يكتبها شخص حقيقي.
- نوّع الصياغة: بعضها مباشر، وبعضها يشرح المشكلة دون ذكر اسم الخدمة صراحة.
- لا تكرر نفس الكلمات في كل الطلبات.
- لا تضف أي شرح أو ترقيم.

أعد JSON فقط بهذا الشكل: {"requests": ["...", "..."]}
```

The values filled in were `n = 5`, the Arabic intent description and the Arabic dialect description.
The same template is also stored in `data/synthetic/intents_generation.json`.

**How it is split:** For each intent, 3 requests go to calibration and 3 to test, chosen with seed 42.
The dialect mix alternates by intent: one intent puts 2 MSA + 1 Saudi in calibration, the next puts 1 MSA + 2 Saudi.
Each split ends up with exactly 15 MSA + 15 Saudi requests and 3 per intent. The 40 leftover generated requests are not used.

**Limitations:**
- **The labels are correct by construction, not checked by a human.** Each request is labeled with the intent it was generated for. A manual check of a sample is pending, and we will record its result here.
- **Possible bias.** The generator (gpt-oss-120b) is also one of the models KAYL routes to. It may find its own phrasing easier to classify, which could make 120b look better on this task than it would on real users.
- **The Saudi dialect is the model's imitation of it.** It may be lighter or more uniform than real user text, and some "Saudi" requests are close to MSA.
- **The task is easier than real traffic.** Every request has exactly one intent, there is no "other / out of scope" class, and there are no typos or mixed intents.
- **Small sample.** With 3 requests per intent per split, we cannot report accuracy for each intent separately.

## 4. Robustness — English GSM8K (test only)

| Field | Value |
|---|---|
| Source | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), config `main`, split `test` |
| Revision | `740312add88f781978c0658806c59bc2815b9866` (downloaded 2026-10-30) |
| License | MIT |
| Size | 20 problems, seed 42, `type: math`, `lang: en` |
| Purpose | Tests whether KAYL still behaves sensibly when the input is not Arabic |

**No leakage from calibration:** Row *i* of English GSM8K is the same problem as row *i* of Arabic-GSM8K. At 1,199 of 1,319 rows the final answers are identical, which confirms the rows line up.
So before sampling, we excluded the rows that are used in Arabic math calibration, and we checked that the overlap is 0.

**Limitations:**
- 20 problems is a very small set, so we report results with that caveat.
- This set measures language robustness only. It does not cover dialects or noisy input, which are planned separately.

---

## Planned datasets — *not created yet*

| Dataset | Purpose | Kind |
|---|---|---|
| Dialect versions of GSM8K problems | The same problems in Gulf, Egyptian and Levantine Arabic | **Synthetic** |
| Noisy input versions | Typos and Arabic-Indic digits | **Synthetic**, made by a script or an LLM |

### Rule for synthetic data
Any data we generate ourselves is marked **synthetic** (intent items carry `"synthetic": true`). Its section must record all of the following:
- **Generation model** and its parameters.
- **Generation prompt**, copied verbatim.
- **Generation script**: the command that reproduces the data.
- **Source items**: which ids it was derived from. Variants of test items stay on the test side and variants of calibration items stay on the calibration side, so nothing leaks between them.
- **Checks**: how we confirmed the answers or labels are correct.
