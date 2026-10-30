# KAYL — Data

This file documents every dataset KAYL uses: where it comes from, its license, how we built our files, how big they are, and what they cannot tell us.

**Rule:** KAYL learns and tunes only on **calibration** data. **Test** data and the **robustness** set are held out and used only for the final evaluation.

## Overview

| File | Items | Contents |
|---|---|---|
| `data/calibration.jsonl` | 90 | 30 math + 30 knowledge + 30 intent |
| `data/test.jsonl` | 90 | 30 math + 30 knowledge + 30 intent (held out) |
| `data/robustness_en.jsonl` | 20 | English GSM8K math (test only) |
| `data/synthetic/intents.csv` | 80 | All generated intent messages (the source for the intent splits). It can be edited by hand. |
| `data/synthetic/intents_generation.json` | — | Generation metadata: model, parameters, prompt, date, tokens |

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

## 3. Intent classification — synthetic chatbot messages

| Field | Value |
|---|---|
| Source | **SYNTHETIC**, generated by us with `generate_intents.py` |
| Generation model | `groq/qwen/qwen3.8-27b` (via LiteLLM) |
| Parameters | `temperature=0`, `reasoning_effort="none"` (thinking mode off), `max_tokens=800`, JSON response format |
| Generated on | 2026-10-30 |
| Generation tokens | 4,282 input + 1,756 output (final run). The price of qwen3.8-27b is not verified, so no cost is stated. |
| Size | 8 intents × 2 dialects × 4 + `other` × 2 dialects × 8 = **80 messages**, all unique |
| File | `data/synthetic/intents.csv`, with columns `text, intent, dialect, human_reviewed` (UTF-8 with BOM, so Excel shows the Arabic correctly) |
| Grading | The correct label, which must exactly match one of the 9 labels |
| License | Our own data, released with this repo |

**Why this generator.**
- The first version was generated by gpt-oss-120b, which is one of KAYL's own candidate models. That risks bias, because a model may find its own phrasing easier to classify.
- The plan was then to use `llama-3.3-70b-versatile`, but it is not available on our Groq account (the API returned `model_not_found`).
- We chose `qwen/qwen3.8-27b` because it is the strongest available model that is **not** a KAYL candidate.

**History.**
- Version 1 used gpt-oss-120b and had 10 intents and 100 long, formal requests. It was replaced because the requests sounded unnatural and came from a candidate model.
- Version 1 is still in git history (commit `c9d951c`).

**The 9 labels.** Each description is written so the label matches the text exactly. For example, `renew_*` means renewal only, not a first-time document and not a replacement for a lost one.

| Label | Arabic description given to the generator |
|---|---|
| `renew_national_id` | تجديد بطاقة هوية وطنية موجودة انتهت أو قربت تنتهي (فقط تجديد، ليس إصدار أول مرة ولا بدل فاقد) |
| `renew_passport` | تجديد جواز سفر موجود انتهى أو قرب ينتهي (فقط تجديد، ليس إصدار أول جواز ولا بدل فاقد) |
| `traffic_violation_inquiry` | الاستعلام عن المخالفات المرورية أو سدادها |
| `vehicle_ownership_transfer` | نقل ملكية سيارة من شخص لآخر (بيع أو شراء سيارة مستعملة) |
| `book_medical_appointment` | حجز موعد في مركز صحي أو مستشفى حكومي |
| `report_utility_outage` | الإبلاغ عن انقطاع الكهرباء أو الماء في البيت |
| `social_support_inquiry` | الاستفسار عن الضمان الاجتماعي أو حساب المواطن (الأهلية، موعد الصرف، حالة الطلب) |
| `file_complaint` | تقديم شكوى على جهة حكومية أو موظف بسبب سوء خدمة أو تأخير |
| `other` | رسائل لا علاقة لها بأي خدمة حكومية: سلام وتحية، الطقس، كرة القدم، وصفات أكل، نكت، أسئلة عامة عشوائية |

Compared with version 1, we removed `renew_driving_license` because it was too close to the other two `renew_*` intents. We also removed `exit_reentry_visa`, and we added `other`.

**The 2 dialects, as described to the generator:**
- `msa`: فصحى بسيطة كما يكتبها الناس العاديون، ليست رسمية ولا بلغة المعاملات الحكومية. مثال على الأسلوب: «كيف أجدد جوازي؟» وليس «أرجو التكرم بإفادتي عن إجراءات التجديد»
- `saudi`: لهجة سعودية حقيقية يومية كما يكتبها الناس في الواتساب. استخدم كلمات مثل: ابي، ابغى، ابغا، وش، كيف اسوي، ودي، ليش، الحين، وين، مره، يعطيك العافية

**Generation prompt.** The template is used once for each (label, dialect) pair, which makes 18 calls. Here it is verbatim:

```
اكتب {n} رسائل مختلفة يكتبها أشخاص حقيقيون في السعودية لروبوت محادثة (chatbot) تابع لبوابة خدمات حكومية.
موضوع كل الرسائل: {intent_desc}
اللغة: {dialect_desc}

القواعد:
- كل رسالة قصيرة: من 4 إلى 15 كلمة.
- عفوية كأن الشخص يكتب بسرعة من جواله: أحيانًا بدون علامات ترقيم، وأحيانًا فيها خطأ إملائي بسيط.
- كل رسالة تبدأ بشكل مختلف، ولا تكرر نفس الكلمات.
- كل رسالة يجب أن تطابق الموضوع بالضبط، ولا تخلطه بخدمة أخرى.
- لا تكتب أي شرح أو ترقيم أو أسماء.

أعد JSON فقط بهذا الشكل: {"messages": ["...", "..."]}
```

- `n` is the number needed plus 2 spares: 6 for each intent and 10 for `other`.
- The script keeps the first N messages that are unique and 4–15 words long.
- In the final run, it skipped one duplicate (`السلام عليكم كيف حالك اليوم`, which was generated for both dialects).
- The template is also stored in `data/synthetic/intents_generation.json`.

**How it is split:**
- For each of the 8 intents, 3 messages go to calibration and 3 to test. The dialect mix alternates by intent between (2 MSA + 1 Saudi) and (1 MSA + 2 Saudi).
- For `other`, 3 MSA + 3 Saudi go to each split.
- Each split therefore has 30 messages: 15 MSA, 15 Saudi, 3 per intent and 6 `other`. Of the 80 messages, 20 are not used.

**Automated pre-check (by Claude Code, not a human review):**
- Message length is 4–8 words, with an average of 5.4. Every message is within the 4–15 rule, but none is long.
- 69 of 80 messages have no final punctuation.
- 38 of 40 `saudi` messages contain a common Saudi word.
- **Problem: many `msa` messages are actually colloquial or Saudi.** Examples: `موعد انتهاء هويتي قريب ابي اجددها`, `جوازي قرب ينتهي وش اسوي`, `وش اطبخ اليوم عشان العشاء`. The generator interpreted "simple, not formal" as dialect, so the `dialect` column is not reliable for `msa` rows.
- **Labels that look ambiguous or wrong:**
  - `ابغا اعرف وين ملفي الحين` is labeled `file_complaint`, but it reads like a status question, not a complaint.
  - `طلبي مرفوض بدون سبب واضح` (`social_support_inquiry`) and `رفضوا قبول طلبي بدون سبب` (`file_complaint`) are nearly the same message with different labels.
  - `ابغى اعرف حالة طلبي الحين` (`social_support_inquiry`) does not mention which service it is about.
  - `وين اجدد موعد المستشفى` (`book_medical_appointment`) uses the word اجدد ("renew"), which could confuse a model with the `renew_*` intents.

**Human review: pending.** All rows have `human_reviewed = false`.
To review, edit `intents.csv` by hand (fix `text`, `intent` or `dialect`, and set `human_reviewed` to `true`), then run `python prepare_data.py` again.
Warning: running `generate_intents.py` again overwrites the CSV and erases the review.

**Limitations:**
- **The labels come from the generation prompt.** Until the human review is done, a label is the intent the message was *generated for*, not a checked judgment.
- **The dialects are a model's imitation.** The MSA/Saudi split is blurry (see above).
- **The messages are shorter and cleaner than real traffic.** There are almost no typos, every message has a single intent, and only one class (`other`) covers anything off-topic.
- **Small sample.** With 3 messages per intent per split, we cannot report accuracy for each intent.
- **Row ids.** Ids (`intent-<row>`) are row numbers in the CSV. If rows are deleted or reordered, the ids and the split change. Do not edit the CSV after experiments start.

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
