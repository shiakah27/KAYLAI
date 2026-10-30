# KAYL — Data

This file documents every dataset KAYL uses: where it comes from, its license, how we built our files, how big they are, and what they cannot tell us.

**Rule:** KAYL learns and tunes only on **calibration** data. **Test** data is held out and used only for the final evaluation.

---

## 1. Arabic-GSM8K (math word problems) — *in use*

| Field | Value |
|---|---|
| Source | [Omartificial-Intelligence-Space/Arabic-gsm8k](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic-gsm8k) on Hugging Face |
| Dataset revision | `f59109bf8fef290c582de1b2658867b2a33330e4` (the version downloaded on 2026-10-30) |
| License | MIT (from the dataset's Hugging Face license tag) |
| Origin | Arabic translation of OpenAI's GSM8K (grade-school math word problems). We did not verify how the translation was made or checked; see the dataset card. |
| Split used | `main_test` (1,319 problems). We do not use `main_train` or the `socratic_*` splits. |
| Synthetic? | Not by us. We use the published data unchanged. |

### How our files are created
The command `python prepare_data.py` does the following:

1. Downloads the `main_test` split (1,319 problems).
2. Extracts the final answer, which is the number after `####` in the answer field. It also turns Arabic-Indic digits (٠–٩) into 0–9, removes thousands separators, and drops a trailing `.0`.
3. Drops problems whose final answer cannot be extracted. That is **111** problems, leaving **1,208**.
4. Shuffles the remaining problems with `random.Random(42)`, using a fixed seed.
5. Takes the first 30 as calibration and the next 30 as test.
6. Checks that no id appears in both files.

### Sample sizes

| File | Size | Purpose |
|---|---|---|
| `data/calibration.jsonl` | 30 | KAYL learns its routing decisions from this only |
| `data/test.jsonl` | 30 | Held out, used only for the final evaluation |
| Overlap | 0 | Checked by an `assert` in `prepare_data.py` |

Each line looks like this: `{"id": "gsm8k-ar-608", "type": "math", "question": "...", "answer": "8"}`.
The `id` is the problem's row index in `main_test`.

### Limitations
- **Small samples.** With n = 30, one question is worth 3.3 accuracy points. An observed 80% has a 95% confidence interval of about 63–90% (Wilson). Small differences between configurations are **not** significant. We will report n next to every number.
- **Extraction drops problems.** 108 of the 111 dropped problems end with `###` (three #) instead of `####`, which looks like a formatting slip in the dataset. The other 3 have `####` with no number after it. Problems with this slip are under-represented. That is probably harmless, but it is not proven.
- **Integer answers only.** All 60 sampled answers happen to be whole, non-negative numbers, so decimal and negative answers are not tested.
- **Translation quality is unverified.** Some problems may read unnaturally, or have meanings that shifted in translation.
- **Possible contamination.** GSM8K is very widely used. The models may have seen the English originals, or this translation, during training.
- **Math only, Modern Standard Arabic only.** This data says nothing about other task types or dialects. That is why we plan the robustness sets below.
- **The revision is not pinned in code.** `prepare_data.py` downloads the latest version. If the dataset changes, the files could change too. The committed `data/*.jsonl` files are the reference copy.

---

## 2. Planned datasets — *not created yet*

These are planned for the robustness scenarios listed in [CLAUDE.md](CLAUDE.md). Each one gets a full section like the one above when it is created.

| Dataset | Purpose | Kind |
|---|---|---|
| ArabicMMLU subset | A different task type (multiple choice) | Published data; source and license to be recorded |
| Dialect versions of GSM8K problems | The same problems written in Gulf, Egyptian and Levantine Arabic | **Synthetic** |
| Noisy input versions | Typos and Arabic-Indic digits in the question | **Synthetic**, made by a script or an LLM |
| Arabic service questions | Realistic customer-service requests | **Synthetic**, or written by hand |

### Rule for synthetic data
Any data we generate ourselves is marked **synthetic**. Its section must record all of the following:
- **Generation model**: the exact model id and its parameters (temperature, seed if any).
- **Generation prompt**: the full prompt text, copied verbatim into this file or into a linked file in the repo.
- **Generation script**: the command that reproduces the data.
- **Source items**: which calibration or test ids it was derived from. Variants of test items stay on the test side, and variants of calibration items stay on the calibration side, so nothing leaks between them.
- **Checks**: how we confirmed that answers are still correct, for example a manual review of N items.
