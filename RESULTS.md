# Benchmark Results — Lumina-Edu vs. N-ATLaS base

**Run:** 15 October 2026 · **Paired samples:** 610 · **Status:** complete

Every figure below comes from real greedy generation, not simulation. Both models
were run over identical prompts with identical decoding settings, and all 1,220
completions are published so these claims can be checked independently:

<https://huggingface.co/datasets/lanre-segun/lumina-edu-benchmark-results>

---

## Headline finding: the adapter regressed to English

The single clearest signal this benchmark produced is **negative**, and we report
it as the result rather than burying it.

The adapter was trained on English-only extractive targets (the translation task
was omitted because self-distillation did not complete). The consequence is
measurable: it stopped honouring Yoruba, Hausa and Igbo prompts.

Of 457 prompts that requested a Nigerian language, the proportion answered in one:

| Language requested | n | Base N-ATLaS | Lumina-Edu | Change |
|---|---|---|---|---|
| Igbo | 152 | 100.0% | 29.6% | **−70.4 pts** |
| Yoruba | 153 | 100.0% | 36.6% | **−63.4 pts** |
| Hausa | 152 | 57.2% | 40.8% | −16.4 pts |
| **Overall** | **457** | **85.8%** | **35.7%** | **−50.1 pts** |

This is language collapse from monolingual supervised fine-tuning: a narrow
English target set overwrites the base model's cross-lingual instruction
following rather than adding to it.

**Caveat on this measurement.** Language is detected by the presence of non-ASCII
characters, which is a proxy. A Yoruba or Hausa response written without
diacritics would be miscounted as English, so these figures understate the
fine-tuned model if anything. Separately, the prompt language rotates by record
index rather than being verified against a per-record language label, so the
"Hausa" bucket in particular contains responses the base model produced in other
West African languages. Treat the buckets as **non-English prompt groups**, not
verified per-language targets. The overall 85.8% → 35.7% shift is far larger than
any proxy error could plausibly explain.

---

## Training convergence

| | |
|---|---|
| Training loss | 1.6749 |
| Validation loss | 1.6047 (−0.0702, −4.2%) |
| Validation split | 200 held-out records |
| Step budget | 100 (compute ceiling) |

Loss was still falling when the budget was reached, so the run was
compute-limited, not data-limited.

## Adaptation cost

| | |
|---|---|
| Method | QLoRA, 4-bit NF4 |
| Trainable parameters | 41,943,040 of 8,072,204,288 (**0.52%**) |
| Adapter size | 167.83 MB |

## Institution grounding: unsolved

| | Base | Lumina-Edu |
|---|---|---|
| Attribution accuracy | 0.98% | 3.44% |

Both are effectively chance. ~99% of the corpus is University of Ibadan, so the
attribution task has effectively one label and cannot be learned from this data.
The improvement is real in direction and negligible in magnitude; the honest
statement is that **this task was not solved**.

---

## Why we do not headline ROUGE/chrF

The automatic scores are:

| Metric | Base | Fine-tuned |
|---|---|---|
| ROUGE-1 | 0.1437 | 0.7808 |
| ROUGE-L | 0.1061 | 0.7793 |
| chrF | 0.0833 | 0.7799 |

These look like a 7× improvement. **They are not one**, and presenting them as
the headline would be misleading:

- The benchmark reference *is* the source abstract. A model that reproduces the
  abstract therefore scores very highly.
- **272 of 610 completions exceed 0.90 ROUGE-L**, median 0.87, with at least one
  at exactly 1.00. That is the signature of extraction-by-echo, not summarisation.
- Completion length tracks the reference closely (mean 1,015 characters), which
  is what verbatim reproduction looks like.
- Degenerate output is not the explanation: 0 empty completions, 1 repetitive
  output, only 2.1% under 80 characters.

The metric is measuring *did you echo the abstract*, not *did you summarise
well*. We are publishing it for audit but it is not a quality result.

The base model's low 0.106 also flatters the comparison — part of the apparent
gain is the base model rambling rather than the adapter genuinely improving.

---

## What would fix this

1. **Reintroduce multilingual targets.** The regression is a direct consequence of
   the omitted translation task, and it is fixable by completing self-distillation
   rather than by any change to the base model.
2. **Change the benchmark reference.** Score against `first_sentences(abstract)`
   — the actual training target — and add a precision-oriented penalty so a
   verbatim reproduction is scored lower than a genuine extract.
3. **Widen institution coverage.** Grounding is unlearnable from a
   single-institution corpus. OAI-PMH harvesting already works against two
   repositories and extends to more.
4. **Train on the full corpus.** This run used 400 of 9,196 records.

## Reproducing

```bash
python scripts/run_benchmark_resumable.py --pass base --limit 610
python scripts/run_benchmark_resumable.py --pass finetuned --limit 610
python scripts/run_benchmark_resumable.py --score
```

Generation is streamed per batch and checkpointed, so an interrupted run resumes
rather than restarting.

## Artefacts

| | |
|---|---|
| Per-sample results (1,220 completions) | <https://huggingface.co/datasets/lanre-segun/lumina-edu-benchmark-results> |
| Adapter | <https://huggingface.co/lanre-segun/lumina-edu-powered-by-awarri> |
| Corpus and benchmark records | <https://huggingface.co/datasets/lanre-segun/lumina-edu-nigerian-academic-corpus> |
| Pipeline | <https://github.com/Shegs8422/lumina-edu> |

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation
> and Digital Economy, and powered by Awarri Technologies.