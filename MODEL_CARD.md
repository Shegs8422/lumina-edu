---
license: other
license_name: n-atlas-research-innovation
license_link: https://huggingface.co/NCAIR1/N-ATLaS
library_name: peft
base_model: NCAIR1/N-ATLaS
tags:
- nigerian-academic
- higher-education
- naic-2026
- qLoRA
---

# Lumina-Edu — Powered by Awarri

**QLoRA fine-tune of [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS)** for
Nigerian higher-education literature research.

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.

**Generated:** 2026-10-15 20:08 UTC

---

## What this model does

Lumina-Edu adapts N-ATLAS for three tasks grounded in real Nigerian academic
literature:

| Task | Purpose |
|---|---|
| **compress** | Reduce a research abstract to its main claim |
| **attribute** | Identify the university that produced a study |
| **translate** | Explain findings in Yoruba / Hausa / Igbo *(not present in this adapter — see Scope)* |

The attribution task targets the project's stated failure mode: generic models
naming plausible but false Nigerian institutions.

## What we measured, and what it means

The benchmark was run end to end: 610 held-out abstracts, base and fine-tuned
over identical prompts, greedy decoding, all 1,220 completions published.

**Two of the three findings are negative, and we report them as the result.**

1. **Training converged** — validation loss improved 4.2%, still falling at the
   compute ceiling.
2. **The adapter regressed to English** — it stopped honouring Yoruba, Hausa and
   Igbo prompts. This is the clearest signal the benchmark produced.
3. **Institution grounding is unsolved** — 3.4% against chance, because ~99% of
   the corpus is a single institution.

The favourable-looking ROUGE and chrF numbers are reported below but explicitly
not headlined: the benchmark reference is the source abstract, so the metric
rewards reproducing it. We would rather present a result that survives scrutiny
than one that does not.

## Primary finding: the adapter regressed to English

The clearest result this benchmark produced is negative, and we report it
rather than burying it.

The adapter was trained on English-only extractive targets. The consequence
is measurable: it stopped honouring Yoruba, Hausa and Igbo prompts. Of the
**457** benchmark prompts requesting a Nigerian
language, the proportion answered in one:

| Language requested | n | Base N-ATLaS | Lumina-Edu | Change |
|---|---|---|---|---|
| Igbo | 152 | 100.0% | 29.6% | **-70.4 pts** |
| Yoruba | 153 | 100.0% | 36.6% | **-63.4 pts** |
| Hausa | 152 | 57.2% | 40.8% | **-16.4 pts** |
| **Overall** | **457** | **85.8%** | **35.7%** | **-50.1 pts** |

This is **language collapse from monolingual supervised fine-tuning**: a narrow
English target set overwrites the base model's cross-lingual instruction
following rather than adding to it. N-ATLAS advertises Yoruba, Hausa and Igbo
support, so this is a regression against a capability the base model had.

## Training configuration

| | |
|---|---|
| Method | QLoRA, 4-bit NF4 |
| LoRA rank / alpha | 16 / 16 |
| Target modules | q, k, v, o, gate, up, down |
| Trainable parameters | 41,943,040 of 8,072,204,288 (**0.52%**) |
| Precision | fp16 (T4 has no bf16) |
| Sequence length | 1024 (measured: median 543 tokens, p90 931) |
| Base model context | 8,092 tokens |
| Hardware | Kaggle / Colab T4 |

Training loss **1.6749** -> validation loss **1.6047** (-4.2%) on a 200-record held-out dev split. Loss was still falling at the 100-step compute ceiling, so the run was compute-limited rather than data-limited.

Adaptation cost: **41,943,040** trainable parameters of 8,072,204,288 (**0.52%**), producing a 167.83 MB adapter.

## Scope of this adapter

This adapter was trained on a **reduced subset** of the corpus
(400 of 9,196 training records, 2 epochs) after an infrastructure failure
destroyed the first full training run. It is a genuine LoRA fine-tune and the
benchmark below is real, but it is **not** the full-corpus model we intended
to produce. The full run remains feasible and is the next step.


## Training data

10006 curated academic records from the **University of Ibadan Institutional
Repository**, harvested via OAI-PMH with OAI handles retained for provenance.
Non-academic contamination (newspaper archive) was filtered, then
near-duplicates removed **before** splitting so identical text cannot straddle
the train/evaluation boundary.

Held-out benchmark: **610 records**.

Full provenance and the curation audit: `DATASET_CARD.md`.

## Benchmark results

Held-out set: **610 real abstracts**, disjoint from training (0 ID overlap, max train/benchmark cosine 0.684).
Both models were run over identical prompts with greedy decoding (`do_sample=False`).

### Automatic metrics — reported, not headlined

| Metric | Base N-ATLaS | Lumina-Edu | Delta |
|---|---|---|---|
| ROUGE-1 | 0.1437 | 0.7808 | +0.6371 |
| ROUGE-L | 0.1061 | 0.7793 | +0.6732 |
| chrF | 0.0833 | 0.7799 | +0.6966 |
| Institution attribution accuracy | 0.0098 | 0.0344 | +0.0246 |

> **These scores should not be read as an improvement.**
>
> The benchmark reference *is* the source abstract, so a model that
> reproduces the abstract scores highly against it. **272 of 610** completions
> exceed 0.90 ROUGE-L (median 0.8699, maximum
> 1.00), and mean completion length
> (1015 chars) tracks the reference. That is the
> signature of extraction-by-echo, not summarisation.
>
> Degenerate output is ruled out: 0 empty completions,
> 1 repetitive, 13 under 80 characters.
>
> ROUGE also applies an English stemmer and is unreliable for Yoruba,
> Hausa and Igbo, where chrF is the appropriate metric. The base model's
> low ROUGE further flatters the comparison — part of the apparent gain is
> the base model rambling.

### Per-language

| Language | n | Metric | Base | Lumina-Edu | Delta |
|---|---|---|---|---|---|
| English | 153 | rougeL | 0.2806 | 0.8156 | +0.5350 |
| Hausa | 152 | chrf | 0.0236 | 0.8317 | +0.8081 |
| Igbo | 152 | chrf | 0.0285 | 0.7311 | +0.7026 |
| Yoruba | 153 | chrf | 0.0204 | 0.7403 | +0.7199 |

All figures come from real generation. Every completion is published
for independent audit:

<https://huggingface.co/datasets/lanre-segun/lumina-edu-benchmark-results>

Full analysis: <https://github.com/Shegs8422/lumina-edu/blob/main/RESULTS.md>

## Intended use

Summarising and grounding Nigerian academic literature, and exploring
multilingual explanation of research findings.

**This adapter should not be used for multilingual work.** It measurably
degrades the base model's Yoruba, Hausa and Igbo instruction following. Use
base N-ATLAS for those languages.

## Limitations

- **Multilingual ability was lost.** The share of Nigerian-language prompts
  answered in a Nigerian language fell from 85.8% to 35.7%. This is the primary
  measured weakness of this adapter.
- **The model reproduces abstracts rather than summarising them.** 272/610
  completions exceed 0.90 ROUGE-L against the source abstract. Do not treat this
  adapter as a summariser.
- **Single-dominant institution.** ~99% University of Ibadan, so attribution
  generalises poorly to other Nigerian universities.
- **Abstracts only.** Findings stated solely in a results section are absent.
- **No multilingual training targets in this adapter.** The translation task was
  omitted because self-distillation did not complete, which is the direct cause
  of the regression above.
- **ROUGE is unreliable for Yoruba/Hausa/Igbo** (English stemmer); chrF is the
  appropriate metric there.
- **Automatic metrics only.** No human evaluation was performed.

## Licence

Derived from N-ATLAS and bound by its Terms of Use: same-licence distribution,
the **"Powered by Awarri"** suffix is required, and use is capped at 1,000
active end-users. Not licensed for commercial deployment without a separate
agreement. Training data remains the property of the University of Ibadan and
the respective authors.

## Citation

```bibtex
@misc{lumina_edu_2026,
  title  = {Lumina-Edu: A Sectoral Fine-Tune of N-ATLAS for Nigerian Higher-Education Literature},
  author = {Lighthouse Design Academy},
  year   = {2026},
  note   = {NAIC 2026, Problem Statement 3}
}
```

*Adapter path in the training run: `(not recorded)`*
