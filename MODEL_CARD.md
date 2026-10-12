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

**Generated:** 2026-10-12 16:44 UTC

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

**Not available at time of publication.** The benchmark run had not
completed when this card was generated. No result is claimed here.


## Intended use

Summarising and grounding Nigerian academic literature, and exploring
multilingual explanation of research findings.

## Limitations

- **Single-dominant institution.** ~99% University of Ibadan, so attribution
  generalises poorly to other Nigerian universities.
- **Abstracts only.** Findings stated solely in a results section are absent.
- **No multilingual targets in this adapter.** The translation task was
  omitted because self-distillation did not complete.
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

*Adapter path in the training run: `/content/outputs/adapter`*
