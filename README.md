# Lumina-Edu — Powered by Awarri

**NAIC 2026 · Problem Statement 3: Sectoral Fine-Tuning (Education & Learning)**

A parameter-efficient fine-tune of [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS)
— Nigeria's sovereign multilingual LLM — adapted for Nigerian higher-education
literature research.

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation
> and Digital Economy, and powered by Awarri Technologies.
> See [`ATTRIBUTION.md`](ATTRIBUTION.md) for licence compliance.

## Why this project exists

Generic models fail on Nigerian academic literature in three specific ways:

1. **Hallucinated local citations** — inventing plausible-but-false Nigerian
   institutions and studies.
2. **No NUC curriculum awareness** — no grounding in how Nigerian degree
   programmes are actually structured.
3. **Weak African-language explanation** — N-ATLaS itself rates Yoruba 2.69/5
   and Hausa bias/fairness 1.11/5.

Lumina-Edu targets all three, and is evaluated on **500+ held-out real
abstracts** against the unmodified base model.

## Architecture

```
University of Ibadan IR (OAI-PMH)
        │  15,585 records harvested
        ▼
   curate + dedupe + stratify  ──►  10,006 academic records
        │                        9,196 train / 200 dev / 610 benchmark
        ▼
   self-distillation (N-ATLaS as its own teacher)
        │  Yoruba / Hausa / Igbo targets, filtered + audited
        ▼
   QLoRA fine-tune (r=16, chat-template native)
        │
        ▼
   FastAPI ──► Next.js console
```

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/harvest_oai.py` | OAI-PMH client for Nigerian repositories |
| `scripts/build_corpus.py` | Filter → near-dedupe → stratified disjoint split |
| `scripts/self_distill_translations.py` | Generates real Yoruba/Hausa/Igbo targets |
| `scripts/train_lumina_edu.py` | QLoRA fine-tune |
| `scripts/evaluate_benchmark.py` | 500+ real generations vs base model |
| `server/main.py` | FastAPI inference service |
| `kaggle_train_lumina_edu.ipynb` | 4-stage Kaggle training pipeline |
| [`DATASET_CARD.md`](DATASET_CARD.md) | Provenance, audit trail, limitations |
| `data/samples/` | 50-record sample of each split |

## Data provenance

All records come from the **University of Ibadan Institutional Repository**
via OAI-PMH (`oai_dc`). Every row retains its OAI handle, so any record can be
traced to its source deposit. **Nothing in the dataset is synthetic.**

The repository had absorbed a national newspaper archive, so the raw harvest
was filtered to remove "Political news", "Adverts", "Vacancies" and "Sports"
entries before deduplication.

### Leakage control

Near-duplicate removal runs **before** the train/benchmark split. The
repository contains the same work deposited under different document types, so
splitting first would place near-identical text on both sides of the boundary
and inflate every metric. Verification results are in
`data/processed/curation_report.json`.

## Reproducing

```bash
pip install -r requirements.txt

python scripts/probe_repositories.py          # discover live OAI endpoints
python scripts/harvest_oai.py --limit 20000  # fetch
python scripts/build_corpus.py               # curate + split
python scripts/make_dataset_card.py          # regenerate this documentation
```

Training and evaluation run on Kaggle (T4 x2) via
`kaggle_train_lumina_edu.ipynb`.

## Integrity notes

The original technical blueprint contained three components that produced
**fabricated results** rather than real measurements. They were rewritten:

| Blueprint | Problem | This repo |
|---|---|---|
| Benchmark script | Compared two hard-coded strings and reported the difference as ROUGE-L "improvement" | Loads both models and generates 500+ real responses |
| Dataset generator | 1,000 "samples" cycled from **2** hard-coded papers | 10,006 real records with OAI handles |
| FastAPI server | Returned a canned sentence with no model in the loop | Runs actual inference |

The NAIC requires "live benchmarks, not simulated or hypothetical results".
A simulated benchmark would also have failed the mandatory N-ATLAS
integration check.

## Licence

Derived from N-ATLaS and bound by its Terms of Use: same-licence
distribution, "Powered by Awarri" suffix, and a 1,000 active end-user cap.
Not licensed for commercial deployment without a separate agreement.