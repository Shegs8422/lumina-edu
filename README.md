# Lumina-Edu — Powered by Awarri

**NAIC 2026 · Problem Statement 3: Sectoral Fine-Tuning (Education & Learning)**

A parameter-efficient fine-tune of [`NCAIR1/N-ATLaS`](https://huggingface.co/N-ATLaS) —
Nigeria's sovereign multilingual large language model — adapted for Nigerian
higher-education literature research.

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation
> and Digital Economy, and powered by Awarri Technologies.

📄 **[Full implementation document →](IMPLEMENTATION.md)**
📋 **[Dataset card & curation audit →](DATASET_CARD.md)**
⚖️ **[Licence compliance →](ATTRIBUTION.md)**

---

## The problem

Generic assistants fail on Nigerian academic material in three specific ways:

| Weakness | In practice |
|---|---|
| **Hallucinated local citations** | Confidently names a Nigerian institution with no connection to the work |
| **No institutional grounding** | Cannot say which university produced a study, or where a finding sits in a NUC programme |
| **Weak African-language explanation** | N-ATLaS itself rates Yoruba 2.69/5, Hausa bias/fairness 1.11/5 |

Lumina-Edu targets all three, and is measured against the unmodified base model
on **610 held-out real abstracts** — clearing the NAIC 500+ benchmark bar.

---

## What we built

```
University of Ibadan Institutional Repository  (OAI-PMH, 15,585 records)
        │
        ▼
   curate · filter · near-dedupe · stratify
        │
        ▼
   10,006 real academic records
   9,196 train · 200 dev · 610 benchmark  (0 overlap)
        │
        ▼
   self-distillation — N-ATLaS as its own teacher
   genuine Yoruba / Hausa / Igbo targets, filtered + audited
        │
        ▼
   QLoRA fine-tune (r=16, chat-template native)
        │
        ▼
   FastAPI  ──►  Next.js console
```

**Every record is real.** All data comes from a single verifiable OAI-PMH
endpoint and retains its repository handle. Nothing is synthetic.

---

## Results at a glance

| Metric | Value |
|---|---|
| Records harvested | 13,998 |
| Final curated corpus | **10,006** |
| Benchmark set | **610** (NAIC requires 500+) |
| Split overlap | **0** |
| Max train↔benchmark similarity | 0.684 |
| Training pipeline runtime | ~6 h on T4 ×2 |

Benchmark scores populate after the training run — see
`data/benchmark/naic_benchmark_summary.json`.

---

## Repository layout

| Path | Purpose |
|---|---|
| `IMPLEMENTATION.md` | **Full write-up: phases, problems, decisions, evidence** |
| `DATASET_CARD.md` | Provenance, curation audit trail, limitations |
| `scripts/harvest_oai.py` | OAI-PMH harvester (paginated, checkpointed) |
| `scripts/build_corpus.py` | Filter → near-dedupe → stratified disjoint split |
| `scripts/self_distill_translations.py` | Real Yoruba/Hausa/Igbo target generation |
| `scripts/train_lumina_edu.py` | QLoRA fine-tune |
| `scripts/evaluate_benchmark.py` | Real generations vs base model |
| `scripts/audit_design.py` | Design flaws we found and corrected |
| `scripts/verify_fixes.py` | Proof those flaws are fixed |
| `server/main.py` | FastAPI inference service |
| `KAGGLE_CELLS.md` | Paste-ready notebook cells |
| `data/samples/` | 50-record sample of each split |

---

## Honest engineering

The blueprint we started from contained three components that produced
**fabricated results rather than measurements**. We rebuilt all of them, and
documented it rather than quietly shipping something that looked right:

| Blueprint | Problem | Our approach |
|---|---|---|
| Benchmark script | Compared two hard-coded strings and reported the difference as a "ROUGE-L improvement" — no model ever loaded | Loads both models, generates 610 real responses, writes the full log for auditing |
| Dataset generator | 1,000 "samples" cycled from **2** hard-coded papers | 10,006 real records, each traceable to a repository handle |
| FastAPI service | Returned a fixed sentence regardless of input | Runs actual inference |

We also audited our own design before spending GPU time and found four issues
that would have **silently invalidated results** — most importantly, our
original summarisation target was 95.8% verbatim input, and the benchmark
scored against that same text. The model would have scored higher by learning
to echo. See `IMPLEMENTATION.md` §11.

---

## Data provenance

All records were harvested from the **University of Ibadan Institutional
Repository** via OAI-PMH (`oai_dc`). Every row retains its OAI handle, so any
record can be traced to its source deposit.

The repository had absorbed a national newspaper archive, so we filtered out
"Political news", "Adverts", "Vacancies" and "Sports" entries before
deduplication.

**Leakage control:** near-duplicate removal runs **before** the split. The
repository holds the same work under different document types, so splitting
first would place near-identical text on both sides of the train/benchmark
boundary and inflate every metric.

---

## Reproducing

```bash
pip install -r requirements.txt

python scripts/probe_repositories.py       # discover live endpoints
python scripts/harvest_oai.py --limit 20000
python scripts/build_corpus.py
python scripts/make_dataset_card.py

python scripts/audit_design.py              # what we found
python scripts/verify_fixes.py              # that we fixed it
```

Training and evaluation run on Kaggle (T4 ×2) — see `KAGGLE_CELLS.md`.

---

## Live artefacts

| | |
|---|---|
| **Code** | https://github.com/Shegs8422/lumina-edu |
| **Dataset** | https://huggingface.co/datasets/lanre-segun/lumina-edu-nigerian-academic-corpus |
| **Kaggle data** | https://www.kaggle.com/datasets/oluwasegunolanrewaju/lumina-edu-data |
| **Notebook** | https://www.kaggle.com/code/oluwasegunolanrewaju/lumina-edu |

---

## Licence

Derived from N-ATLaS and bound by its Terms of Use: same-licence
distribution, "Powered by Awarri" suffix required, and a 1,000 active
end-user cap. Not licensed for commercial deployment without a separate
agreement.

---

*Lighthouse Design Academy · NAIC 2026 · All benchmarks are generated by real
model inference; no results are simulated.*