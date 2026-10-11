---
license: other
license_name: n-atlas-research-innovation
license_link: https://huggingface.co/NCAIR1/N-ATLaS
tags:
- nigerian-academic
- higher-education
- yoruba
- hausa
- igbo
- naic-2026
---

# Lumina-Edu Dataset Card

**Generated:** 2026-10-11 13:50 UTC
**Project:** Lumina-Edu - Powered by Awarri (NAIC 2026, PS3 Sectoral Fine-Tuning)
**Base model:** `NCAIR1/N-ATLaS`

## 1. Provenance

All records were harvested programmatically from the **University of Ibadan
Institutional Repository** via the OAI-PMH protocol
(`https://repository.ui.edu.ng/server/oai/request`, `metadataPrefix=oai_dc`).
The repository is listed in OpenDOAR and is operated by the University of
Ibadan, a federally funded NUC-recognised institution.

Every record retains its OAI handle, so any row can be traced back to the
source deposit:

```json
{
  "id": "ui_ibadan:oai:repository.ui.edu.ng:123456789/11603",
  "institution": "University of Ibadan",
  "title": "On the vowels of Imilike Dialect of the Igbo Language",
  "url": "https://repository.ui.edu.ng/handle/123456789/11603"
}
```

**No record in this dataset is synthetic.** The original project blueprint
proposed generating 13998+ "samples" by cycling two
hard-coded example papers; that approach was discarded because it cannot
support genuine evaluation.

## 2. Curation audit trail

| Stage | Count |
|---|---|
| Raw records harvested | 13998 |
| Removed: non-academic subjects | 2388 |
| Removed: non-research document types | 532 |
| Removed: encoding damage / low text ratio | 42 |
| Removed: abstract length out of bounds | 65 |
| Survived academic filtering | 10971 |
| Removed: exact duplicates | 672 |
| Removed: near-duplicates (cosine >= 0.82) | 293 |
| **Final deduplicated corpus** | **10006** |

The repository had absorbed a national newspaper archive, so the initial
harvest was dominated by "Political news", "Adverts", "Vacancies" and
"Sports" entries. These were filtered out before deduplication.

## 3. Splits

| Split | Records | Education-flagged | Median abstract chars |
|---|---|---|---|
| train | 9196 | 4240 | 1490 |
| dev | 200 | 30 | 1320 |
| benchmark_eval | 610 | 278 | 1469 |

- **train** - used for the QLoRA fine-tune only.
- **dev** - held out for loss monitoring; never trained on.
- **benchmark_eval** - the NAIC-mandated 500+ evaluation set. **Never seen
  during training.** Meets the "500+ example benchmark vs base N-ATLaS"
  requirement: **True**.

Splits are stratified by document type and education flag, so the benchmark
mirrors the corpus composition rather than over-representing one document type.

### Leakage verification

| Check | Result |
|---|---|
| ID overlap across splits | True |
| Overlap train/dev | 0 |
| Overlap train/benchmark | 0 |
| Overlap dev/benchmark | 0 |
| Max cosine similarity train vs benchmark | 0.6861 |
| Train/benchmark pairs above 0.5 similarity | 19 |

Near-duplicate removal was performed **before** splitting. This matters: the
repository contains the same work deposited under different document types,
so splitting first would have placed near-identical text on both sides of the
train/benchmark boundary and inflated every metric.

## 4. Benchmark composition

- Article: 524
- Thesis: 55
- Book chapter: 19
- Book: 2
- Dissertation: 2
- Technical Report: 2
- Dataset: 2
- Working Paper: 1
- In Proceedings: 1
- Book Chapter: 1
- INPROCEEDINGS: 1

## 5. Intended use

Fine-tuning `NCAIR1/N-ATLaS` for Nigerian higher-education literature tasks:
summarising theses, explaining findings in Yoruba/Hausa/Igbo, and grounding
responses in real institutional context rather than hallucinated citations.

## 6. Licensing and attribution

Source content remains the property of the University of Ibadan and the
respective authors, and is used here for open research under the repository's
terms. The derivative model inherits the N-ATLaS licence and carries the
required attribution:

> N-ATLaS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

Derivatives must retain the "Powered by Awarri" suffix and remain within the
1,000 active end-user cap defined in the N-ATLaS Terms of Use.

## 7. Known limitations

- **Single-institution corpus.** All records come from one repository, so
  institutional and disciplinary coverage skews towards Ibadan's research
  profile. Redeemer's University (~6,912 records) and University of Port
  Harcourt were identified but not included; the former aborted on a stale
  OAI resumption token.
- **Predominantly journal articles.** Only ~10% are theses or dissertations.
- **Metadata is Dublin Core**, so discipline labels are inconsistent
  (`Nigeria` is used as a subject alongside real subject terms).
- **No manual language annotation.** Language tags come from the repository
  and are only partially populated; the Yoruba/Hausa/Igbo targets rely on the
  base model's own capability rather than on labelled training signal.
- Records are abstracts, not full texts. Findings that live in a results
  section will not always be present in the abstract.

## 8. Reproduction

```bash
python scripts/probe_repositories.py     # discover live OAI endpoints
python scripts/harvest_oai.py --limit 20000
python scripts/build_corpus.py --eval-size 600 --dev-size 200
python scripts/make_dataset_card.py
```

Files under `data/processed/`: `corpus_all.jsonl`, `train.jsonl`, `dev.jsonl`,
`benchmark_eval.jsonl`, `curation_report.json`.
