"""
Generate DATASET_CARD.md for the Lumina-Edu corpus from the curation audit.

The card is the artefact a NAIC judge or NCAIR reviewer reads to judge whether
our training data is real, licensed and reproducible. It is generated from
data/processed/curation_report.json rather than hand-written, so the numbers
quoted can never drift from what actually shipped.

Run:  python scripts/make_dataset_card.py
"""

import json
import os
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(PROJECT_ROOT, "data", "processed", "curation_report.json")
OUT = os.path.join(PROJECT_ROOT, "DATASET_CARD.md")


def load():
    with open(REPORT, encoding="utf-8") as fh:
        return json.load(fh)


def build(a):
    s = a.get("stats", {})
    tr, dv, be = s.get("train", {}), s.get("dev", {}), s.get("benchmark", {})

    def row(name, d):
        return (
            f"| {name} | {d.get('count', 0)} | "
            f"{d.get('education_flagged', 0)} | {d.get('median_abstract_chars', 0)} |"
        )

    types = be.get("by_type", {})
    type_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(types.items(), key=lambda x: -x[1]))

    return f"""# Lumina-Edu Dataset Card

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
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
{{
  "id": "ui_ibadan:oai:repository.ui.edu.ng:123456789/11603",
  "institution": "University of Ibadan",
  "title": "On the vowels of Imilike Dialect of the Igbo Language",
  "url": "https://repository.ui.edu.ng/handle/123456789/11603"
}}
```

**No record in this dataset is synthetic.** The original project blueprint
proposed generating {a.get('raw_records', 0)}+ "samples" by cycling two
hard-coded example papers; that approach was discarded because it cannot
support genuine evaluation.

## 2. Curation audit trail

| Stage | Count |
|---|---|
| Raw records harvested | {a.get('raw_records', 0)} |
| Removed: non-academic subjects | {a.get('dropped_by_filter', {}).get('non_academic_subject', 0)} |
| Removed: non-research document types | {a.get('dropped_by_filter', {}).get('non_research_type', 0)} |
| Removed: encoding damage / low text ratio | {a.get('dropped_by_filter', {}).get('encoding_damage', 0) + a.get('dropped_by_filter', {}).get('low_alphabetic_ratio', 0)} |
| Removed: abstract length out of bounds | {a.get('dropped_by_filter', {}).get('abstract_too_short', 0) + a.get('dropped_by_filter', {}).get('abstract_truncated_garbage', 0)} |
| Survived academic filtering | {a.get('after_filter', 0)} |
| Removed: exact duplicates | {a.get('dropped_exact_dupe', 0)} |
| Removed: near-duplicates (cosine >= {a.get('near_dupe_threshold', 0)}) | {a.get('dropped_near_dupe', 0)} |
| **Final deduplicated corpus** | **{a.get('after_near_dedupe', 0)}** |

The repository had absorbed a national newspaper archive, so the initial
harvest was dominated by "Political news", "Adverts", "Vacancies" and
"Sports" entries. These were filtered out before deduplication.

## 3. Splits

| Split | Records | Education-flagged | Median abstract chars |
|---|---|---|---|
{row('train', tr)}
{row('dev', dv)}
{row('benchmark_eval', be)}

- **train** - used for the QLoRA fine-tune only.
- **dev** - held out for loss monitoring; never trained on.
- **benchmark_eval** - the NAIC-mandated 500+ evaluation set. **Never seen
  during training.** Meets the "500+ example benchmark vs base N-ATLaS"
  requirement: **{a.get('benchmark_meets_naic_500', False)}**.

Splits are stratified by document type and education flag, so the benchmark
mirrors the corpus composition rather than over-representing one document type.

### Leakage verification

| Check | Result |
|---|---|
| ID overlap across splits | {a.get('splits_disjoint', False)} |
| Overlap train/dev | {a.get('split_overlap', {}).get('train&dev', 0)} |
| Overlap train/benchmark | {a.get('split_overlap', {}).get('train&benchmark', 0)} |
| Overlap dev/benchmark | {a.get('split_overlap', {}).get('dev&benchmark', 0)} |
| Max cosine similarity train vs benchmark | {a.get('max_train_eval_cosine', 0)} |
| Train/benchmark pairs above 0.5 similarity | {a.get('train_eval_pairs_above_0.5', 0)} |

Near-duplicate removal was performed **before** splitting. This matters: the
repository contains the same work deposited under different document types,
so splitting first would have placed near-identical text on both sides of the
train/benchmark boundary and inflated every metric.

## 4. Benchmark composition

{type_lines}

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
"""


def main():
    a = load()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(build(a))
    print(f"Wrote {OUT}")
    print(f"  corpus={a.get('after_near_dedupe', 0)} benchmark={a.get('benchmark_size', 0)} "
          f"disjoint={a.get('splits_disjoint', False)}")


if __name__ == "__main__":
    main()