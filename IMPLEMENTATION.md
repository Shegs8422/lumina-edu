# Implementation Document

**Project:** Lumina-Edu — Powered by Awarri
**Challenge:** National AI Innovation Challenge (NAIC) 2026 · Track B (Innovation & Enterprise) · Problem Statement 3 (Sectoral Fine-Tuning)
**Base model:** [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS)
**Team:** Lighthouse Design Academy

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation
> and Digital Economy, and powered by Awarri Technologies.

---

## Table of contents

1. [The problem we set out to solve](#1-the-problem-we-set-out-to-solve)
2. [What we found in the original blueprint](#2-what-we-found-in-the-original-blueprint)
3. [System architecture](#3-system-architecture)
4. [Phase 1 — Source discovery](#4-phase-1--source-discovery)
5. [Phase 2 — Corpus construction](#5-phase-2--corpus-construction)
6. [Phase 3 — Self-distillation](#6-phase-3--self-distillation)
7. [Phase 4 — Fine-tuning](#7-phase-4--fine-tuning)
8. [Phase 5 — Evaluation](#8-phase-5--evaluation)
9. [Phase 6 — Deployment](#9-phase-6--deployment)
10. [Performance engineering](#10-performance-engineering)
11. [Quality audit](#11-quality-audit)
12. [Results and evidence](#12-results-and-evidence)
13. [Known limitations](#13-known-limitations)
14. [Licence compliance](#14-licence-compliance)
15. [Reproduction](#15-reproduction)

---

## 1. The problem we set out to solve

N-ATLaS is Nigeria's sovereign multilingual model. It performs well on
instruction following, but our experience with it on Nigerian academic
material surfaced three specific weaknesses:

| Weakness | What it looks like in practice |
|---|---|
| **Hallucinated local citations** | Produces a confident, plausible-sounding Nigerian institution that has no connection to the work being discussed |
| **No institutional grounding** | Cannot tell which university produced a piece of research, or where a finding sits in a NUC programme |
| **Weak African-language explanation** | The model's own card rates Yoruba at 2.69/5 and Hausa bias/fairness at 1.11/5 |

These matter because Nigerian students and researchers work in exactly this
domain daily, and generic assistants are confidently wrong there.

**Our goal:** a fine-tuned model that stays anchored to real institutional
context, can explain findings in Yoruba, Hausa and Igbo, and is measurably
better than the base model on held-out real data.

---

## 2. What we found in the original blueprint

We were given a detailed technical blueprint. Reviewing it before writing any
code surfaced three components that could not work as described. We want to be
direct about this, because it shaped the whole approach.

### 2.1 The benchmark did not measure anything

The blueprint's benchmark script compared two hard-coded strings:

```python
base_simulated = f"Generic academic output for case {case['id']}."
fine_tuned_simulated = f"Localized analysis {case['id']} citing proper institutional standards."
```

It then scored those two strings against each other and reported the
difference as a ROUGE-L "improvement percentage". No model was ever loaded.
The result would have looked impressive and meant nothing.

The NAIC requires "live benchmarks, **not simulated or hypothetical
results**", and NCAIR runs a separate N-ATLAS integration verification pass.
We rewrote this to load the base model, generate real responses, attach the
adapter, regenerate, and score the actual text.

### 2.2 The dataset was two examples repeated

The blueprint's dataset generator produced 1,000 "samples" by cycling through
two hard-coded papers. Two hundred copies of a Yoruba sentence and two hundred
copies of a Hausa sentence. Fine-tuning on that teaches memorisation of two
paragraphs, not a sector.

### 2.3 The API returned a canned response

The blueprint's FastAPI service returned a fixed, scholarly-sounding sentence
regardless of input — no model loaded at all. A judge testing it with any query
would have seen the same answer.

**Our decision:** we kept the blueprint's architecture and intent, and rebuilt
every component that could not produce evidence. Section 11 documents this in
full.

---

## 3. System architecture

```
┌──────────────────────────────────────────────────────────────┐
│  CLIENT          Next.js 14 on Vercel ($0)                  │
│                  Query console: English / Yoruba / Hausa /  │
│                  Igbo, with required N-ATLaS attribution     │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTPS
┌───────────────────────────▼──────────────────────────────────┐
│  INFERENCE      FastAPI + Unsloth 4-bit                      │
│                 NCAIR1/N-ATLaS  +  Lumina-Edu LoRA adapter    │
│                 (HF Space / ZeroGPU — NOT Render free tier)  │
└───────────────────────────┬──────────────────────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        │                                        │
┌───────▼────────────┐              ┌────────────▼─────────────┐
│ ADAPTER            │              │ CORPUS                   │
│ LoRA r=16          │              │ 10,006 curated records   │
│ HF Hub             │              │ University of Ibadan IR   │
│ "Powered by        │              │ via OAI-PMH, real handles │
│  Awarri"           │              └──────────────────────────┘
└────────────────────┘
```

**Base model:** Llama-3 8B, 391M instruction tokens across English, Hausa,
Igbo and Yoruba. 8,092-token effective context.

**Why we fine-tune rather than prompt:** the three failure modes are
consistent behavioural tendencies, not one-off errors. Fine-tuning on
thousands of real examples is the right instrument; prompting cannot reliably
suppress a pattern learned across 400M tokens.

---

## 4. Phase 1 — Source discovery

**Goal:** find real Nigerian academic corpora with machine-readable access.

We assumed repository URLs would follow a predictable pattern and tested 53
candidate hosts across NUC federal and premier state universities. The
assumption was wrong in instructive ways.

| Finding | Detail |
|---|---|
| 44 of 53 hosts unreachable | Connection errors, timeouts, or Cloudflare bot protection |
| **UNILAG's real repository is `ir.unilag.edu.ng`** | `repository.unilag.edu.ng` is an unrelated WordPress careers site — we had been probing the wrong host entirely |
| UNILAG backend is dark | DSpace 9.1 frontend loads, but every API and OAI path returns 404 |
| UNIPORT is DSpace 9.2 | REST API live, OAI not mounted |
| FUNAAB and Babcock respond but hold nothing | `noRecordsMatch` under every metadata format |

**Outcome:** four live OAI-PMH endpoints. Two held substantial data:

| Institution | Records | Status |
|---|---|---|
| **University of Ibadan** | **15,585** | Fully harvested |
| **Redeemer's University** | **6,912** | Partially harvested (316 usable) |
| FUNAAB Abeokuta | 0 | OAI live, empty |
| Babcock University | 0 | OAI live, empty |

### 4.1 Three protocol errors we had to diagnose

Each produced a *plausible but false* reading, which is the most dangerous
kind of bug:

**`completeListSize` is on the wrong element.** The total record count is an
attribute on `<resumptionToken>`, not on `<ListIdentifiers>`. Reading it off
the parent returned `None`, and the script's fallback inferred 100 records —
making a healthy 15,585-record repository look nearly empty. Had we trusted
that first result, we would have abandoned a viable corpus.

**Schema URLs are not prefixes.** `ListMetadataFormats` returns XML *schema*
URIs, but the protocol requires the short *prefix* (`oai_dc`). Sending the URL
back verbatim produced `cannotDisseminateFormat` on all four repositories.

**`resumptionToken` invalidates other parameters.** Page 1 always succeeds;
page 2 rejects `metadataPrefix` sent alongside the token with `badArgument`.
This fails late and looks like a server fault.

---

## 5. Phase 2 — Corpus construction

**Goal:** turn 13,998 raw records into a defensible training and evaluation set.

### 5.1 Contamination we did not expect

The University of Ibadan repository has absorbed a **national newspaper
archive**. Our first profile showed the top "subjects" as Political news
(1,417), Adverts (1,183), Vacancies (1,060) and Sports (1,024) — all
outranking genuine academic headings.

This matters for a competition submission: a judge who opens our dataset
expects Nigerian academic research, not newspaper clippings. We filter on both
subject vocabulary and document type.

### 5.2 Curation pipeline

| Stage | Records | Removed |
|---|---|---|
| Raw harvest | 13,998 | — |
| After academic filter | 10,971 | 2,388 non-academic subjects, 532 non-research types, 107 encoding/length rejects |
| After exact deduplication | 10,299 | 672 duplicate titles |
| **After near-duplicate removal** | **10,006** | 293 near-duplicates (cosine ≥ 0.82) |

Exact matching is not sufficient. The repository holds the same work deposited
under different document types, and syndicated copies share long verbatim
spans. We use TF-IDF cosine similarity with radius-based neighbour search to
catch these.

### 5.3 Leakage control — the part that makes the benchmark real

**We deduplicate before splitting, not after.** This is the single most
important decision in the project.

The repository contains the same article deposited as both an `Article` and a
`Book chapter`. If we had split first, near-identical text would land on both
sides of the train/benchmark boundary — and every metric would be inflated by
memorisation rather than capability.

Verified in `data/processed/curation_report.json`:

| Check | Result |
|---|---|
| ID overlap across splits | **0** |
| Maximum train↔benchmark cosine similarity | **0.684** |
| Train/benchmark pairs above 0.5 similarity | 35 of 5.6M |
| NAIC 500+ benchmark requirement | **Met (610)** |

Final split: **9,196 train / 200 dev / 610 benchmark**, stratified by source,
document type and education flag.

---

## 6. Phase 3 — Self-distillation

**Goal:** obtain genuine Yoruba, Hausa and Igbo training targets without
fabricating them.

### 6.1 The problem with a translation task

The obvious approach — ask the model to explain findings in Yoruba — has no
target text to train against. Our first draft used a placeholder:

```
[Yoruba explanation of: <English abstract>]
```

We rejected this. Training on it teaches the model to emit bracketed
scaffolding rather than Yoruba — hollowing out the exact claim the project
makes. **Our trainer now omits the translation task entirely when no real
target exists**, rather than falling back to a placeholder.

### 6.2 Our approach: the base model as its own teacher

We generate translations with N-ATLaS itself, then fine-tune on that output.
This is legitimate self-distillation, but only if we are honest about its
limits: the base model's Yoruba is rated 2.69/5, so these targets are
imperfect. We therefore:

1. **Screen every generation.** Output that stayed in English is discarded —
   it is a failed sample, not a target.
2. **Detect decoding collapse.** Repetition loops are rejected automatically.
3. **Emit an audit sample** for manual review by fluent speakers, and report
   the accuracy honestly in the dataset card.

We do not claim translation quality without that audit.

---

## 7. Phase 4 — Fine-tuning

**Approach:** QLoRA via Unsloth, targeting all linear projections.

| Parameter | Value | Reason |
|---|---|---|
| Method | QLoRA, 4-bit NF4 | Fits 8B on T4 GPUs |
| LoRA rank / alpha | 16 / 16 | Standard for adaptation of this size |
| Sequence length | 1024 | Measured, not guessed (Section 10) |
| Context ceiling | 8,092 | N-ATLaS effective limit; we stay well below |
| Prompt format | **N-ATLaS's own chat template** | See below |

### 7.1 A training-format mismatch we corrected

The blueprint trained on raw Alpaca text:

```
### Instruction: {} ### Input: {} ### Response: {}
```

N-ATLaS is a chat model with its own template that injects a system preamble
and date. Training on Alpaca while serving with the chat template puts the
model in a format it never saw at evaluation — which degrades the adapter and
can make the fine-tuned model *worse* than the baseline it is meant to beat.

We train through `tokenizer.apply_chat_template`, so training and inference
formats match exactly.

### 7.2 Training tasks

| Task | Purpose | Target |
|---|---|---|
| `compress` | Summarise to the main claim | Extractive lead, ~24% of input length |
| `translate_*` | Yoruba / Hausa / Igbo explanation | Self-distilled teacher output |
| `attribute` | Identify the source institution | Verified from repository metadata |

---

## 8. Phase 5 — Evaluation

**NAIC requirement:** a benchmark of 500+ examples against base N-ATLaS.

Our benchmark runs **500 held-out records through both models** — base first,
then adapter — with greedy decoding, fixed seed, identical prompts, and writes
the full per-sample generation log to disk so any judge can audit it.

### 8.1 Metrics, and why we chose them

| Metric | Applies to | Rationale |
|---|---|---|
| ROUGE-1 / ROUGE-L | English | Standard lexical overlap |
| **chrF** | **Yoruba, Hausa, Igbo** | Character-level, accent-normalised — see 8.2 |
| **Attribution accuracy** | All | Did the model name the correct institution? |

### 8.2 ROUGE is unreliable for our target languages

ROUGE applies an **English stemmer**. Measured directly:

```
rougeL(Yoruba reference, English text) = 0.000
chrf  (Yoruba reference, English text) = 0.047
```

Scoring Yoruba with English stemming is close to meaningless — and Yoruba is
precisely what we claim to improve. We headline **chrF** for non-English and
label the ROUGE figures as unreliable there.

### 8.3 Attribution accuracy tests the actual problem

The project's stated problem is hallucinated local citations. We measure it
directly: given an abstract, does the model name the institution that actually
produced it — and avoid naming a plausible wrong one?

---

## 9. Phase 6 — Deployment

**Infrastructure constraint we discovered:** Render's free tier provides 512 MB
of RAM. An 8B model in 4-bit needs roughly 6 GB. The blueprint's "deploy to
Render ($0)" plan cannot work, and we would rather say so than ship something
that crashes on first request.

**Our choice:** a Hugging Face Space with ZeroGPU hosting FastAPI, with the
adapter loaded from our own repository. This keeps the $0 budget, keeps the
adapter publicly loadable as integration evidence, and satisfies the licence
requirement that derivatives carry the "Powered by Awarri" suffix.

---

## 10. Performance engineering

We measured rather than estimated. Our first scripts declared a `batch_size`
parameter and then **never used it** — every generation ran one at a time,
leaving most of the GPU idle.

### 10.1 Sequence length, measured

We downloaded only the tokenizer (a few MB, not the 16 GB of weights) and
measured real token lengths on 3,000 actual training examples:

```
min 182 | median 543 | p75 756 | p90 931 | p99 1211 | max 2149

MAX_SEQ_LENGTH=512   truncates 56.1%   saves 75% compute   rejected
MAX_SEQ_LENGTH=768   truncates 23.9%   saves 62% compute   rejected
MAX_SEQ_LENGTH=1024  truncates  5.1%   saves 50% compute   chosen
MAX_SEQ_LENGTH=2048  truncates  0.1%   saves  0% compute   the waste
```

2048 was padding virtually every sequence to double its natural length for
0.1% benefit.

### 10.2 Batched generation

Both the distillation and benchmark generators now batch, with left padding
(the requirement for correct batched decoding on a decoder-only model). We also
truncate abstracts before translating and cap generation length, since a
focused explanation is sufficient training signal.

### 10.3 Net effect

| Stage | Before | After |
|---|---|---|
| Self-distillation | 9–13 h | **3–4 h** |
| Fine-tuning | ~4 h | **~2 h** |
| Benchmark (500 × 2 models) | ~3 h | **~25 min** |
| **Total** | **16–20 h** | **~6 h** |

This turned a pipeline that would have consumed our entire GPU allocation into
one that leaves real margin for failure.

---

## 11. Quality audit

Before committing GPU hours, we audited our own design. Four findings would
have **silently invalidated the results** — none would have raised an error.

### 11.1 We were training the model to echo (critical)

The original `summarise` target was the **entire abstract verbatim** with a
few words prepended — 95.8% copy, on 400/400 examples. Because the benchmark
scores against that same abstract, and the abstract is also in the prompt, a
model trained to copy would score **higher for the wrong reason**.

*Fix:* extractive compression. Target is now 24% of input length, so copying
the whole input is penalised.

### 11.2 The hallucination metric scored correct answers as hallucinations

The original check asked whether the named institution appeared in the prompt.
The prompt **never names the institution** — so naming it correctly returned
`False`. The metric inverted its own signal.

*Fix:* match the prediction against the record's verified source institution,
and flag a wrong-but-plausible institution as confabulation.

While fixing this we introduced a fresh bug: `"Ibadan"` is a substring of
`"University of Ibadan"`, so including bare city names made *correct* answers
look like confabulations. Caught by test, fixed by using distinct institution
names only.

### 11.3 The benchmark was mostly an unseen institution

Stratifying by document type alone placed **all 316 Redeemer's records in the
benchmark and none in training** — 46% of the evaluation set drawn from an
institution the model had never seen. That measures memorisation failure, not
sector adaptation.

*Fix:* stratify by source as well. Train and benchmark now share matching
distributions.

### 11.4 The language filter accepted English as Igbo

The marker `"a"` — the English article — was in the Igbo list. An English
sentence could pass the gate and be trained on as Igbo.

*Fix:* replaced with unambiguous Igbo orthography.

### 11.5 Verification

All fixes are covered by `scripts/verify_fixes.py`, including edge cases. The
suite reports **ALL CHECKS PASS**.

---

## 12. Results and evidence

### Artefacts

| Artefact | Location |
|---|---|
| Source code & pipeline | https://github.com/Shegs8422/lumina-edu |
| Dataset (public) | https://huggingface.co/datasets/lanre-segun/lumina-edu-nigerian-academic-corpus |
| Kaggle dataset | https://www.kaggle.com/datasets/oluwasegunolanrewaju/lumina-edu-data |
| Training notebook | https://www.kaggle.com/code/oluwasegunolanrewaju/lumina-edu |
| Dataset card & audit | [`DATASET_CARD.md`](DATASET_CARD.md) |
| Curation report | `data/processed/curation_report.json` |

### Corpus

| Metric | Value |
|---|---|
| Records harvested (OAI-PMH) | 13,998 |
| Final curated corpus | **10,006** |
| Train / dev / benchmark | 9,196 / 200 / **610** |
| Splits disjoint | **Yes** (0 overlap) |
| Benchmark meets NAIC 500+ | **Yes** |

### Benchmark results

*Populated after the training run — see `data/benchmark/naic_benchmark_summary.json`.*

---

## 13. Known limitations

We state these plainly, as the N-ATLaS terms require transparent reporting of
accuracy and limitations.

1. **Single-dominant institution.** The corpus is 99% University of Ibadan
   (9,132 of 9,196 training records). Redeemer's University contributes 64.
   Our attribution task therefore tests discrimination between two
   institutions, not the breadth of Nigerian higher education.
2. **Predominantly journal articles.** Only ~10% are theses or dissertations.
3. **Metadata quality.** Dublin Core subject terms are inconsistent — `Nigeria`
   is used as a subject alongside genuine subject headings.
4. **Abstracts only, not full texts.** Findings in a results section may not be
   present in the abstract.
5. **Self-distilled translations are imperfect.** The teacher rates Yoruba
   2.69/5. Quality must be audited by fluent speakers before any claim.
6. **No human evaluation.** All metrics are automatic. A Nigerian evaluator
   should review the audit sample.
7. **English stemming limits ROUGE** for non-English; we headline chrF there.

---

## 14. Licence compliance

The N-ATLaS Terms of Use bind this derivative. We comply as follows:

| Condition | How |
|---|---|
| Required attribution | Stated in every README, model card, API response and UI footer |
| "Powered by Awarri" suffix | Carried in the adapter name and product name |
| Same-licence distribution | Documented in [`ATTRIBUTION.md`](ATTRIBUTION.md) |
| 1,000 end-user cap | Demo is a competition artefact, not a public service |
| Transparent limitation reporting | Section 13 and `DATASET_CARD.md` §7 |
| No commercial deployment | Submitted as a competition entry |

---

## 15. Reproduction

```bash
pip install -r requirements.txt

# Phase 1 — discover live OAI endpoints
python scripts/probe_repositories.py

# Phase 2 — harvest, curate, split
python scripts/harvest_oai.py --limit 20000
python scripts/build_corpus.py --eval-size 600 --dev-size 200
python scripts/make_dataset_card.py

# Verification
python scripts/measure_seq_lengths.py    # justifies MAX_SEQ_LENGTH
python scripts/audit_design.py           # the flaws we found
python scripts/verify_fixes.py           # proves they are fixed

# Phases 3-5 — GPU (Kaggle T4 x2)
python scripts/self_distill_translations.py --limit 3000 --batch-size 16
python scripts/train_lumina_edu.py
python scripts/evaluate_benchmark.py --limit 500 --batch-size 16 \
    --adapter lumina-edu-natlas-adapter-powered-by-awarri

# Phase 6 — serving
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

---

*Prepared by Lighthouse Design Academy for NAIC 2026. All benchmarks are
generated by real model inference; no results in this document are simulated.*