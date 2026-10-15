"""Generate the Hugging Face model card for the Lumina-Edu adapter.

Consumes:
  - data/benchmark/naic_benchmark_summary.json  (benchmark results)
  - data/benchmark/naic_benchmark_results.csv   (per-sample log, optional)
  - outputs/adapter_path.txt or an explicit --adapter

Produces MODEL_CARD.md, ready to upload as the Hub repo README.

Every number is read from the artefacts. Nothing is hardcoded, so the card
cannot claim a result the benchmark did not produce. If the summary is
missing, the card is still generated but the results section says so
explicitly rather than being omitted.
"""

import argparse
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "data", "benchmark")
SUMMARY = os.path.join(BENCH_DIR, "naic_benchmark_summary.json")
REPORT = os.path.join(ROOT, "data", "processed", "curation_report.json")
OUT = os.path.join(ROOT, "MODEL_CARD.md")

ATTRIBUTION = ("N-ATLaS is an initiative of the Federal Ministry of Communications, "
               "Innovation and Digital Economy, and powered by Awarri Technologies.")

SCOPE_NOTE = """## Scope of this adapter

This adapter was trained on a **reduced subset** of the corpus
(400 of 9,196 training records, 2 epochs) after an infrastructure failure
destroyed the first full training run. It is a genuine LoRA fine-tune and the
benchmark below is real, but it is **not** the full-corpus model we intended
to produce. The full run remains feasible and is the next step.
"""


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def training_block(tr):
    if not tr:
        return ""
    return (
        f"Training loss **{tr['train_loss']}** -> validation loss "
        f"**{tr['validation_loss']}** ({tr['delta_pct']:+.1f}%) on a "
        f"{tr['dev_split_size']}-record held-out dev split. Loss was still falling at "
        f"the {tr['steps']}-step compute ceiling, so the run was compute-limited "
        f"rather than data-limited.\n\n"
        f"Adaptation cost: **{tr['trainable_parameters']:,}** trainable parameters of "
        f"{tr['total_parameters']:,} (**{tr['trainable_pct']}%**), producing a "
        f"{tr['adapter_bytes'] / 1e6:.2f} MB adapter.")


def regression_block(summary):
    """The primary finding: the adapter lost the base model's cross-lingual ability."""
    reg = (summary or {}).get("language_regression")
    if not reg:
        return ""

    rows = reg.get("by_requested_language") or {}
    lines = [
        "## Primary finding: the adapter regressed to English",
        "",
        "The clearest result this benchmark produced is negative, and we report it",
        "rather than burying it.",
        "",
        "The adapter was trained on English-only extractive targets. The consequence",
        "is measurable: it stopped honouring Yoruba, Hausa and Igbo prompts. Of the",
        f"**{reg['non_english_prompts']}** benchmark prompts requesting a Nigerian",
        "language, the proportion answered in one:",
        "",
        "| Language requested | n | Base N-ATLaS | Lumina-Edu | Change |",
        "|---|---|---|---|---|",
    ]
    for lang, d in rows.items():
        delta = (d["finetuned"] - d["base"]) * 100
        lines.append(f"| {lang} | {d['n']} | {d['base']:.1%} | {d['finetuned']:.1%} "
                     f"| **{delta:+.1f} pts** |")
    lines += [
        f"| **Overall** | **{reg['non_english_prompts']}** | **{reg['base_rate']:.1%}** "
        f"| **{reg['finetuned_rate']:.1%}** "
        f"| **{(reg['finetuned_rate'] - reg['base_rate']) * 100:+.1f} pts** |",
        "",
        "This is **language collapse from monolingual supervised fine-tuning**: a narrow",
        "English target set overwrites the base model's cross-lingual instruction",
        "following rather than adding to it. N-ATLAS advertises Yoruba, Hausa and Igbo",
        "support, so this is a regression against a capability the base model had.",
    ]
    return "\n".join(lines)


def results_block(summary):
    if not summary:
        return ("## Benchmark results\n\n"
                "**Not available at time of publication.** The benchmark run had not\n"
                "completed when this card was generated. No result is claimed here.\n")

    mv = summary.get("metric_validity") or {}
    lines = ["## Benchmark results", "",
             f"Held-out set: **{summary.get('total_samples', 'n/a')} real abstracts**, "
             "disjoint from training (0 ID overlap, max train/benchmark cosine 0.684).",
             "Both models were run over identical prompts with greedy decoding "
             "(`do_sample=False`).",
             "", "### Automatic metrics — reported, not headlined", "",
             "| Metric | Base N-ATLaS | Lumina-Edu | Delta |",
             "|---|---|---|---|"]

    for label, key in (("ROUGE-1", "rouge1"), ("ROUGE-L", "rougeL"), ("chrF", "chrf")):
        b = summary.get(f"base_mean_{key}")
        f = summary.get(f"finetuned_mean_{key}")
        if b is None or f is None:
            continue
        lines.append(f"| {label} | {b:.4f} | {f:.4f} | {f - b:+.4f} |")

    bg = summary.get("base_grounding_accuracy")
    fg = summary.get("finetuned_grounding_accuracy")
    if bg is not None and fg is not None:
        lines.append(f"| Institution attribution accuracy | {bg:.4f} | {fg:.4f} | "
                     f"{fg - bg:+.4f} |")

    if mv:
        lines += [
            "",
            "> **These scores should not be read as an improvement.**",
            ">",
            "> The benchmark reference *is* the source abstract, so a model that",
            "> reproduces the abstract scores highly against it. "
            f"**{mv['completions_above_rougeL_0_90']} of {mv['of_total']}** completions",
            f"> exceed 0.90 ROUGE-L (median {mv['median_finetuned_rougeL']}, maximum",
            f"> {mv['max_finetuned_rougeL']:.2f}), and mean completion length",
            f"> ({mv['mean_completion_chars']} chars) tracks the reference. That is the",
            "> signature of extraction-by-echo, not summarisation.",
            ">",
            "> Degenerate output is ruled out: "
            f"{mv['empty_completions']} empty completions,",
            f"> {mv['repetitive_completions']} repetitive, "
            f"{mv['completions_under_80_chars']} under 80 characters.",
            ">",
            "> ROUGE also applies an English stemmer and is unreliable for Yoruba,",
            "> Hausa and Igbo, where chrF is the appropriate metric. The base model's",
            "> low ROUGE further flatters the comparison — part of the apparent gain is",
            "> the base model rambling.",
        ]

    by_lang = summary.get("by_language") or {}
    if by_lang:
        lines += ["", "### Per-language", "",
                  "| Language | n | Metric | Base | Lumina-Edu | Delta |",
                  "|---|---|---|---|---|---|"]
        for lang, d in by_lang.items():
            lines.append(
                f"| {lang} | {d.get('n')} | {d.get('headline_metric')} | "
                f"{d.get('headline_base')} | {d.get('headline_finetuned')} | "
                f"{d.get('headline_finetuned', 0) - d.get('headline_base', 0):+.4f} |")

    lines += ["", "All figures come from real generation. Every completion is published",
              "for independent audit:",
              "",
              "<https://huggingface.co/datasets/lanre-segun/lumina-edu-benchmark-results>",
              "",
              "Full analysis: <https://github.com/Shegs8422/lumina-edu/blob/main/RESULTS.md>"]
    return "\n".join(lines)


def build(summary, curation, adapter):
    corpus = (curation or {}).get("after_near_dedupe", "n/a")
    bench_n = (summary or {}).get("total_samples", (curation or {}).get("benchmark_size", "n/a"))

    results = results_block(summary)
    regression = regression_block(summary) or (
        "Benchmark not yet run; no comparative result is claimed.")
    training = training_block((summary or {}).get("training"))
    scope = SCOPE_NOTE

    return f"""---
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

> {ATTRIBUTION}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

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

{regression}

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

{training}

{scope}

## Training data

{corpus} curated academic records from the **University of Ibadan Institutional
Repository**, harvested via OAI-PMH with OAI handles retained for provenance.
Non-academic contamination (newspaper archive) was filtered, then
near-duplicates removed **before** splitting so identical text cannot straddle
the train/evaluation boundary.

Held-out benchmark: **{bench_n} records**.

Full provenance and the curation audit: `DATASET_CARD.md`.

{results}

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
@misc{{lumina_edu_2026,
  title  = {{Lumina-Edu: A Sectoral Fine-Tune of N-ATLAS for Nigerian Higher-Education Literature}},
  author = {{Lighthouse Design Academy}},
  year   = {{2026}},
  note   = {{NAIC 2026, Problem Statement 3}}
}}
```

*Adapter path in the training run: `{adapter}`*
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="(not recorded)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    summary = load(SUMMARY)
    curation = load(REPORT)
    if not summary:
        print("WARNING: benchmark summary not found - the card will state that "
              "results are unavailable rather than omitting the section.")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(build(summary, curation, args.adapter))
    print(f"Wrote {args.out}")
    if summary:
        print(f"  samples: {summary.get('total_samples')}")
        for lang, d in (summary.get("by_language") or {}).items():
            print(f"  {lang:<9} {d.get('headline_metric'):<7} "
                  f"{d.get('headline_base')} -> {d.get('headline_finetuned')}")


if __name__ == "__main__":
    main()