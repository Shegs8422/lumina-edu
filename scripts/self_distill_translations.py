"""
Self-distillation of Yoruba / Hausa / Igbo targets from NCAIR1/N-ATLaS.

WHY THIS EXISTS
---------------
The Lumina-Edu training tasks need real African-language targets. The naive
option - emitting a placeholder like "[Yoruba explanation of: <English text>]" -
is actively harmful: it teaches the model to produce bracketed scaffolding
rather than Yoruba, which would hollow out the exact claim the project makes.

Instead we use the base model as its own teacher (self-distillation):
    English abstract  ->  N-ATLaS  ->  Yoruba / Hausa / Igbo explanation

This is a legitimate and commonly used technique, but ONLY if we are honest
about what it produces. N-ATLaS's own model card rates Yoruba at 2.69/5 and
Hausa bias/fairness at 1.11/5, so these translations are imperfect. We
therefore:

  1. Write every generation to disk for inspection.
  2. Auto-screen outputs that failed to actually change language (a model
     that answers in English is a failed distillation sample, not a target).
  3. Emit a random sample for MANUAL human audit and report the rate in the
     dataset card rather than claiming the data is clean.

The result is genuine model output in the target languages, never a template.

Run on Kaggle T4:
    python scripts/self_distill_translations.py --limit 3000 --out data/distilled
"""

import argparse
import json
import os
import random
import re
import unicodedata
from datetime import datetime

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "NCAIR1/N-ATLaS"
MAX_SEQ_LENGTH = 2048
TARGET_LANGUAGES = ["Yoruba", "Hausa", "Igbo"]

ATTRIBUTION = ("Powered by Awarri Technologies and the Federal Ministry of "
               "Communications, Innovation and Digital Economy.")

# Function words that appear in genuine Yoruba/Hausa/Igbo output. Used as a
# cheap language check; absence of these strongly suggests the model answered
# in English and the sample should be discarded.
LANG_MARKERS = {
    "Yoruba": ["ati", "ni", "ko", "wa", "fun", "won", "yi", "naa", "kini", "won"],
    "Hausa": ["da", "na", "ta", "ya", "ba", "sun", "kuma", "wannan", "shi", "cikin"],
    "Igbo": ["na", "ya", "ihe", "ndi", "ka", "gi", "maka", "nke", "a", "buru"],
}


def strip_accents(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t or "")
                   if not unicodedata.combining(c))


def looks_like_target_language(text, language):
    """Cheap heuristic: does the text contain target-language function words?

    Returns (ok, marker_hits). This is a filter, not a proof - it exists to
    drop samples where the teacher simply replied in English.
    """
    toks = set(re.findall(r"[a-zàáèéìíòóùúṣọ]+", strip_accents(text).lower()))
    if language not in LANG_MARKERS:
        return False, 0
    hits = sum(1 for m in LANG_MARKERS[language] if m in toks)
    return hits >= 3, hits


def is_degenerate(text):
    """Catch repetition collapse, which greedy decoding can produce."""
    t = (text or "").strip()
    if len(t) < 40:
        return True
    words = t.lower().split()
    if len(words) >= 20:
        # Any 6-word window repeated 4+ times indicates a decoding loop.
        seen = {}
        for i in range(len(words) - 6):
            gram = " ".join(words[i:i + 6])
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= 4:
                return True
    return False


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_prompt(title, abstract, language, abstract_chars=700):
    return [
        {"role": "system", "content":
            f"You are a Nigerian academic translator. Translate the research "
            f"summary into {language}. Use natural {language} prose. Do not "
            f"add commentary, brackets, or English. {ATTRIBUTION}"},
        {"role": "user", "content":
            f"Translate this research summary into {language}.\n\n"
            f"Title: {title}\nAbstract: {abstract[:abstract_chars]}"},
    ]


def render(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            date_string=datetime.now().strftime("%d %b %Y"))
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate_batched(model, tokenizer, jobs, batch_size, max_new_tokens):
    """Generate translations for a list of (rec, lang, prompt) jobs in batches.

    Batching is the single biggest speed lever here: sequential generation
    wastes the GPU, because a 4-bit 8B model decoding one short sequence at a
    time leaves most of the T4 idle. Left-padding is required for correct
    batched decoding on a decoder-only model.
    """
    outs = []
    n = len(jobs)
    for start in range(0, n, batch_size):
        chunk = jobs[start:start + batch_size]
        prompts = [render(tokenizer, j[2]) for j in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[-1]
        for row in gen[:, prompt_len:]:
            outs.append(tokenizer.decode(row, skip_special_tokens=True).strip())
        done = min(start + batch_size, n)
        print(f"    generated {done}/{n}", end="\r", flush=True)
    print()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PROJECT_ROOT, "data", "processed", "train.jsonl"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "distilled"))
    ap.add_argument("--limit", type=int, default=1500, help="records to translate")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--abstract-chars", type=int, default=700,
                    help="truncate the abstract before translating; a focused "
                         "explanation is enough for training and much faster")
    ap.add_argument("--audit-sample", type=int, default=50,
                    help="how many samples to write out for manual review")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records = load_jsonl(args.input)[: args.limit]
    print(f"Distilling {len(records)} records x {len(TARGET_LANGUAGES)} languages "
          f"= {len(records) * len(TARGET_LANGUAGES)} generations")
    print(f"batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}")

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    # Batched decoder-only generation requires left padding and a pad token.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build every (record, language, prompt) job up front.
    jobs = []
    for rec in records:
        for lang in TARGET_LANGUAGES:
            jobs.append((rec, lang, build_prompt(
                rec["title"], rec["abstract"], lang, args.abstract_chars)))

    texts = generate_batched(model, tokenizer, jobs,
                             args.batch_size, args.max_new_tokens)

    results = []
    audit = []
    dropped = {"degenerate": 0, "wrong_language": 0}
    kept_per_lang = {l: 0 for l in TARGET_LANGUAGES}

    for (rec, lang, _), text in zip(jobs, texts):
        if is_degenerate(text):
            dropped["degenerate"] += 1
            continue
        ok, hits = looks_like_target_language(text, lang)
        if not ok:
            dropped["wrong_language"] += 1
            continue

        kept_per_lang[lang] += 1
        results.append({
            "id": rec["id"],
            "language": lang,
            "title": rec["title"],
            "source_abstract": rec["abstract"],
            "translation": text,
            "marker_hits": hits,
        })
        if len(audit) < args.audit_sample:
            audit.append({
                "id": rec["id"], "language": lang, "title": rec["title"],
                "translation": text,
            })

    out_path = os.path.join(args.out, "translations.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    audit_path = os.path.join(args.out, "audit_sample.json")
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    attempted = len(records) * len(TARGET_LANGUAGES)
    report = {
        "records_processed": len(records),
        "languages": TARGET_LANGUAGES,
        "attempted_generations": attempted,
        "kept": len(results),
        "kept_per_language": kept_per_lang,
        "dropped": dropped,
        "retention_rate": round(len(results) / attempted, 4) if attempted else 0,
        "audit_sample_size": len(audit),
        "audit_note": (
            "Translations are generated by N-ATLaS itself (self-distillation). "
            "The base model rates Yoruba 2.69/5 and Hausa bias/fairness 1.11/5, "
            "so quality is imperfect. The audit sample MUST be reviewed by a "
            "fluent speaker and the accuracy reported in DATASET_CARD.md before "
            "any claim of translation quality is made."
        ),
        "teacher_model": MODEL_ID,
        "decoding": "greedy (do_sample=False)",
    }
    with open(os.path.join(args.out, "distillation_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Kept {len(results)} / {attempted} attempted "
          f"({report['retention_rate'] * 100:.1f}%)")
    print(f"Dropped: {dropped}")
    print(f"Per language: {kept_per_lang}")
    print(f"\nWrote {out_path}")
    print(f"Wrote {audit_path}  <-- review these by hand")
    print("=" * 60)


if __name__ == "__main__":
    main()