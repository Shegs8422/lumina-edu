"""
Resumable, streaming benchmark runner for Lumina-Edu.

Why this exists
---------------
`evaluate_benchmark.py` holds both models' completions in memory and writes
the CSV only at the very end. Twice now a run has been lost at that final
write - once to CUDA OOM. With ~3 hours left, losing a 60-minute run to a
single crash is not an acceptable failure mode.

This runner fixes the persistence model:

  * Streams each batch's completions straight to JSONL.
  * Checkpoints after every batch.
  * On restart, skips prompts already recorded (resume).
  * Never holds more than one batch in memory.

Two passes are run separately (base, then fine-tuned) so each is
independently resumable and independently valuable. If the second pass
dies, the first pass's data is still on disk.

Usage on Colab:
    python run_benchmark_resumable.py --pass base
    python run_benchmark_resumable.py --pass finetuned
    python run_benchmark_resumable.py --score
"""

import argparse
import json
import os
import time

import torch
from huggingface_hub import hf_hub_download

MODEL_ID = "NCAIR1/N-ATLaS"
ADAPTER_DIR = "/content/outputs/adapter"
DATA_DIR = "/content/data/processed"
OUT_DIR = "/data/benchmark"
LANGUAGES = ["English", "Yoruba", "Hausa", "Igbo"]
MAX_NEW = 256
BATCH = 4

os.makedirs(OUT_DIR, exist_ok=True)


def sysout(*a):
    print(*a, flush=True)


def load_bench():
    p = os.path.join(DATA_DIR, "benchmark_eval.jsonl")
    if not os.path.exists(p):
        os.makedirs(DATA_DIR, exist_ok=True)
        hf_hub_download(
            "lanre-segun/lumina-edu-nigerian-academic-corpus",
            "benchmark_eval.jsonl", repo_type="dataset", local_dir=DATA_DIR)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_prompts(records, tok):
    from datetime import datetime
    out = []
    for i, r in enumerate(records):
        lang = LANGUAGES[i % len(LANGUAGES)]
        if lang == "English":
            user = (f"Summarise the main finding of this research.\n\n"
                    f"Title: {r['title']}\nAbstract: {r['abstract']}")
        else:
            user = (f"Explain the main finding of this research in {lang}.\n\n"
                    f"Title: {r['title']}\nAbstract: {r['abstract']}")
        msgs = [
            {"role": "system", "content":
                "You are Lumina-Edu, an academic research assistant for Nigerian "
                "higher education. Base model: NCAIR1/N-ATLaS, powered by Awarri "
                "Technologies and the Federal Ministry of Communications, "
                "Innovation and Digital Economy."},
            {"role": "user", "content": user},
        ]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                date_string=datetime.now().strftime("%d %b %Y"))
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        out.append({"id": r["id"], "lang": lang, "institution": r.get("institution", ""),
                    "reference": r["abstract"], "prompt": user, "chat": text})
    return out


def run_pass(items, model, tok, tag):
    """Generate with per-batch persistence and resume."""
    path = os.path.join(OUT_DIR, f"{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
        sysout(f"[{tag}] resuming: {len(done)} already done")

    todo = [it for it in items if it["id"] not in done]
    if not todo:
        sysout(f"[{tag}] already complete ({len(done)})")
        return

    t0 = time.time()
    fh = open(path, "a", encoding="utf-8")
    try:
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            enc = tok([c["chat"] for c in chunk], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=2048).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=MAX_NEW, do_sample=False,
                    repetition_penalty=1.1, use_cache=True,
                    pad_token_id=tok.pad_token_id, max_length=None)
            plen = enc["input_ids"].shape[-1]
            for c, row in zip(chunk, gen[:, plen:]):
                txt = tok.decode(row, skip_special_tokens=True).strip()
                fh.write(json.dumps({
                    "id": c["id"], "lang": c["lang"],
                    "institution": c["institution"],
                    "reference": c["reference"],
                    "completion": txt}, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n = start + len(chunk)
            rate = n / max(time.time() - t0, 1e-6)
            eta = (len(todo) - n) / max(rate, 1e-9)
            sysout(f"[{tag}] {n}/{len(todo)} | {rate*60:.1f}/min | ETA {eta/60:.1f} min")
            del gen, enc
            torch.cuda.empty_cache()
    finally:
        fh.close()
    sysout(f"[{tag}] PASS COMPLETE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", choices=["base", "finetuned"], required=False)
    ap.add_argument("--limit", type=int, default=610)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    records = load_bench()[: args.limit]
    sysout(f"records: {len(records)}")

    if not args.score:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID, max_seq_length=2048, load_in_4bit=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if args.which == "finetuned":
            from peft import PeftModel
            sysout(f"attaching adapter from {ADAPTER_DIR}")
            model = PeftModel.from_pretrained(model, ADAPTER_DIR)
            model.eval()
            tag = "finetuned"
        else:
            FastLanguageModel.for_inference(model)
            tag = "base"

        items = build_prompts(records, tok)
        run_pass(items, model, tok, tag)
        sysout("PASS DONE - now run the other pass, then --score")
        return

    # ---- scoring ----
    import sys as _s
    _s.path.insert(0, "/content")
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Shegs8422/lumina-edu/main/scripts/evaluate_benchmark.py",
        "/content/evaluate_benchmark.py")
    import importlib
    import evaluate_benchmark as ev
    import pandas as pd

    base = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(OUT_DIR, "base.jsonl"), encoding="utf-8")}
    ft = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(OUT_DIR, "finetuned.jsonl"), encoding="utf-8")}
    common = [i for i in base if i in ft]
    sysout(f"scoring {len(common)} paired samples")

    rows = []
    for i in common:
        b, f = base[i], ft[i]
        ref = b["reference"]
        rows.append({
            "sample_id": i, "target_language": b["lang"],
            "base_rouge1": round(ev.scorer.score(ref, b["completion"])["rouge1"].fmeasure, 4),
            "base_rougeL": round(ev.scorer.score(ref, b["completion"])["rougeL"].fmeasure, 4),
            "ft_rouge1": round(ev.scorer.score(ref, f["completion"])["rouge1"].fmeasure, 4),
            "ft_rougeL": round(ev.scorer.score(ref, f["completion"])["rougeL"].fmeasure, 4),
            "base_chrf": round(ev.chrf(b["completion"], ref), 4),
            "ft_chrf": round(ev.chrf(f["completion"], ref), 4),
            "base_attribution": ev.attribution_hit(b["completion"], b["institution"]),
            "ft_attribution": ev.attribution_hit(f["completion"], f["institution"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "naic_benchmark_results.csv"), index=False)

    summary = {"total_samples": len(df)}
    for k in ["rouge1", "rougeL", "chrf"]:
        summary[f"base_mean_{k}"] = round(float(df[f"base_{k}"].mean()), 4)
        summary[f"finetuned_mean_{k}"] = round(float(df[f"ft_{k}"].mean()), 4)
    for c, n in (("base_attribution", "base"), ("ft_attribution", "finetuned")):
        sub = df[df[c].notna()]
        summary[f"{n}_grounding_accuracy"] = round(float(sub[c].mean()), 4) if len(sub) else None
    per = {}
    for lang, g in df.groupby("target_language"):
        m = "rougeL" if lang == "English" else "chrf"
        per[lang] = {"n": int(len(g)), "headline_metric": m,
                     "headline_base": round(float(g[f"base_{m}"].mean()), 4),
                     "headline_finetuned": round(float(g[f"ft_{m}"].mean()), 4)}
    summary["by_language"] = per
    with open(os.path.join(OUT_DIR, "naic_benchmark_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    sysout("SCORING COMPLETE")
    sysout(json.dumps({k: v for k, v in summary.items() if k != "by_language"}, indent=2))
    for lang, d in per.items():
        sysout(f"  {lang:<9} n={d['n']:<4} {d['headline_metric']:<6} "
               f"{d['headline_base']} -> {d['headline_finetuned']}")


if __name__ == "__main__":
    main()