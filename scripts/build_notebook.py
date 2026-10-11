"""Generate the Kaggle notebook as valid .ipynb JSON.

Writing JSON by hand is error-prone and the notebook must actually run on
Kaggle (IPython magics, exec() of helper scripts, subprocess calls), so we
build it programmatically instead.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = [
md("""# Lumina-Edu - QLoRA fine-tune of NCAIR1/N-ATLaS (Powered by Awarri)

**NAIC 2026 | Problem Statement 3: Sectoral Fine-Tuning (Education & Learning)**

Base model: [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS)

This notebook runs the full training pipeline in four stages:

1. **Environment + authentication**
2. **Language capability smoke test** - verifies the model can actually produce
   Yoruba/Hausa/Igbo before we spend GPU hours on it
3. **Self-distillation** - generates genuine translation targets with the base
   model itself
4. **QLoRA fine-tune**

### Before running

**1. Enable hardware** — Session options → Accelerator **GPU T4 x2**, Internet **On**.

**2. Attach the dataset** — Add Input → `oluwasegunolanrewaju/lumina-edu-data`.

**3. Supply the Hugging Face token as a Kaggle Secret**

N-ATLaS is a **gated** repository and cannot be downloaded anonymously, so the
token is required to run this notebook at all.

> **Add-ons → Secrets → Add new secret**
> - Label: `HF_TOKEN`
> - Value: your Hugging Face access token
> - Tick the checkbox beside `HF_TOKEN` to attach it to this session

Kaggle Secrets are **per-notebook**; they do not appear in account Settings
(`kaggle.com/settings/secrets` returns 404).

**Credential handling in this notebook:** the token is only ever read through
`UserSecretsClient().get_secret()`, which returns the decrypted value at
runtime. It is never written into a cell, printed, logged, or committed, and
Kaggle masks it in cell output. Stage 0 deliberately prints only the account
*name* returned by `whoami()`, never the credential.

This is the only supported way to supply the token. Do not paste it into a
cell — notebook source is public.
"""),

md("## Stage 0 - Environment and authentication"),

code("""import os, sys, json, subprocess, importlib.util

def get_hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("KAGGLE_HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None

HF_TOKEN = get_hf_token()

if not HF_TOKEN:
    raise SystemExit(
        "HF_TOKEN not found.\\n\\n"
        "N-ATLaS is a GATED repo - it cannot be downloaded anonymously.\\n"
        "Kaggle Secrets are per-notebook, not in account Settings:\\n\\n"
        "  1. In THIS notebook: Add-ons -> Secrets -> Add new secret\\n"
        "  2. Label  : HF_TOKEN\\n"
        "  3. Value  : your Hugging Face access token\\n"
        "  4. Tick the checkbox beside HF_TOKEN to attach it to the session\\n\\n"
        "There is nothing at kaggle.com/settings/secrets - that URL is a 404.\\n"
        "Do not paste the token into a cell; notebook source is public."
    )

from huggingface_hub import login, whoami
login(token=HF_TOKEN, add_to_git_credential=False)
print("Logged in as:", whoami()["name"])

# Confirm gated access BEFORE spending GPU time on a 16 GB download.
from huggingface_hub import hf_hub_download
try:
    hf_hub_download("NCAIR1/N-ATLaS", "config.json")
    print("Gated access to NCAIR1/N-ATLaS: CONFIRMED")
except Exception as e:
    raise SystemExit(
        f"No access to NCAIR1/N-ATLaS: {type(e).__name__}\\n"
        "Request and accept the licence at "
        "https://huggingface.co/NCAIR1/N-ATLaS"
    )"""),

code("""!pip install -q unsloth
!pip install -q unsloth_zoo
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q rouge-score scikit-learn lxml pandas

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f}GB")
assert torch.cuda.is_available(), "Enable GPU in notebook settings\""""),

code("""import glob
DATA_ROOT = None
for cand in ["/kaggle/input/lumina-edu-data",
             "/kaggle/input/lumina-edu-nigerian-academic-corpus",
             "/kaggle/working"]:
    if os.path.exists(os.path.join(cand, "scripts", "train_lumina_edu.py")):
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    print("Project dataset not attached. Falling back to /kaggle/working.")
    DATA_ROOT = "/kaggle/working"
print("DATA_ROOT =", DATA_ROOT)
assert os.path.exists(os.path.join(DATA_ROOT, "data", "processed", "train.jsonl")), \\
    "train.jsonl not found - attach the lumina-edu-data dataset"
print("train.jsonl OK:", os.path.getsize(
    os.path.join(DATA_ROOT, "data", "processed", "train.jsonl")) // 1024, "KB")"""),

md("""## Stage 1 - Language capability smoke test

The base model's own card rates Yoruba at **2.69/5**. If it cannot produce
target-language text at all, self-distillation is not viable and we must fall
back to English-only training. This check costs ~2 minutes and prevents
committing hours of GPU time to a doomed approach."""),

code("""src = open(os.path.join(DATA_ROOT, "scripts/smoke_test_language.py")).read()
exec(compile(src, "smoke_test_language.py", "exec"))"""),

md("""## Stage 2 - Self-distillation

Generates real Yoruba/Hausa/Igbo targets using N-ATLaS itself as teacher.
Output is filtered: samples where the model replied in English or collapsed
into repetition are discarded rather than trained on.

`--limit 3000` costs roughly 4-6 GPU hours. Reduce it if the deadline is tight."""),

code("""DISTILL_LIMIT = 3000  # reduce if short on time

rc = subprocess.run([
    sys.executable,
    os.path.join(DATA_ROOT, "scripts/self_distill_translations.py"),
    "--limit", str(DISTILL_LIMIT),
    "--input", os.path.join(DATA_ROOT, "data/processed/train.jsonl"),
    "--out", os.path.join(DATA_ROOT, "data/distilled"),
]).returncode
print("distill exit code:", rc)"""),

code("""rep_path = os.path.join(DATA_ROOT, "data/distilled/distillation_report.json")
if os.path.exists(rep_path):
    print(json.dumps(json.load(open(rep_path)), indent=2, ensure_ascii=False))
    print("\\n>>> REVIEW data/distilled/audit_sample.json BEFORE training.")
    print(">>> These are teacher outputs; a fluent speaker must check quality.")
else:
    print("No distillation report - falling back to English-only training.")"""),

md("## Stage 3 - Load trainer module and distilled data"),

code("""spec = importlib.util.spec_from_file_location(
    "trainer", os.path.join(DATA_ROOT, "scripts/train_lumina_edu.py"))
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

DISTILLED = os.path.join(DATA_ROOT, "data/distilled/translations.jsonl")
distilled = trainer.load_jsonl(DISTILLED) if os.path.exists(DISTILLED) else []
print(f"distilled translations: {len(distilled)}")
if distilled:
    from collections import Counter
    print(Counter(d["language"] for d in distilled))"""),

md("## Stage 4 - QLoRA fine-tune"),

code("""trainer.main()"""),

md("""## Next steps

Run the benchmark (500+ samples, real generations vs base model):

```bash
python scripts/evaluate_benchmark.py --limit 500 \\
    --adapter lumina-edu-natlas-adapter-powered-by-awarri
```

Artefacts: `data/benchmark/naic_benchmark_results.csv` and
`naic_benchmark_summary.json`.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(ROOT, "kaggle_train_lumina_edu.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print(f"Wrote {out} ({len(cells)} cells)")