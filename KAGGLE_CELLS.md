# Lumina-Edu — Kaggle notebook cells

Paste these into your Kaggle notebook in order, top to bottom.
They mirror https://github.com/Shegs8422/lumina-edu/blob/main/kaggle_train_lumina_edu.ipynb

**Before running:** Session options → Accelerator **GPU T4 ×2**, Internet **On**.
Add-ons → Secrets → `HF_TOKEN` (tick the checkbox).

---

## Cell 1 — markdown (optional, for orientation)

```markdown
# Lumina-Edu - QLoRA fine-tune of NCAIR1/N-ATLaS (Powered by Awarri)

NAIC 2026 | Problem Statement 3: Sectoral Fine-Tuning (Education & Learning)

Stages: 0 auth -> 1 language smoke test -> 2 self-distillation -> 3 assemble -> 4 fine-tune

The token is read only via UserSecretsClient().get_secret(); it is never written
into a cell, printed, or committed. Stage 0 prints only the account *name*.
```

---

## Cell 2 — Stage 0: auth and gated-access check  (RUN THIS FIRST)

```python
import os, sys, json, subprocess, importlib.util

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
        "HF_TOKEN not found.\n\n"
        "N-ATLaS is a GATED repo - it cannot be downloaded anonymously.\n"
        "Kaggle Secrets are per-notebook, not in account Settings:\n\n"
        "  1. In THIS notebook: Add-ons -> Secrets -> Add new secret\n"
        "  2. Label  : HF_TOKEN\n"
        "  3. Value  : your Hugging Face access token\n"
        "  4. Tick the checkbox beside HF_TOKEN to attach it to the session\n\n"
        "There is nothing at kaggle.com/settings/secrets - that URL is a 404.\n"
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
        f"No access to NCAIR1/N-ATLaS: {type(e).__name__}\n"
        "Request and accept the licence at "
        "https://huggingface.co/NCAIR1/N-ATLaS"
    )
```

---

## Cell 3 — install packages

```python
!pip install -q unsloth
!pip install -q unsloth_zoo
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q rouge-score scikit-learn lxml pandas

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f}GB")
assert torch.cuda.is_available(), "Enable GPU in notebook settings"
```

---

## Cell 4 — locate the dataset

```python
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
assert os.path.exists(os.path.join(DATA_ROOT, "data", "processed", "train.jsonl")), \
    "train.jsonl not found - attach the lumina-edu-data dataset"
print("train.jsonl OK:", os.path.getsize(
    os.path.join(DATA_ROOT, "data", "processed", "train.jsonl")) // 1024, "KB")
```

---

## Cell 5 — Stage 1: language smoke test  (RUN THIS, SEND ME OUTPUT)

```python
src = open(os.path.join(DATA_ROOT, "scripts/smoke_test_language.py")).read()
exec(compile(src, "smoke_test_language.py", "exec"))
```

---

## Cell 6 — Stage 2: self-distillation  **<< THIS IS THE CELL TO EDIT**

```python
# Batched generation (batch_size 16) cut this from ~13 h to ~3 h.
#   1500  ~4,500 generations, ~3,000 usable targets, ~1.5-2 h   <- safe default
#   3000  ~9,000 generations, ~6,000 targets,        ~3-4 h   <- recommended now
#   600   ~1,800 generations, ~1,200 targets,        ~45-60 min
DISTILL_LIMIT = 3000
BATCH_SIZE = 16      # 16 fits comfortably in 4-bit on a single T4

rc = subprocess.run([
    sys.executable,
    os.path.join(DATA_ROOT, "scripts/self_distill_translations.py"),
    "--limit", str(DISTILL_LIMIT),
    "--batch-size", str(BATCH_SIZE),
    "--max-new-tokens", "200",
    "--abstract-chars", "700",
    "--input", os.path.join(DATA_ROOT, "data/processed/train.jsonl"),
    "--out", os.path.join(DATA_ROOT, "data/distilled"),
]).returncode
print("distill exit code:", rc)
```

---

## Cell 7 — inspect distillation report

```python
rep_path = os.path.join(DATA_ROOT, "data/distilled/distillation_report.json")
if os.path.exists(rep_path):
    print(json.dumps(json.load(open(rep_path)), indent=2, ensure_ascii=False))
    print("\n>>> REVIEW data/distilled/audit_sample.json BEFORE training.")
    print(">>> These are teacher outputs; a fluent speaker must check quality.")
else:
    print("No distillation report - falling back to English-only training.")
```

---

## Cell 8 — Stage 3: load trainer and distilled data

```python
spec = importlib.util.spec_from_file_location(
    "trainer", os.path.join(DATA_ROOT, "scripts/train_lumina_edu.py"))
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

DISTILLED = os.path.join(DATA_ROOT, "data/distilled/translations.jsonl")
distilled = trainer.load_jsonl(DISTILLED) if os.path.exists(DISTILLED) else []
print(f"distilled translations: {len(distilled)}")
if distilled:
    from collections import Counter
    print(Counter(d["language"] for d in distilled))
```

---

## Cell 9 — Stage 4: QLoRA fine-tune

```python
trainer.main()
```

---

## Cell 10 — benchmark (run after training)

```python
!python {DATA_ROOT}/scripts/evaluate_benchmark.py \
  --limit 500 \
  --batch-size 16 \
  --adapter /kaggle/working/lumina-edu-natlas-adapter-powered-by-awarri
```

Batched generation takes this from ~3 h to ~25 min.

---

## Time budget (T4 x2, 4-bit)

| Stage | Before batching | After |
|---|---|---|
| Stage 1 smoke test | 5 min | 5 min |
| Stage 2 distill 3000 | 9-13 h | **3-4 h** |
| Stage 4 train | ~4 h | **~2 h** |
| Benchmark 500x2 | ~3 h | **~25 min** |
| **Total** | **~16-20 h** | **~6 h** |

`MAX_SEQ_LENGTH` is 1024 (was 2048). Measured on 3,000 real examples: median
543 tokens, p90 931, so 1024 truncates only 5.1% of examples for a 50% compute
saving. Re-run `scripts/measure_seq_lengths.py` to reproduce.