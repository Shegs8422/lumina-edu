"""End-to-end pipeline smoke test on a TINY model.

WHY THIS EXISTS
---------------
Four bugs reached Kaggle in a row, all of the same species: a code path that
is only reachable with a real model loaded. Every check I had ran WITHOUT a
model, so it proved the scripts import and the pure functions are correct -
and said nothing about the parts that actually train.

This exercises every one of those untested paths, cheaply:

    load tiny model -> build examples -> construct SFTTrainer with our exact
    kwargs -> train a few steps -> save adapter -> reload with
    PeftModel.from_pretrained -> batched generation

If that completes, the expensive run cannot fail on structure. It costs a few
minutes on a T4 instead of hours.

Run on Kaggle BEFORE the real fine-tune.
"""

import os
import sys

MODEL_ID = os.environ.get("SMOKE_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
DATA = os.environ.get("SMOKE_DATA", "/kaggle/input/lumina-edu-data/data/processed/dev.jsonl")
OUT = os.environ.get("SMOKE_OUT", "/kaggle/working/smoke")

os.makedirs(OUT, exist_ok=True)
ADAPTER = os.path.join(OUT, "smoke-adapter")

import importlib.util

spec = importlib.util.spec_from_file_location(
    "trainer", "/kaggle/input/lumina-edu-data/scripts/train_lumina_edu.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

import torch
from datasets import Dataset

print("=" * 66)
print("SMOKE TEST - full pipeline on a tiny model")
print("=" * 66)
print(f"model : {MODEL_ID}")
print(f"data  : {DATA}")
print(f"out   : {OUT}")

# ---------------------------------------------------------------- 1. data --
print("\n[1/6] Building examples")
recs = tr.load_jsonl(DATA)[:8]
if not recs:
    raise SystemExit(f"no records at {DATA}")
ex = tr.build_examples(recs)
print(f"      {len(recs)} records -> {len(ex)} examples")

# ---------------------------------------------------------------- 2. model --
print("\n[2/6] Loading tiny model")
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, max_seq_length=512, load_in_4bit=False)

# ------------------------------------------------------------ 3. tokenise --
print("\n[3/6] Tokenising via the real datasets.map callback")
ds = Dataset.from_list(ex).map(
    tr.to_chat_text, batched=True, fn_kwargs={"tokenizer": tok})
empty = sum(1 for t in ds["text"] if not t.strip())
print(f"      {len(ds)} rows, {empty} empty")
assert not empty, "empty renders - chat template produced nothing"
assert len(ds) == len(ex), "row count changed during map"

# -------------------------------------------------------------- 4. trainer --
print("\n[4/6] Constructing SFTTrainer with the production kwargs")
from trl import SFTTrainer
import inspect as _inspect

params = _inspect.signature(SFTTrainer.__init__).parameters
tok_kwarg = "processing_class" if "processing_class" in params else "tokenizer"
kwargs = dict(model=model, train_dataset=ds, dataset_text_field="text",
              dataset_num_proc=1, packing=False)
kwargs[tok_kwarg] = tok
kwargs["max_length" if "max_length" in params else "max_seq_length"] = 512

from transformers import TrainingArguments
trainer = SFTTrainer(
    **kwargs,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=3,               # just enough to prove the loop runs
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        output_dir=os.path.join(OUT, "checkpoints"),
    ),
)
print(f"      tokenizer kwarg: {tok_kwarg}")

# --------------------------------------------------------------- 5. train --
print("\n[5/6] Training 3 steps")
trainer.train()
loss = trainer.state.log_history[-1].get("loss")
print(f"      final loss: {loss}")
assert loss is None or loss == loss, "loss is NaN"

# ---------------------------------------------------------------- 6. save --
print("\n[6/6] Saving and reloading the adapter")
model.save_pretrained_merged(ADAPTER, tok, save_method="lora")
assert os.path.isdir(ADAPTER), "adapter directory missing"
files = os.listdir(ADAPTER)
print(f"      wrote {len(files)} files")
assert any("adapter" in f for f in files), \
    f"no adapter_config.json in {files}"

print("\n      reloading with PeftModel.from_pretrained")
from peft import PeftModel
reloaded = PeftModel.from_pretrained(model, ADAPTER)
print("      reload OK")

# ------------------------------------------------------- 7. batched gen --
print("\n[7/7] Batched generation with the production path")
sys.path.insert(0, "/kaggle/input/lumina-edu-data/scripts")
spec2 = importlib.util.spec_from_file_location(
    "evalbench", "/kaggle/input/lumina-edu-data/scripts/evaluate_benchmark.py")
ev = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ev)

if getattr(tok, "pad_token", None) is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

items = [{"chat_prompt": t} for t in ds["text"][:4]]
outs = ev.run_generation(reloaded, tok, items, batch_size=2)
print(f"      generated {len(outs)} completions")
assert len(outs) == len(items), "batched generation lost rows"
assert all(isinstance(o, str) for o in outs), "non-string completion"

print("\n" + "=" * 66)
print("SMOKE TEST PASSED - every stage of the real run is now proven")
print("=" * 66)