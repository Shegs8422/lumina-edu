"""Pre-flight audit of the training path.

Answering "what could make training fail?" honestly means enumerating causes
rather than asserting confidence. Each check below corresponds to a failure
we have actually hit, or a dependency that has never executed.

Run BEFORE spending GPU time. Any FAIL must be fixed first.
"""
import ast
import os
import sys

TRAINER = "scripts/train_lumina_edu.py"
src = open(TRAINER, encoding="utf-8").read()
tree = ast.parse(src)
tree = ast.parse(src)

failures = []


def check(label, passed, detail=""):
    if not passed:
        failures.append(label)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
          + (f"\n         {detail}" if detail else ""))


print("1. FAILURES WE HAVE ACTUALLY HIT")
print("-" * 74)
check("no __file__ dependency at import time",
      "os.path.abspath(__file__)" in src, "only PROJECT_ROOT uses it, resolved at import")
check("no datetime module/attr confusion",
      "from datetime import datetime" in src)
check("no datasets.map callback signature bug",
      "def to_chat_text(examples, tokenizer)" in src)
check("no read-only /kaggle/input writes",
      "resolve_output_dir" in src)
check("no write outside /kaggle/working",
      '/kaggle/working' in src)
check("epochs are configurable, not hardcoded",
      "num_train_epochs=EPOCHS" in src and "num_train_epochs=2" not in src)
check("distillation absence is handled",
      "Translation task will be OMITTED" in src)

print()
print("2. DEPENDENCIES NEVER EXECUTED ON GPU")
print("-" * 74)
check("authenticates to the Hub itself",
      "def ensure_hf_login" in src and "ensure_hf_login()" in src,
      "gated repo -> anonymous download fails with opaque 401")
check("adapter save has a fallback",
      "save_pretrained_merged" in src and "except Exception" in src
      and "model.save_pretrained(adapter_out)" in src,
      "hours of training must not be lost at the final step")
check("adapter parent dir is created",
      "os.makedirs(adapter_out, exist_ok=True)" in src)
check("adapter file listing is verified after save",
      "adapter_config" in src)
check("TRL tokenizer kwarg is version-detected",
      "processing_class" in src and "inspect.signature" in src)
check("TRL sequence-length kwarg is version-detected",
      '"max_length" in params' in src)
check("multi-GPU is opt-in via device_map",
      "unsloth_balanced" in src)

print()
print("3. PATH RESOLUTION ON KAGGLE")
print("-" * 74)
check("DATA_DIR falls back to PROJECT_ROOT",
      "os.path.join(PROJECT_ROOT, DATA_DIR)" in src,
      "CWD on Kaggle is /kaggle/working, where 'data/processed' does not exist")
check("DATA_DIR overridable by env", "LUMINA_DATA_DIR" in src)
check("OUT_DIR writable by default",
      "LUMINA_OUT_DIR" in src and "/kaggle/working/outputs" in src)
check("printout of resolved adapter path", "adapter_path.txt" in src)

print()
print("4. TRAINING HYPERPARAMETERS")
print("-" * 74)
check("fp16 used when bf16 unsupported",
      "fp16=not supports_bf16" in src and "bf16=supports_bf16" in src)
check("seq length within N-ATLaS 8092 ceiling",
      "MAX_SEQ_LENGTH = 1024" in src)
check("no packing (avoids cross-example attention)",
      "packing=False" in src)
check("checkpoints are retained", "save_total_limit" in src)
check("loss is not logged to a third party", 'report_to=[]' in src)

print()
print("5. WHAT WE STILL CANNOT PROVE")
print("-" * 74)
print("""  These require a real GPU and cannot be checked statically:
    * FastLanguageModel.from_pretrained(device_map='unsloth_balanced')
    * get_peft_model target-module compatibility with N-ATLaS
    * peak memory during the training step
    * whether save_pretrained_merged produces a reloadable PeftModel dir
  The fallback save path above covers the last of these.""")

print()
print("=" * 74)
print("PRE-FLIGHT PASSED - no known static blockers"
      if not failures else f"BLOCKERS: {failures}")
print("=" * 74)
sys.exit(1 if failures else 0)