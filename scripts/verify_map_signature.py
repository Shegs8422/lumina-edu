"""Verify every datasets.map callback has the signature map() will actually call.

The trainer failed on Kaggle with:
    TypeError: to_chat_text() got multiple values for argument 'tokenizer'

because `Dataset.map(fn, batched=True, fn_kwargs={"tokenizer": tok})` passes
the batch as the FIRST POSITIONAL argument. Declaring the callback as
(tokenizer, examples) collides with the `tokenizer=` keyword.

This exercises the real datasets library with a stub tokenizer, so the
signature is tested rather than assumed.
"""
import importlib.util
import inspect
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import Dataset  # noqa: E402


class StubTokenizer:
    def apply_chat_template(self, messages, **kw):
        assert isinstance(messages, list)
        return "\n".join(m.get("content", "") for m in messages) or "empty"


spec = importlib.util.spec_from_file_location(
    "trainer", os.path.join("scripts", "train_lumina_edu.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

failures = []

print("1. CALLBACK SIGNATURE")
print("-" * 70)
sig = inspect.signature(tr.to_chat_text)
params = list(sig.parameters)
ok = params[:2] == ["examples", "tokenizer"]
print(f"  [{'PASS' if ok else 'FAIL'}] to_chat_text{params[:2]} "
      f"(must start with examples, tokenizer)")
if not ok:
    failures.append("signature order")

print()
print("2. EXECUTE THROUGH THE REAL datasets.map")
print("-" * 70)
rows = [
    {"messages": [{"role": "user", "content": "Summarise this."},
                  {"role": "assistant", "content": "A summary."}],
     "task": "compress"},
    {"messages": [{"role": "user", "content": "Explain in Yoruba."},
                  {"role": "assistant", "content": "Idahun ni."}],
     "task": "translate_yoruba"},
]
ds = Dataset.from_list(rows)
try:
    out = ds.map(tr.to_chat_text, batched=True,
                 fn_kwargs={"tokenizer": StubTokenizer()})
    ok = "text" in out.column_names
    print(f"  [{'PASS' if ok else 'FAIL'}] map() produced columns "
          f"{out.column_names}")
    if not ok:
        failures.append("map output")
    else:
        print(f"        sample rendered: {out[0]['text'][:60]!r}")
except TypeError as e:
    print(f"  [FAIL] {e}")
    failures.append("map TypeError")

print()
print("3. FULL PIPELINE SHAPE")
print("-" * 70)
recs = tr.load_jsonl(os.path.join("data", "processed", "dev.jsonl"))[:20]
ex = tr.build_examples(recs)
ds2 = Dataset.from_list(ex)
mapped = ds2.map(tr.to_chat_text, batched=True,
                 fn_kwargs={"tokenizer": StubTokenizer()})
n_empty = sum(1 for t in mapped["text"] if t == "empty")
print(f"  examples built : {len(ex)}")
print(f"  after map      : {len(mapped)}")
print(f"  empty renders  : {n_empty}")
if len(mapped) != len(ex):
    failures.append("row count changed")
print(f"  [{'PASS' if len(mapped) == len(ex) else 'FAIL'}] row count preserved")

print()
print("=" * 70)
print("ALL CHECKS PASS" if not failures else f"FAILED: {failures}")
print("=" * 70)
sys.exit(1 if failures else 0)