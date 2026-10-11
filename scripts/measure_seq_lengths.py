"""Measure real token lengths of the training examples to size MAX_SEQ_LENGTH.

Guessing sequence length is how you either OOM or waste half your GPU budget.
This downloads only the tokenizer (a few MB, not the 16 GB weights) and
measures the chat-formatted examples we would actually train on.
"""
import importlib.util
import os

from transformers import AutoTokenizer

TOKEN = os.environ.get("HF_TOKEN")
spec = importlib.util.spec_from_file_location(
    "trainer", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "train_lumina_edu.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

tok = AutoTokenizer.from_pretrained("NCAIR1/N-ATLaS", token=TOKEN)

recs = tr.load_jsonl(os.path.join("data", "processed", "train.jsonl"))[:1500]
examples = tr.build_examples(recs)

texts = [tok.apply_chat_template(e["messages"], tokenize=False,
                                 add_generation_prompt=False)
         for e in examples]

lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
lens_sorted = sorted(lens)


def pct(p):
    return lens_sorted[int(len(lens) * p) - 1]


print(f"examples measured : {len(lens)}")
print(f"min / median / p75 / p90 / p99 / max:")
print(f"  {lens_sorted[0]} / {pct(0.50)} / {pct(0.75)} / {pct(0.90)} "
      f"/ {pct(0.99)} / {lens_sorted[-1]}")
print()
for window in (512, 768, 1024, 1536, 2048):
    over = sum(1 for x in lens if x > window)
    print(f"  MAX_SEQ_LENGTH={window:<5} truncates {over:>5} / {len(lens)} "
          f"({100 * over / len(lens):5.1f}%)   "
          f"compute saved vs 2048: {100 * (1 - window / 2048):4.0f}%")