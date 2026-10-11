"""Re-verify every flaw found by audit_design.py after the fixes."""
import importlib.util
import json
import os
from collections import Counter


def load(name):
    p = os.path.join("scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tr = load("train_lumina_edu")
sd = load("self_distill_translations")
ev = load("evaluate_benchmark")

recs = tr.load_jsonl(os.path.join("data", "processed", "train.jsonl"))[:300]
bench = tr.load_jsonl(os.path.join("data", "processed", "benchmark_eval.jsonl"))

ok = True


def check(label, passed, detail=""):
    global ok
    if not passed:
        ok = False
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


print("1. COPY-TRAINING ELIMINATED")
print("-" * 70)
ex = tr.build_examples(recs)
comp = [e for e in ex if e["task"] == "compress"]
ratio = []
for e in comp[:200]:
    user = e["messages"][1]["content"]
    tgt = e["messages"][2]["content"]
    abst = user.split("Abstract: ", 1)[1]
    ratio.append(len(tgt) / max(len(abst), 1))
mean_ratio = sum(ratio) / len(ratio)
print(f"  mean target/input length ratio : {mean_ratio:.2f}")
check("compress target is a fraction of the input", mean_ratio < 0.5,
      f"ratio={mean_ratio:.2f}")
check("no task emits the full abstract verbatim",
      all(len(e["messages"][2]["content"]) < len(e["messages"][1]["content"])
          for e in comp[:200]))
print(f"  task mix: {Counter(e['task'] for e in ex).most_common()}")

print()
print("2. ATTRIBUTION METRIC NO LONGER INVERTS")
print("-" * 70)
u = "This study was conducted at University of Ibadan."
r = "This study was conducted at Redeemer's University."
check("correct institution scores True",
      ev.attribution_hit(u, "University of Ibadan") is True)
check("wrong institution scores False",
      ev.attribution_hit(r, "University of Ibadan") is False)
check("confabulated institution scores False",
      ev.attribution_hit("Conducted at University of Lagos.", "University of Ibadan") is False)

print()
print("3. TRAIN/TEST SOURCE MATCH")
print("-" * 70)
tr_src = Counter(r["source"] for r in tr.load_jsonl(os.path.join("data", "processed", "train.jsonl")))
be_src = Counter(r["source"] for r in bench)
tr_frac = tr_src["redeemers"] / sum(tr_src.values())
be_frac = be_src["redeemers"] / sum(be_src.values())
print(f"  redeemers share  train={tr_frac:.1%}  benchmark={be_frac:.1%}")
check("source distribution roughly matches", abs(tr_frac - be_frac) < 0.06,
      f"delta={abs(tr_frac - be_frac):.1%}")

print()
print("4. LANGUAGE GATE REJECTS ENGLISH")
print("-" * 70)
english = ("The study examined the response of students in a class of many "
           "participants, and we can note that a range of data was gathered "
           "from a sample of the population in Nigeria over time.")
for lang in ("Yoruba", "Hausa", "Igbo"):
    passed, hits = sd.looks_like_target_language(english, lang)
    check(f"English rejected as {lang}", passed is False, f"hits={hits}")
check("'a' removed from Igbo markers", "a" not in sd.LANG_MARKERS["Igbo"])

print()
print("5. CHRF HEADLINED FOR NON-ENGLISH")
print("-" * 70)
class _D:
    pass
import pandas as pd
df = pd.DataFrame([
    {"target_language": "English", "base_rouge1": .30, "ft_rouge1": .40,
     "base_rougeL": .30, "ft_rougeL": .40, "base_chrf": .31, "ft_chrf": .42,
     "base_grounding": True, "ft_grounding": True},
    {"target_language": "Yoruba", "base_rouge1": .00, "ft_rouge1": .01,
     "base_rougeL": .00, "ft_rougeL": .01, "base_chrf": .12, "ft_chrf": .25,
     "base_grounding": False, "ft_grounding": True},
])
s = ev.summarise(df)
check("English uses ROUGE-L", s["by_language"]["English"]["headline_metric"] == "rougeL")
check("Yoruba uses chrF", s["by_language"]["Yoruba"]["headline_metric"] == "chrf")
check("Yoruba delta reported from chrF",
      s["by_language"]["Yoruba"]["headline_finetuned"] == .25)

print()
print("=" * 70)
print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
print("=" * 70)