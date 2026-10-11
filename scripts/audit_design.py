"""Audit the training + benchmark design for methodological flaws.

Speculation is cheap here; several of these are silent correctness bugs that
would not raise an exception but would invalidate the results. Each check
below is executable evidence, not opinion.
"""
import importlib.util
import json
import os
import re
import statistics
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    p = os.path.join(ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tr = load("train_lumina_edu")
sd = load("self_distill_translations")
ev = load("evaluate_benchmark")

recs = tr.load_jsonl(os.path.join("data", "processed", "train.jsonl"))[:400]
bench = tr.load_jsonl(os.path.join("data", "processed", "benchmark_eval.jsonl"))

print("=" * 74)
print("1. IS THE 'SUMMARISE' TASK ACTUALLY SUMMARISATION?")
print("=" * 74)
ex = tr.build_examples(recs[:1])[0]
user = ex["messages"][1]["content"]
asst = ex["messages"][2]["content"]
# How much of the assistant target is verbatim from the user prompt?
verbatim = asst.count(recs[0]["abstract"][:200])
print(f"  assistant target contains abstract verbatim : {bool(verbatim)}")
frac = len(recs[0]["abstract"]) / max(len(asst), 1)
print(f"  share of target that is the copied abstract : {frac:.1%}")
copy_targets = 0
for e in tr.build_examples(recs):
    if e["task"] == "summarise" and e["messages"][2]["content"].count(
            next(r["abstract"] for r in recs if r["title"] == e["messages"][1]["content"].split("Title: ")[1].split("\n")[0])[:150]):
        copy_targets += 1
print(f"  summarise examples that echo the input      : {copy_targets}/400")
print("  => ROUGE is scored against the SAME abstract that appears in the")
print("     prompt, so a model trained to copy scores well for the wrong reason.")

print()
print("=" * 74)
print("2. DOES THE GROUNDING METRIC MARK CORRECT ANSWERS AS HALLUCINATIONS?")
print("=" * 74)
# Reproduce what the benchmark actually puts in the prompt.
sample = bench[0]
if sample["target_language"] if "target_language" in sample else True:
    pass
prompt_built = f"Summarise the main finding of this research.\n\nTitle: {sample['title']}\nAbstract: {sample['abstract']}"
print(f"  prompt mentions institution? {'University of Ibadan' in prompt_built}")
print(f"  ground-truth institution     : {sample['institution']}")
v = ev.grounding_ok(f"This study was conducted at {sample['institution']}.", prompt_built)
print(f"  model correctly says '{sample['institution']}' -> grounded = {v}")
print("  => The prompt never names the institution, so naming it is scored")
print("     as a hallucination. The metric inverts a correct answer.")

print()
print("=" * 74)
print("3. ARE THE INSTITUTION DISTRIBUTIONS DEGENERATE?")
print("=" * 74)
print("  train institutions     :", Counter(r["institution"] for r in
      tr.load_jsonl(os.path.join("data", "processed", "train.jsonl"))).most_common())
print("  benchmark institutions :", Counter(r["institution"] for r in bench).most_common())
print("  => A single-institution corpus makes the 'grounding' training task a")
print("     constant, and gives the hallucination metric almost no signal.")

print()
print("=" * 74)
print("4. IS THE LANGUAGE GATE AT RISK OF FALSE POSITIVES?")
print("=" * 74)
english = ("The study examined the response of students in a class of many "
           "participants, and we can note that a range of data was gathered "
           "from a sample of the population in Nigeria over time.")
ok, hits = sd.looks_like_target_language(english, "Igbo")
print(f"  English sentence scored as Igbo? {ok} (marker hits={hits})")
print(f"  Igbo marker list: {sd.LANG_MARKERS['Igbo']}")
print("  => 'a' is a marker and is the English article; the gate is fragile.")

print()
print("=" * 74)
print("5. ARE ROUGE METRICS MEANINGFUL FOR THE TARGET LANGUAGES?")
print("=" * 74)
from rouge_score import rouge_scorer
sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
yor = "Iloko ni ọ̀kan láàárín àwọn èdè Yorùbá tí ó ń ṣàlàyé nípa ẹ̀kọ́"
eng = "The study was conducted among Yoruba speakers in the southwest region"
print(f"  rougeL(yoruba ref, english text) = "
      f"{sc.score(yor, eng)['rougeL'].fmeasure:.3f}   <- English stemmer")
print(f"  chrf  (yoruba ref, english text) = {ev.chrf(eng, yor):.3f}")
print("  => English stemming on Yoruba/Hausa/Igbo is close to meaningless;")

print()
print("=" * 74)
print("6. HOW MUCH OF THE BENCHMARK IS SINGLE-INSTITUTION?")
print("=" * 74)
single = sum(1 for r in bench if r["institution"] == Counter(
    r["institution"] for r in bench).most_common(1)[0][0])
print(f"  {single}/{len(bench)} records from one institution "
      f"({100*single/len(bench):.0f}%)")
print("  The stated problem is 'hallucinated local citations', which cannot")
print("  be measured on a corpus drawn from a single university.")