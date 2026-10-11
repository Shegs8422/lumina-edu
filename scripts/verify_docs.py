"""Fact-check the documentation against the real artefacts.

Documentation drifts. Every number we assert to NAIC judges must match what
actually shipped, so we verify each claim programmatically.
"""
import json
import os
import re
from collections import Counter

ok = True
report = json.load(open("data/processed/curation_report.json", encoding="utf-8"))


def check(label, passed, detail=""):
    global ok
    if not passed:
        ok = False
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


print("CORPUS CLAIMS")
print("-" * 70)
check("raw = 13,998", report["raw_records"] == 13998, f"actual={report['raw_records']}")
check("after filter = 10,971", report["after_filter"] == 10971, f"actual={report['after_filter']}")
check("after dedupe = 10,006", report["after_near_dedupe"] == 10006, f"actual={report['after_near_dedupe']}")
check("train = 9,196", report["train_size"] == 9196, f"actual={report['train_size']}")
check("dev = 200", report["dev_size"] == 200, f"actual={report['dev_size']}")
check("benchmark = 610", report["benchmark_size"] == 610, f"actual={report['benchmark_size']}")
check("NAIC 500+ met", report["benchmark_meets_naic_500"] is True)
check("splits disjoint", report["splits_disjoint"] is True)
check("max train-eval cosine = 0.684", abs(report["max_train_eval_cosine"] - 0.684) < 0.002,
      f"actual={report['max_train_eval_cosine']}")
check("pairs above 0.5 = 35", report["train_eval_pairs_above_0.5"] == 35,
      f"actual={report['train_eval_pairs_above_0.5']}")

print()
print("FILTER CLAIMS")
print("-" * 70)
d = report["dropped_by_filter"]
check("non-academic subjects = 2,388", d["non_academic_subject"] == 2388, f"actual={d['non_academic_subject']}")
check("non-research types = 532", d["non_research_type"] == 532, f"actual={d['non_research_type']}")
enc = d["encoding_damage"] + d["low_alphabetic_ratio"]
lens = d["abstract_too_short"] + d["abstract_truncated_garbage"]
check("encoding+ratio rejects = 42", enc == 42, f"actual={enc}")
check("length rejects = 65", lens == 65, f"actual={lens}")
total_removed = report["raw_records"] - report["after_filter"]
check("every removed record is accounted for",
      d["non_academic_subject"] + d["non_research_type"] + enc + lens == total_removed,
      f"sum={d['non_academic_subject'] + d['non_research_type'] + enc + lens} "
      f"raw-filter={total_removed}")
check("exact dupes = 672", report["dropped_exact_dupe"] == 672, f"actual={report['dropped_exact_dupe']}")
check("near dupes = 293", report["dropped_near_dupe"] == 293, f"actual={report['dropped_near_dupe']}")

print()
print("INSTITUTION CLAIMS (README says 97% UI)")
print("-" * 70)
tr = load("data/processed/train.jsonl")
be = load("data/processed/benchmark_eval.jsonl")
tc = Counter(r["institution"] for r in tr)
bc = Counter(r["institution"] for r in be)
UI = "University of Ibadan"
RUN = "Redeemer" + chr(39) + "s University"
ui_share = tc[UI] / sum(tc.values())
print(f"  train  : {dict(tc)}")
print(f"  bench  : {dict(bc)}")
print(f"  UI share of train: {ui_share:.1%}")
check("train is ~99% one institution", 0.98 < ui_share < 0.995, f"actual={ui_share:.1%}")
check("train holds Redeemers records = 64", tc[RUN] == 64, f"actual={tc[RUN]}")
check("benchmark exceeds 500", len(be) >= 500, f"actual={len(be)}")

print()
print("DOCUMENTATION REFERENCES RESOLVE")
print("-" * 70)
for doc in ("README.md", "IMPLEMENTATION.md", "DATASET_CARD.md", "ATTRIBUTION.md"):
    check(f"{doc} exists", os.path.exists(doc))
for f in ("scripts/audit_design.py", "scripts/verify_fixes.py",
          "scripts/measure_seq_lengths.py", "KAGGLE_CELLS.md"):
    check(f"{f} referenced exists", os.path.exists(f))

readme = open("README.md", encoding="utf-8").read()
impl = open("IMPLEMENTATION.md", encoding="utf-8").read()
for token in ("13,998", "10,006", "9,196", "610", "0.684"):
    check(f"README states {token}", token in readme)
for token in ("13,998", "10,971", "9,196", "610", "0.684", "2,388"):
    check(f"IMPLEMENTATION states {token}", token in impl)
check("README links IMPLEMENTATION.md", "IMPLEMENTATION.md" in readme)
check("README has no live token", not re.search(r"hf_[A-Za-z0-9]{20,}|KGAT_[a-f0-9]{20,}", readme))
check("IMPLEMENTATION has no live token", not re.search(r"hf_[A-Za-z0-9]{20,}|KGAT_[a-f0-9]{20,}", impl))

print()
print("=" * 70)
print("DOCUMENTATION IS ACCURATE" if ok else "DISCREPANCIES FOUND")
print("=" * 70)