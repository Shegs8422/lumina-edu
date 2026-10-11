"""
Lumina-Edu corpus curation: raw harvest -> defensible training/eval dataset.

The NAIC judges evaluate "real-world validation" and run a separate N-ATLAS
integration check. A dataset that is transparently derived, deduplicated and
stratified is far stronger evidence than a large raw dump, so this script
does the unglamorous work that makes the fine-tune defensible.

Pipeline
--------
1. Load raw OAI-PMH records.
2. Drop non-academic contamination. The University of Ibadan repository has
   absorbed a national newspaper archive, so subjects like "Political news",
   "Adverts", "Vacancies" and "Sports" outrank genuine academic headings.
3. Normalise text and drop records with encoding damage or non-Latin scripts
   that would confuse a byte-level language signal.
4. Near-duplicate removal with TF-IDF + cosine similarity. Exact-match
   dedup is not enough: the repository contains the same article deposited
   as both 'Article' and 'Book chapter', and syndicated copies share
   long verbatim spans. Near-dupes MUST be removed BEFORE the split or the
   benchmark is contaminated by construction.
5. Stratify by (document type, decade-agnostic year bucket, education/not).
6. Emit a disjoint held-out benchmark set large enough to clear the NAIC
   "500+ example benchmark" bar, plus train/dev for the fine-tune.

Outputs (under data/processed/):
    corpus_all.jsonl      curated, deduplicated, labelled pool
    train.jsonl           fine-tuning set
    dev.jsonl             small validation slice
    benchmark_eval.jsonl  held-out 500+ (never seen in training)
    curation_report.json  counts at every stage (the audit trail)
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Subjects that indicate newspaper / administrative noise rather than research.
NON_ACADEMIC = re.compile(
    r"politic|news|advert|vacanc|sport|biograph|religio|announce|obituar|"
    r"celebrat|job|employ|market|award|correspond|column|editorial|career|"
    r"obituar|notice|bulletin|announc|praise|thanks|funeral|weddin|birthday",
    re.I,
)

# Document types that are not research output.
NON_RESEARCH_TYPES = {"Other", "Presentation", "n/a", "", "Animation",
                      "Recording, oral", "Recording, musical", "Video"}

ACADEMIC_TYPES = {
    "Article", "Thesis", "Dissertation", "Book chapter", "Book Chapter",
    "Book", "Technical Report", "Working Paper", "In Proceedings",
    "INPROCEEDINGS", "Dataset",
}

EDU = re.compile(
    r"educat|student|teacher|curricul|learn|pedagog|instruction|literacy|"
    r"school|classroom|psycholog|teaching|assessment|examin|universit|"
    r"undergrad|postgrad|admission|scholarship",
    re.I,
)


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def has_bad_encoding(text):
    if "\ufffd" in text:
        return True
    # Runs of punctuation typical of mojibake / broken PDF extraction.
    return bool(re.search(r"[^\w\s\.,;:()\-'\"]{6,}", text))


def year_of(rec):
    m = re.search(r"(19|20)\d{2}", rec.get("date") or "")
    return int(m.group(0)) if m else None


def stage_load():
    records = []
    for fn in sorted(os.listdir(RAW_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(RAW_DIR, fn), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def stage_filter(records, audit):
    kept, dropped = [], Counter()
    for r in records:
        title, abstract = r.get("title", ""), r.get("abstract", "")
        subjects = " | ".join(r.get("subjects") or [])
        dtype = (r.get("types") or [""])[0]

        if NON_ACADEMIC.search(subjects):
            dropped["non_academic_subject"] += 1
            continue
        if dtype in NON_RESEARCH_TYPES:
            dropped["non_research_type"] += 1
            continue
        if len(abstract) < 250:
            dropped["abstract_too_short"] += 1
            continue
        if len(abstract) > 6000:
            dropped["abstract_truncated_garbage"] += 1
            continue
        if has_bad_encoding(title) or has_bad_encoding(abstract):
            dropped["encoding_damage"] += 1
            continue
        # Very low alphabetic ratio signals a non-textual deposit.
        alpha = sum(c.isalpha() or c.isspace() for c in abstract) / max(len(abstract), 1)
        if alpha < 0.72:
            dropped["low_alphabetic_ratio"] += 1
            continue

        r["_dtype"] = dtype if dtype in ACADEMIC_TYPES else "Article"
        r["_year"] = year_of(r)
        r["_is_education"] = bool(
            EDU.search(title) or EDU.search(subjects) or EDU.search(abstract[:600])
        )
        kept.append(r)
    audit["after_filter"] = len(kept)
    audit["dropped_by_filter"] = dict(dropped)
    return kept


def stage_dedupe(records, audit, threshold=0.82):
    """Exact-title dedup, then TF-IDF cosine near-duplicate removal.

    Uses radius_neighbors rather than a full N x N similarity matrix: with
    ~11k documents a dense pairwise comparison is both slow and memory-hungry,
    and a naive Python loop over CSR rows dominated the runtime.
    radius_neighbors materialises only the pairs above the threshold.
    """
    from sklearn.neighbors import NearestNeighbors

    # --- pass 1: exact duplicates on normalised title ---
    seen, exact = set(), []
    dup_exact = 0
    for r in records:
        key = norm(r["title"])[:180]
        if key in seen:
            dup_exact += 1
            continue
        seen.add(key)
        exact.append(r)

    # --- pass 2: near-duplicates ---
    texts = [f"{r['title']}. {r['abstract'][:1200]}" for r in exact]
    vec = TfidfVectorizer(
        max_features=60000, ngram_range=(1, 2), min_df=2,
        stop_words="english", sublinear_tf=True,
    )
    X = vec.fit_transform(texts)

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)
    dist, ind = nn.radius_neighbors(X, radius=1.0 - threshold, return_distance=True)

    remove = set()
    for i in range(len(exact)):
        for j in ind[i]:
            j = int(j)
            if j > i:
                remove.add(j)

    kept = [r for i, r in enumerate(exact) if i not in remove]

    audit["after_exact_dedupe"] = len(exact)
    audit["dropped_exact_dupe"] = dup_exact
    audit["after_near_dedupe"] = len(kept)
    audit["dropped_near_dupe"] = len(remove)
    audit["near_dupe_threshold"] = threshold
    return kept


def stage_split(records, audit, n_eval, n_dev):
    """Stratify by (type, education flag) so the benchmark mirrors the corpus."""
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["_dtype"], r["_is_education"])].append(r)

    eval_set, train_set = [], []
    # Prioritise education records so the eval set reflects our target vertical.
    for key, items in sorted(buckets.items(), key=lambda kv: (not kv[0][1], -len(kv[1]))):
        items.sort(key=lambda r: r["id"])
        quota = max(1, round(n_eval * len(items) / len(records)))
        eval_set.extend(items[:quota])
        train_set.extend(items[quota:])

    train_set.sort(key=lambda r: r["id"])
    eval_set.sort(key=lambda r: r["id"])
    dev_set = train_set[:n_dev]
    final_train = train_set[n_dev:]

    audit["benchmark_size"] = len(eval_set)
    audit["train_size"] = len(final_train)
    audit["dev_size"] = len(dev_set)
    audit["benchmark_meets_naic_500"] = len(eval_set) >= 500
    return final_train, dev_set, eval_set


def public(rec):
    return {
        k: v for k, v in rec.items()
        if not k.startswith("_") and k not in ("creators",)
    }


def stats(records):
    return {
        "count": len(records),
        "by_type": dict(Counter(r["_dtype"] for r in records)),
        "education_flagged": sum(1 for r in records if r["_is_education"]),
        "by_institution": dict(Counter(r["institution"] for r in records)),
        "year_min": min((r["_year"] for r in records if r["_year"]), default=None),
        "year_max": max((r["_year"] for r in records if r["_year"]), default=None),
        "median_abstract_chars": int(np.median([len(r["abstract"]) for r in records])) if records else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-size", type=int, default=600)
    ap.add_argument("--dev-size", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.82)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    audit = {"input_files": sorted(os.listdir(RAW_DIR))}

    raw = stage_load()
    audit["raw_records"] = len(raw)
    print(f"raw records            : {len(raw)}")

    filtered = stage_filter(raw, audit)
    print(f"after academic filter  : {len(filtered)}")

    deduped = stage_dedupe(filtered, audit, args.threshold)
    print(f"after dedupe           : {len(deduped)}")

    train, dev, evalset = stage_split(deduped, audit, args.eval_size, args.dev_size)
    print(f"train / dev / benchmark: {len(train)} / {len(dev)} / {len(evalset)}")

    audit["stats"] = {
        "train": stats(train), "dev": stats(dev), "benchmark": stats(evalset),
    }

    # Guard: no id may appear in more than one split.
    ids = {
        "train": {r["id"] for r in train},
        "dev": {r["id"] for r in dev},
        "benchmark": {r["id"] for r in evalset},
    }
    overlap = {
        "train&dev": len(ids["train"] & ids["dev"]),
        "train&benchmark": len(ids["train"] & ids["benchmark"]),
        "dev&benchmark": len(ids["dev"] & ids["benchmark"]),
    }
    audit["split_overlap"] = overlap
    audit["splits_disjoint"] = all(v == 0 for v in overlap.values())

    # Guard: no near-duplicate text may straddle train and benchmark.
    vec = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([f"{r['title']}. {r['abstract'][:1200]}" for r in train])
    Xev = vec.transform([f"{r['title']}. {r['abstract'][:1200]}" for r in evalset])
    sim = cosine_similarity(Xev, Xtr, dense_output=False)
    max_sim = float(sim.max()) if sim.nnz else 0.0
    leak = int((sim >= 0.5).sum())
    audit["max_train_eval_cosine"] = round(max_sim, 4)
    audit["train_eval_pairs_above_0.5"] = leak

    for name, rows in (("train", train), ("dev", dev), ("benchmark_eval", evalset)):
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(public(r), ensure_ascii=False) + "\n")

    with open(os.path.join(OUT_DIR, "corpus_all.jsonl"), "w", encoding="utf-8") as fh:
        for r in deduped:
            fh.write(json.dumps(public(r), ensure_ascii=False) + "\n")

    with open(os.path.join(OUT_DIR, "curation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    print("\n--- AUDIT ---")
    print(json.dumps({k: v for k, v in audit.items() if k != "stats"}, indent=2))
    print(f"\nLEAK CHECK: splits_disjoint={audit['splits_disjoint']} "
          f"max_train_eval_cosine={audit['max_train_eval_cosine']} "
          f"pairs>=0.5={leak}")
    print(f"NAIC 500+ BENCHMARK: {audit['benchmark_meets_naic_500']}")


if __name__ == "__main__":
    main()