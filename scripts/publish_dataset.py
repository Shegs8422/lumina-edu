"""Publish the Lumina-Edu dataset to the Hugging Face Hub.

The repo ships 50-record samples; the full splits live here so judges and
NCAIR reviewers can download the actual benchmark data rather than trusting
our summary numbers.

Uploads:
    data/processed/train.jsonl          (9,196)
    data/processed/dev.jsonl            (200)
    data/processed/benchmark_eval.jsonl  (610)
    data/processed/curation_report.json
    DATASET_CARD.md  -> README.md

The dataset card becomes the Hub dataset card, which is what renders on the
dataset page.

Run:  python scripts/publish_dataset.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_REPO = os.environ.get("LUMINA_DATASET_REPO", "lanre-segun/lumina-edu-nigerian-academic-corpus")

FILES = [
    ("data/processed/train.jsonl", "train.jsonl"),
    ("data/processed/dev.jsonl", "dev.jsonl"),
    ("data/processed/benchmark_eval.jsonl", "benchmark_eval.jsonl"),
    ("data/processed/curation_report.json", "curation_report.json"),
]


def main():
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Logged in to Hugging Face")

    api = HfApi()
    who = api.whoami()
    print(f"Publishing as: {who['name']}")

    missing = [src for src, _ in FILES if not os.path.exists(os.path.join(PROJECT_ROOT, src))]
    if missing:
        print(f"ERROR: missing files: {missing}")
        return 1

    api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{HF_REPO}")

    # The dataset card becomes the Hub README.
    card = os.path.join(PROJECT_ROOT, "DATASET_CARD.md")
    if os.path.exists(card):
        api.upload_file(
            path_or_fileobj=card,
            path_in_repo="README.md",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print("Uploaded README.md (dataset card)")

    for src, dest in FILES:
        api.upload_file(
            path_or_fileobj=os.path.join(PROJECT_ROOT, src),
            path_in_repo=dest,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print(f"  uploaded {dest}")

    print(f"\nDone: https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    sys.exit(main())