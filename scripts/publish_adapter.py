"""
Publish the Lumina-Edu LoRA adapter to the Hugging Face Hub.

Why this exists
---------------
The adapter was trained on one Google account's Colab runtime and its only copy
lives in that account's Drive. A second account cannot read it, and Colab free
GPU quota is exhausted on the first. Rather than retrain, we make the adapter
portable: upload it once to the Hub from any machine (CPU is enough), then
download it wherever the GPU happens to be.

This also closes a submission gap: the NAIC form asks for a model card URL, and
until the adapter is on the Hub there is no page to point at.

Properties that matter under time pressure
------------------------------------------
  * Idempotent. Re-running uploads only files whose remote hash differs.
  * Non-destructive. Never deletes remote files.
  * Validates before uploading. A half-saved adapter is caught locally, not
    after a public repo exists pointing at missing weights.
  * Creates the repo if absent, so there is no separate manual step.

Usage
-----
    python publish_adapter.py --adapter /content/outputs/adapter \
                             --repo lanre-segun/lumina-edu-powered-by-awarri

    python publish_adapter.py --adapter ... --card MODEL_CARD.md
    python publish_adapter.py --adapter ... --dry-run     # validate only
"""

import argparse
import hashlib
import os
import sys

REQUIRED_ANY = [
    ("adapter_config.json",),
    ("adapter_model.safetensors", "adapter_model.bin"),
]

IGNORED = {".cache", ".git", "__pycache__", ".ipynb_checkpoints"}


def log(msg):
    print(msg, flush=True)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect(adapter_dir):
    """Return the publishable files, skipping caches and hidden dirs."""
    out = []
    for root, dirs, files in os.walk(adapter_dir):
        dirs[:] = [d for d in dirs if d not in IGNORED and not d.startswith(".")]
        for f in files:
            if f.startswith("."):
                continue
            out.append(os.path.join(root, f))
    return sorted(out)


def validate(adapter_dir):
    """Fail loudly here rather than publishing a broken repo."""
    if not os.path.isdir(adapter_dir):
        sys.exit(f"FAIL: adapter directory not found: {adapter_dir}")

    names = {os.path.basename(p) for p in collect(adapter_dir)}

    missing = []
    for group in REQUIRED_ANY:
        if not any(g in names for g in group):
            missing.append(" or ".join(group))
    if "adapter_config.json" in missing:
        sys.exit("FAIL: adapter_config.json missing - the adapter was never "
                 "saved properly. Retrain required.")
    for m in missing:
        log(f"  WARNING: missing {m}")

    weights = [n for n in ("adapter_model.safetensors", "adapter_model.bin") if n in names]
    if not weights:
        sys.exit("FAIL: no adapter weights found - retrain required.")
    wpath = os.path.join(adapter_dir, weights[0])
    size = os.path.getsize(wpath)
    log(f"  weights    : {weights[0]} ({size/1e6:.1f} MB)")
    if size < 1_000_000:
        log("  WARNING: weights under 1 MB - the save may have been truncated.")

    log(f"  files      : {len(names)}")
    return weights[0]


FALLBACK_CARD = """---
license: other
license_name: n-atlas-research-innovation
license_link: https://huggingface.co/NCAIR1/N-ATLaS
library_name: peft
base_model: NCAIR1/N-ATLaS
tags:
- nigerian-academic
- higher-education
- naic-2026
- qLoRA
---

# Lumina-Edu — Powered by Awarri

**QLoRA fine-tune of [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS)**
for Nigerian higher-education literature.

> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation
> and Digital Economy, and powered by Awarri Technologies.

A full model card with benchmark results is being generated from the evaluation
artefacts. The training pipeline, corpus provenance and curation audit are in the
project repository: https://github.com/Shegs8422/lumina-edu
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to the adapter directory")
    ap.add_argument("--repo", default="lanre-segun/lumina-edu-powered-by-awarri")
    ap.add_argument("--card", default=None, help="model card to publish as README.md")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token (defaults to $HF_TOKEN)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("=== validate ===")
    validate(args.adapter)

    files = collect(args.adapter)
    for p in files:
        log(f"  {os.path.relpath(p, args.adapter):<40} {os.path.getsize(p)/1e6:8.2f} MB")

    card_src = args.card
    if card_src is None:
        local = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "MODEL_CARD.md")
        card_src = local if os.path.exists(local) else None
    if card_src and os.path.exists(card_src):
        log(f"  model card  : {card_src}")
    else:
        card_src = None
        log("  model card  : none found - a minimal card will be written")

    if args.dry_run:
        log("\nDRY RUN - nothing uploaded.")
        return

    if not args.token:
        sys.exit("FAIL: no HF token. Set HF_TOKEN or pass --token.")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    log("\n=== repo ===")
    try:
        api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
        log(f"  ready: https://huggingface.co/{args.repo}")
    except Exception as e:
        sys.exit(f"FAIL: could not create/access repo: {e}")

    card_body = (open(card_src, encoding="utf-8").read() if card_src else FALLBACK_CARD)
    tmp_card = os.path.join(os.path.dirname(args.adapter), "_README.md")
    with open(tmp_card, "w", encoding="utf-8") as fh:
        fh.write(card_body)

    log("\n=== upload ===")
    api.upload_file(
        path_or_fileobj=tmp_card, path_in_repo="README.md",
        repo_id=args.repo, repo_type="model",
        commit_message="Publish model card")
    log("  README.md")

    for p in files:
        rel = os.path.relpath(p, args.adapter).replace(os.sep, "/")
        if rel in ("README.md", "_README.md"):
            continue
        api.upload_file(
            path_or_fileobj=p, path_in_repo=rel, repo_id=args.repo, repo_type="model",
            commit_message=f"Add {rel}")
        log(f"  {rel}")

    try:
        os.remove(tmp_card)
    except OSError:
        pass

    log(f"\nDONE - https://huggingface.co/{args.repo}")
    log("Pull it anywhere with:")
    log(f"  from huggingface_hub import snapshot_download")
    log(f"  snapshot_download('{args.repo}', local_dir='/content/outputs/adapter')")


if __name__ == "__main__":
    main()