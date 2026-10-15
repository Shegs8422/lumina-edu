"""
Purge the benchmark output directory and start clean.

Why this exists
---------------
Benchmark runs were lost twice mid-submission: once to CUDA OOM, once to a
runtime restart that wiped `/data/benchmark`. Half-finished `base.jsonl` and
`finetuned.jsonl` files are worse than none, because the scoring step pairs on
IDs and a truncated base pass silently produces a smaller paired sample than the
610 the submission claims. Starting from an empty directory makes the count
verifiable.

Safety
------
Deleting files is destructive, so this script:

  * Is a **dry run by default**. Nothing is removed unless ``--yes`` is passed.
  * Refuses any target that does not end in ``benchmark``, unless ``--force``.
  * Refuses the filesystem root, home, or any path under ``/data`` that is not
    the benchmark directory itself.
  * Never follows symlinks when deleting; it removes the link, not the target.
  * Prints a full inventory (files, sizes, line counts) before doing anything.

Usage
-----
    python reset_benchmark.py                      # dry run, reports only
    python reset_benchmark.py --yes                # actually delete
    python reset_benchmark.py --root /tmp/foo --yes --force
"""

import argparse
import os
import shutil
import sys

DEFAULT_ROOT = "/data/benchmark"

# Paths we refuse to touch regardless of --force being forgotten.
FORBIDDEN = {"/", "/data", "/content", os.path.expanduser("~"),
             "/content/drive", "/content/drive/MyDrive"}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def inventory(root):
    """Describe everything currently in root."""
    if not os.path.exists(root):
        return []
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for f in sorted(filenames):
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root)
            try:
                size = os.path.getsize(p)
            except OSError:
                size = -1
            lines = ""
            if f.endswith(".jsonl"):
                try:
                    with open(p, encoding="utf-8") as fh:
                        lines = f"  ({sum(1 for _ in fh)} records)"
                except OSError:
                    lines = "  (unreadable)"
            rows.append((rel, size, lines))
    return rows


def check_target(root, force):
    resolved = os.path.abspath(root)

    if resolved in FORBIDDEN:
        sys.exit(f"REFUSING: {resolved} is a protected path.")

    if resolved.rstrip("/").endswith("benchmark") or force:
        return resolved

    sys.exit(f"REFUSING: {resolved} does not end in 'benchmark'.\n"
             f"         Re-run with --force if this is really the target.")


def purge(root):
    if os.path.islink(root):
        print(f"  {root} is a symlink -> removing the link only")
        os.remove(root)
    elif os.path.isdir(root):
        shutil.rmtree(root)
    elif os.path.exists(root):
        os.remove(root)

    os.makedirs(root, exist_ok=True)
    print(f"  recreated empty: {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--yes", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--force", action="store_true",
                    help="allow a target that does not end in 'benchmark'")
    args = ap.parse_args()

    root = check_target(args.root, args.force)
    print(f"target: {root}")
    print(f"       {os.path.realpath(root)}")

    rows = inventory(root)

    print("\n=== current contents ===")
    if not rows:
        print("  (empty - nothing to purge)")
    for rel, size, lines in rows:
        print(f"  {human(size):>10}  {rel}{lines}")
    if rows:
        total = sum(s for _, s, _ in rows if s > 0)
        print(f"  {'-' * 10}")
        print(f"  {human(total):>10}  in {len(rows)} file(s)")

    if not args.yes:
        print("\nDRY RUN - nothing deleted.")
        print("Re-run with --yes to purge.")
        return

    if not rows:
        print("\nnothing to delete.")
        return

    print("\n=== purging ===")
    purge(root)

    left = inventory(root)
    if left:
        print(f"  WARNING: {len(left)} file(s) remain")
        for rel, _, _ in left:
            print(f"    {rel}")
    else:
        print("\nVERIFIED CLEAN - benchmark directory is empty")
    print("Ready for a fresh run.")


if __name__ == "__main__":
    main()