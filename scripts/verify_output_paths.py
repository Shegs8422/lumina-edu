"""Prove output paths relocate off Kaggle's read-only /kaggle/input.

The failure we hit was `OSError: [Errno 30] Read-only file system:
'/kaggle/input/lumina-edu-data/data/distilled'`. These checks simulate the
Kaggle layout by pointing PROJECT_ROOT at a /kaggle/input path and faking the
working directory, then confirm every write target lands somewhere writable.
"""
import os
import sys
import types

FAILURES = []


def check(label, passed, detail=""):
    if not passed:
        FAILURES.append(label)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))


class FakeDir:
    """Minimal os.path.isdir hook so we can pretend /kaggle/working exists."""

    def __init__(self, real_isdir):
        self.extra = set()

    def __call__(self, p):
        return p in self.extra or self.real_isdir(p)


real_isdir = os.path.isdir
fake = FakeDir(real_isdir)
fake.extra.add("/kaggle/working")

src = open("scripts/evaluate_benchmark.py", encoding="utf-8").read()
tree = __import__("ast").parse(src)
fn = next(n for n in tree.body
          if isinstance(n, __import__("ast").FunctionDef)
          and n.name == "resolve_output_dir")
fn_src = __import__("ast").get_source_segment(src, fn)
ns = {"os": os, "PROJECT_ROOT": "/kaggle/input/lumina-edu-data"}
exec(compile(fn_src, "resolve_output_dir", "exec"), ns)
resolve = ns["resolve_output_dir"]

# Patch isdir inside the namespace so the Kaggle branch triggers.
wrapped = types.SimpleNamespace(
    path=types.SimpleNamespace(isdir=lambda p: fake(p), isabs=os.path.isabs,
                              join=os.path.join),
    environ={})
ns["wrapped"] = wrapped
exec(compile(fn_src.replace("os.path.isdir(", "wrapped.path.isdir(")
             .replace("os.path.isabs(", "wrapped.path.isabs(")
             .replace("os.path.join(", "wrapped.path.join(")
             .replace("os.environ", "wrapped.environ"),
             "resolve_patched", "exec"), ns)
resolve = ns["resolve_output_dir"]

print("KAGGLE SIMULATION (input read-only, /kaggle/working writable)")
print("-" * 70)
cases = [
    ("distilled output",
     "/kaggle/input/lumina-edu-data/data/distilled",
     "/kaggle/working/data/distilled"),
    ("benchmark output",
     "/kaggle/input/lumina-edu-data/data/benchmark",
     "/kaggle/working/data/benchmark"),
]
for label, given, expected in cases:
    got = resolve(given).replace("\\", "/")
    check(f"{label} relocates", got == expected, f"got={got}")

print()
print("LOCAL RUN (no /kaggle/working) - must keep the original path")
print("-" * 70)
local_ns = {"os": os, "PROJECT_ROOT": os.getcwd()}
exec(compile(fn_src, "r", "exec"), local_ns)
got = local_ns["resolve_output_dir"](os.path.join(os.getcwd(), "data", "distilled"))
check("local path unchanged", got == os.path.join(os.getcwd(), "data", "distilled"),
      f"got={got}")

print()
print("ENV OVERRIDE")
print("-" * 70)
ov = types.SimpleNamespace(
    path=types.SimpleNamespace(isdir=lambda p: fake(p), isabs=os.path.isabs,
                               join=os.path.join),
    environ={"LUMINA_OUTPUT_DIR": "/custom/path"})
ov_ns = {"os": os, "wrapped": ov}
exec(compile(fn_src.replace("os.path.isdir(", "wrapped.path.isdir(")
             .replace("os.path.isabs(", "wrapped.path.isabs(")
             .replace("os.path.join(", "wrapped.path.join(")
             .replace("os.environ", "wrapped.environ"),
             "r", "exec"), ov_ns)
got = ov_ns["resolve_output_dir"]("/kaggle/input/x/data/distilled")
check("env override wins", got == "/custom/path", f"got={got}")

print()
print("=" * 70)
print("ALL PATH CHECKS PASS" if not FAILURES else f"FAILED: {FAILURES}")
print("=" * 70)
sys.exit(1 if FAILURES else 0)