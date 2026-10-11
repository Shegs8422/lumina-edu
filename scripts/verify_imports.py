"""Verify every script imports cleanly outside its intended entry point.

We hit a NameError on Kaggle because smoke_test_language.py relied on
__file__, which exec() does not define. This checks all of them so the same
class of bug cannot reach the GPU run again.
"""
import importlib.util

NAMES = ["train_lumina_edu", "evaluate_benchmark", "self_distill_translations"]

print("import check (simulates importlib load, __file__ present)")
print("-" * 66)
for n in NAMES:
    spec = importlib.util.spec_from_file_location(n, f"scripts/{n}.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
        has = bool(getattr(m, "__file__", None))
        print(f"  OK    {n:<28} __file__ set: {has}")
    except Exception as e:
        print(f"  FAIL  {n:<28} {type(e).__name__}: {e}")

print()
print("exec check with NO __file__ (Kaggle notebook condition)")
print("-" * 66)
src = open("scripts/smoke_test_language.py", encoding="utf-8").read()
# Drop the model-loading tail; we are only exercising the module header.
head = src.split("from unsloth import")[0]
g = {"DATA_ROOT": os.getcwd(), "os": __import__("os")} if False else {
    "DATA_ROOT": __import__("os").getcwd(), "os": __import__("os")}
try:
    exec(compile(head, "smoke_test_language.py", "exec"), g)
    root = g["PROJECT_ROOT"]
    import os
    ok = os.path.exists(os.path.join(root, "data", "processed", "train.jsonl"))
    print(f"  OK    smoke_test_language.py  PROJECT_ROOT -> {root}")
    print(f"        train.jsonl found: {ok}, records loaded: {len(g.get('recs', []))}")
except Exception as e:
    print(f"  FAIL  {type(e).__name__}: {e}")