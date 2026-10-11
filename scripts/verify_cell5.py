"""Extract Cell 5 from KAGGLE_CELLS.md and execute it with unsloth stubbed.

Proves the notebook cell is self-contained and free of the two bugs that
reached Kaggle (__file__ under exec, datetime.now on a module).
"""
import importlib.machinery
import os
import re
import sys
import types

md = open("KAGGLE_CELLS.md", encoding="utf-8").read()
block = md.split("## Cell 5")[1].split("## Cell 6")[0]
cells = re.findall(r"```python\n(.*?)```", block, re.S)
assert cells, "no python cell found in section 5"
src = cells[0]
print(f"extracted cell: {len(src.splitlines())} lines")

# Strip comments before scanning: the cell's own comment text mentions
# "exec()" and "__file__", which would otherwise self-trigger this check.
code_only = "\n".join(
    line for line in src.splitlines()
    if not line.strip().startswith("#"))

# The cell must not read any script from the dataset.
bad = [p for p in ("exec(", "compile(", "__file__", "smoke_test_language.py")
       if p in code_only]
print("forbidden constructs in code:", bad or "none")
assert not bad, f"cell is not self-contained: {bad}"


class _FLM:
    @staticmethod
    def from_pretrained(*a, **k):
        print("  [stub] reached model load - cell header executed cleanly")
        raise SystemExit(0)

    @staticmethod
    def for_inference(*a, **k):
        return None


m = types.ModuleType("unsloth")
m.__spec__ = importlib.machinery.ModuleSpec("unsloth", None)
m.FastLanguageModel = _FLM
sys.modules["unsloth"] = m

g = {"__name__": "__main__"}
try:
    exec(compile(src, "cell5.py", "exec"), g)
    print("completed without reaching model load")
except SystemExit as e:
    if e.code == 0:
        print("RESULT: cell header is clean, reached model load")
    else:
        raise
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
    raise