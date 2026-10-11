"""Static + partial-runtime verification of every script.

The datetime bug reached Kaggle because our earlier check split the script at
`from unsloth import`, so it never executed the code after that point. This
version runs the ENTIRE module body with `torch` and `unsloth` stubbed to
raise a sentinel, so every line up to model loading actually executes and any
NameError/AttributeError surfaces locally instead of on the GPU.

Also statically flags the `import datetime` / `datetime.now()` misuse class
across the repo.
"""
import ast
import builtins
import glob
import importlib.machinery
import io
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTINEL = "__MODEL_LOAD_REACHED__"

ok = True


def check(label, passed, detail=""):
    global ok
    if not passed:
        ok = False
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))


print("1. STATIC SCAN: datetime import misuse")
print("-" * 72)
for path in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py"))
                   + glob.glob(os.path.join(ROOT, "server", "*.py"))):
    src = open(path, encoding="utf-8").read()
    name = os.path.relpath(path, ROOT).replace("\\", "/")
    uses_now = "datetime.now(" in src
    good_import = "from datetime import datetime" in src
    bad_import = any(
        isinstance(n, ast.Import) and any(a.name == "datetime" for a in n.names)
        for n in ast.walk(ast.parse(src))
    )
    if uses_now:
        check(f"{name}", good_import and not (bad_import and not good_import),
              "uses datetime.now()")

# Scripts that declare a required flag, mapped to a throwaway value. Kept
# explicit rather than derived from the AST: an undeclared flag makes argparse
# exit(2), which would be a bug in this harness, not in the code under test.
REQUIRED_FLAGS = {
    "evaluate_benchmark": ["--adapter", "dummy-adapter"],
}


print()
print("2. PARTIAL RUNTIME: execute module body with torch/unsloth stubbed")
print("-" * 72)


def stub_torch():
    m = types.ModuleType("torch")
    # Libraries probe torch.__spec__ and torch.Tensor during import, so the
    # stub must provide them or the harness fails for reasons unrelated to
    # our code.
    m.__spec__ = importlib.machinery.ModuleSpec("torch", None)

    class _Tensor:
        pass

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported():
            return False

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def manual_seed_all(*a, **k):
            return None

    m.cuda = _Cuda
    m.Tensor = _Tensor
    m.device = types.SimpleNamespace(cuda="cuda")
    m.float16, m.bfloat16, m.long, m.uint8 = "fp16", "bf16", "long", "uint8"
    m.manual_seed = lambda *a, **k: None
    m.inference_mode = lambda *a, **k: (lambda: None)
    m.no_grad = lambda *a, **k: (_ for _ in ()).throw(SystemExit(SENTINEL))
    return m


def stub_module(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def stub_unsloth():
    class _FLM:
        @staticmethod
        def from_pretrained(*a, **k):
            raise SystemExit(SENTINEL)

        @staticmethod
        def get_peft_model(*a, **k):
            raise SystemExit(SENTINEL)

        @staticmethod
        def for_inference(*a, **k):
            return None

    return stub_module("unsloth", FastLanguageModel=_FLM)


def stub_transformers():
    class TrainingArguments:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    m = stub_module("transformers", TrainingArguments=TrainingArguments)

    # Import chains reach for many transformers symbols. Resolve any missing
    # attribute to a placeholder so the harness tests OUR code path rather
    # than failing on an unrelated third-party import.
    def _any_attr(name):
        return type(name, (), {})

    m.__getattr__ = _any_attr
    return m


for name in ("smoke_test_language", "self_distill_translations",
             "train_lumina_edu", "evaluate_benchmark"):
    path = os.path.join(ROOT, "scripts", f"{name}.py")
    src = open(path, encoding="utf-8").read()
    g = {"__name__": "__main__", "__file__": path,
         "DATA_ROOT": ROOT, "PROJECT_ROOT": ROOT}
    # Only `unsloth` is stubbed. torch and transformers are genuinely
    # installed locally, and stubbing them made scipy's array-type detection
    # raise inside pandas - a harness artifact that told us nothing about
    # our code. Using the real packages is both simpler and more faithful.
    stubs = {"unsloth": stub_unsloth()}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    # Parse against the script's REAL parser so declared required flags are
    # satisfied and unknown flags are rejected exactly as in production.
    src = src.replace("ap.parse_args()", f"ap.parse_args({REQUIRED_FLAGS.get(name, [])})")
    old_argv = sys.argv
    sys.argv = [name]
    try:
        exec(compile(src, path, "exec"), g)
        outcome = "completed without reaching model load"
        passed = True
    except SystemExit as e:
        if SENTINEL in str(e):
            outcome = "reached model load (expected)"
            passed = True
        else:
            outcome = f"SystemExit: {e}"
            passed = False
    except Exception as e:
        outcome = f"{type(e).__name__}: {e}"
        passed = False
    finally:
        sys.argv = old_argv
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    check(f"{name}", passed, outcome)

print()
print("3. COMPILE ALL")
print("-" * 72)
for path in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py"))
                   + glob.glob(os.path.join(ROOT, "server", "*.py"))):
    src = open(path, encoding="utf-8").read()
    try:
        compile(src, path, "exec")
        check(os.path.relpath(path, ROOT).replace("\\", "/"), True)
    except SyntaxError as e:
        check(os.path.relpath(path, ROOT).replace("\\", "/"), False, str(e))

print()
print("=" * 72)
print("ALL CHECKS PASS" if ok else "FAILURES PRESENT")
print("=" * 72)
sys.exit(0 if ok else 1)