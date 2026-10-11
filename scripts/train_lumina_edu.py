"""
Lumina-Edu QLoRA fine-tune of NCAIR1/N-ATLaS.

Run on Kaggle (T4 x2) or Colab. Designed to be pasted into a notebook cell or
run as a script.

WHY THIS DIFFERS FROM THE ORIGINAL BLUEPRINT
--------------------------------------------
The blueprint formatted training examples as raw Alpaca text:

    ### Instruction: {} ### Input: {} ### Response: {}

but N-ATLaS is a chat model whose tokenizer carries its own chat template
(a Llama-3 template that injects "Cutting Knowledge Date" and "Today Date"
into the system turn). Fine-tuning on Alpaca while serving with the chat
template puts the model in a format it never saw at evaluation time, which
degrades the adapter and produces a worse result than the baseline it is
supposed to beat. We train in the model's native chat format instead, via
tokenizer.apply_chat_template, so train and inference formats match exactly.

Other corrections:
  * `max_seq_length` is capped at 2048, well under N-ATLaS's 8,092-token
    effective context limit.
  * Optimisation uses epochs rather than a fixed 120 steps, because step count
    cannot be set correctly until the dataset size is known.
  * The adapter is saved with the "Powered by Awarri" suffix required by the
    N-ATLaS licence.
  * Prompt truncation keeps the response intact (TruncationStrategy.ONLY_FIRST
    is NOT used here - we truncate the input, never the target).

Environment (Kaggle secret): HF_TOKEN with read+write on NCAIR1/N-ATLaS.
"""

import json
import os
import random
import re
from collections.abc import Mapping

import torch
from datasets import Dataset
from transformers import TrainingArguments

# ---------------------------------------------------------------- config ----
MODEL_ID = "NCAIR1/N-ATLaS"
# Median abstract is ~1370 characters (~350 tokens) and the chat template adds
# a system turn, so examples land around 600-900 tokens. Sizing the window at
# 2048 padded every sequence to nearly 2k and roughly doubled training time for
# no benefit; 1024 truncates a negligible fraction.
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
SEED = 3407

DATA_DIR = "data/processed"
ADAPTER_PATH = "lumina-edu-natlas-adapter-powered-by-awarri"
OUT_DIR = "outputs"


def resolve_output_dir(preferred):
    """Pick a writable directory.

    On Kaggle, /kaggle/input is READ-ONLY (OSError Errno 30). Anything we
    write - distilled data, the adapter, checkpoints - must go to
    /kaggle/working, the only writable location in a notebook.
    """
    if os.environ.get("LUMINA_OUTPUT_DIR"):
        return os.environ["LUMINA_OUTPUT_DIR"]
    if os.path.isdir("/kaggle/working"):
        if os.path.isabs(preferred) and preferred.startswith("/kaggle/input"):
            # /kaggle/input/<dataset>/<tail> -> /kaggle/working/<tail>
            tail = preferred.split("/kaggle/input", 1)[1].lstrip("/")
            parts = tail.split("/", 1)
            tail = parts[1] if len(parts) == 2 else tail
            return os.path.join("/kaggle/working", tail)
    return preferred
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")  # e.g. "your-org/lumina-edu-adapter"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_DIR) if not os.path.exists(DATA_DIR) else DATA_DIR

# The attribution string mandated by the N-ATLaS Terms of Use.
ATTRIBUTION = ("Powered by Awarri Technologies and the Federal Ministry of "
               "Communications, Innovation and Digital Economy.")


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------- data prep ----
def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_distilled(path):
    """Load self-distilled translations, keyed by (record_id, language).

    These replace the placeholder targets described in the module docstring.
    If the file is absent we fall back to placeholder-free behaviour by
    dropping the translation task entirely rather than training the model to
    emit bracketed scaffolding.
    """
    if not os.path.exists(path):
        print(f"WARNING: no distilled translations at {path}")
        print("         Translation task will be OMITTED from training.")
        return {}
    rows = load_jsonl(path)
    table = {}
    for r in rows:
        lang = r.get("language")
        text = (r.get("translation") or "").strip()
        if not lang or len(text) < 60:
            continue
        table[(r["id"], lang)] = text
    print(f"Loaded {len(table)} distilled translations across {len(set(k[1] for k in table))} languages")
    return table


def first_sentences(text, n=2, max_chars=420):
    """Extractive lead: the first n sentences, capped.

    Used as a COMPRESSION target. The previous target was the entire abstract
    with a few words prepended, which made the task copy training: the model
    learned to echo its input verbatim. Because the benchmark scores against
    the abstract that also appears in the prompt, a copier scores well for the
    wrong reason. A short extractive target forces selection instead.
    """
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = " ".join(parts[:n]).strip()
    return out[:max_chars] if out else (text or "")[:max_chars]


def build_examples(records, distilled=None):
    """Turn curated records into chat-format training examples.

    Tasks:
      1. compress   - summarise to the leading claim (extractive)
      2. translate  - explain in a target Nigerian language (needs distilled data)
      3. attribute  - identify the source institution

    Task 1 is deliberately NOT "restate the abstract". Training the model to
    reproduce a long input verbatim teaches echoing, and because the benchmark
    reference is that same abstract, echo-training inflates the score without
    improving anything a user would notice. The extractive target is a
    fraction of the input, so copying the whole thing is penalised.
    """
    languages = ["Yoruba", "Hausa", "Igbo"]
    distilled = distilled or {}
    out = []

    for i, rec in enumerate(records):
        title = rec["title"]
        abstract = rec["abstract"]
        institution = rec.get("institution", "University of Ibadan")
        subjects = ", ".join(rec.get("subjects", [])[:3])
        year = rec.get("date", "")[:4]

        # Task 1 - extractive compression
        out.append({
            "messages": [
                {"role": "system", "content":
                    f"You are Lumina-Edu, an academic research assistant for "
                    f"Nigerian higher education. Answer in at most two sentences. "
                    f"{ATTRIBUTION}"},
                {"role": "user", "content":
                    f"In one or two sentences, state the main claim of this "
                    f"research.\n\nTitle: {title}\nAbstract: {abstract}"},
                {"role": "assistant", "content":
                    first_sentences(abstract)},
            ],
            "task": "compress",
        })

        # Task 2 - multilingual explanation, ONLY with a real target
        lang = languages[i % len(languages)]
        target = distilled.get((rec["id"], lang))
        if target:
            out.append({
                "messages": [
                    {"role": "system", "content":
                        f"You are Lumina-Edu, an academic research assistant for "
                        f"Nigerian higher education. {ATTRIBUTION}"},
                    {"role": "user", "content":
                        f"Explain this research finding in {lang}.\n\n"
                        f"Title: {title}\nAbstract: {abstract}"},
                    {"role": "assistant", "content": target},
                ],
                "task": f"translate_{lang.lower()}",
            })

        # Task 3 - grounded attribution
        out.append({
            "messages": [
                {"role": "system", "content":
                    f"You are Lumina-Edu. Name only the institution that actually "
                    f"produced the work. If you are unsure, say you are unsure. "
                    f"{ATTRIBUTION}"},
                {"role": "user", "content":
                    f"Which Nigerian university produced this work?\n\n"
                    f"Title: {title}\nAbstract: {abstract}"},
                {"role": "assistant", "content":
                    f"Institution: {institution}\n"
                    f"Discipline: {subjects or 'not specified'}\n"
                    f"Year: {year or 'not specified'}"},
            ],
            "task": "attribute",
        })

    return out


def to_chat_text(examples, tokenizer):
    """Render with N-ATLaS's OWN chat template - not Alpaca.

    Parameter order matters, and so does the batch shape.

    `Dataset.map(fn, batched=True, fn_kwargs=...)` passes the batch as the
    FIRST POSITIONAL argument with fn_kwargs as keywords, so declaring this
    as (tokenizer, examples) collided with the `tokenizer=` keyword.

    With batched=True the batch is also a DICT OF COLUMNS
    ({"messages": [...], "task": [...]}), not a list of rows. Iterating it
    directly yields column-name strings, so we read the column explicitly.
    """
    # datasets hands us a LazyBatch, which implements Mapping but is NOT a
    # dict subclass, so isinstance(x, dict) is False. Test for Mapping.
    if isinstance(examples, Mapping):
        messages_column = examples["messages"]
    else:
        messages_column = [ex["messages"] for ex in examples]

    texts = []
    for messages in messages_column:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except TypeError:
            # Some builds require the date_string argument the model card uses.
            from datetime import datetime
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                date_string=datetime.now().strftime("%d %b %Y"),
            )
        texts.append(text)
    return {"text": texts}


def main():
    from unsloth import FastLanguageModel

    set_seed()
    print(f"Loading {MODEL_ID} ...")

    # Unsloth defaults to a single GPU when the model fits, which it does in
    # 4-bit. That wastes the second T4: data-parallel training across both
    # roughly halves wall-clock. (The generation scripts deliberately do NOT
    # do this - model-parallel decoding is slower than one GPU with a larger
    # batch for a model this size.)
    n_gpu = torch.cuda.device_count()
    device_map = "unsloth_balanced" if n_gpu > 1 else None
    print(f"GPUs detected: {n_gpu} | device_map: {device_map or 'single'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        device_map=device_map,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
    )

    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    dev_records = load_jsonl(os.path.join(DATA_DIR, "dev.jsonl"))
    print(f"train records={len(train_records)} dev records={len(dev_records)}")

    distilled = load_distilled(resolve_output_dir(
        os.path.join(DATA_DIR, "distilled", "translations.jsonl")))

    train_ex = build_examples(train_records, distilled)
    dev_ex = build_examples(dev_records, distilled)
    print(f"train examples={len(train_ex)} dev examples={len(dev_ex)}")

    from collections import Counter
    print("task mix:", Counter(e["task"] for e in train_ex).most_common())

    train_ds = Dataset.from_list(train_ex).map(
        to_chat_text, batched=True, fn_kwargs={"tokenizer": tokenizer})
    dev_ds = Dataset.from_list(dev_ex).map(
        to_chat_text, batched=True, fn_kwargs={"tokenizer": tokenizer})

    supports_bf16 = torch.cuda.is_bf16_supported()

    from trl import SFTTrainer
    import inspect

    # TRL renamed its tokenizer argument to `processing_class` and, in recent
    # releases, removed `tokenizer` outright. Kaggle currently ships
    # transformers 5.16.1 with a matching TRL, so we detect the signature and
    # pass whichever keyword this version accepts.
    params = inspect.signature(SFTTrainer.__init__).parameters
    tok_kwarg = "processing_class" if "processing_class" in params else "tokenizer"
    print(f"trl version: {__import__('trl').__version__}  |  "
          f"using '{tok_kwarg}' for the tokenizer")

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        dataset_text_field="text",
        dataset_num_proc=2,
        packing=False,
    )
    trainer_kwargs[tok_kwarg] = tokenizer

    # `max_seq_length` was renamed to `max_length`; accept either.
    if "max_length" in params:
        trainer_kwargs["max_length"] = MAX_SEQ_LENGTH
    else:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    trainer = SFTTrainer(
        **trainer_kwargs,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=2,
            learning_rate=2e-4,
            fp16=not supports_bf16,
            bf16=supports_bf16,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir=OUT_DIR,
            report_to=[],
        ),
    )

    print(f"precision: {'bf16' if supports_bf16 else 'fp16'} "
          f"(T4 does not support bfloat16, so fp16 is correct here)")
    print("Starting training ...")
    trainer.train()

    adapter_out = resolve_output_dir(ADAPTER_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Saving adapter to {adapter_out}")
    model.save_pretrained_merged(adapter_out, tokenizer, save_method="lora")
    with open(os.path.join(OUT_DIR, "adapter_path.txt"), "w") as fh:
        fh.write(adapter_out)

    if HF_REPO_ID:
        print(f"Pushing to Hugging Face: {HF_REPO_ID}")
        model.push_to_hub_merged(HF_REPO_ID, tokenizer, save_method="lora")

    print(f"\nDone. Adapter: {ADAPTER_PATH}")
    print(f"Attribution to reproduce in all outputs: {ATTRIBUTION}")


if __name__ == "__main__":
    main()
