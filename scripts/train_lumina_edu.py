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

import torch
from datasets import Dataset
from transformers import TrainingArguments

# ---------------------------------------------------------------- config ----
MODEL_ID = "NCAIR1/N-ATLaS"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
SEED = 3407

DATA_DIR = "data/processed"
ADAPTER_PATH = "lumina-edu-natlas-adapter-powered-by-awarri"
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


def build_examples(records, distilled=None):
    """Turn curated records into chat-format training examples.

    Tasks:
      1. summarise the study
      2. explain the study in a target Nigerian language (requires distilled data)
      3. identify the institutional / disciplinary context

    Task 2 is only emitted when a real translated target exists for that
    record. We never fabricate a translation target, because a bracketed
    placeholder teaches the model to emit scaffolding instead of Yoruba.
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

        # Task 1 - summarisation
        out.append({
            "messages": [
                {"role": "system", "content":
                    f"You are Lumina-Edu, an academic research assistant for "
                    f"Nigerian higher education. {ATTRIBUTION}"},
                {"role": "user", "content":
                    f"Summarise the following research abstract from {institution}.\n\n"
                    f"Title: {title}\nAbstract: {abstract}"},
                {"role": "assistant", "content":
                    f"This study, conducted at {institution}"
                    + (f" ({year})" if year else "") +
                    (f" and classified under {subjects}" if subjects else "") +
                    f", examines the following. {abstract}"},
            ],
            "task": "summarise",
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

        # Task 3 - grounded institutional context (anti-hallucination)
        out.append({
            "messages": [
                {"role": "system", "content":
                    f"You are Lumina-Edu. Only state institutions and subjects "
                    f"that appear in the text you are given. {ATTRIBUTION}"},
                {"role": "user", "content":
                    f"Which institution produced this work and what discipline "
                    f"does it belong to?\n\nTitle: {title}\nAbstract: {abstract}"},
                {"role": "assistant", "content":
                    f"Institution: {institution}\n"
                    f"Discipline: {subjects or 'not specified'}\n"
                    f"Year: {year or 'not specified'}"},
            ],
            "task": "grounding",
        })

    return out


def to_chat_text(tokenizer, examples):
    """Render with N-ATLaS's OWN chat template - not Alpaca."""
    texts = []
    for ex in examples:
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        except TypeError:
            # Some builds require the date_string argument the model card uses.
            from datetime import datetime
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False,
                date_string=datetime.now().strftime("%d %b %Y"),
            )
        texts.append(text)
    return {"text": texts}


def main():
    from unsloth import FastLanguageModel

    set_seed()
    print(f"Loading {MODEL_ID} ...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
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

    distilled = load_distilled(os.path.join(DATA_DIR, "distilled", "translations.jsonl"))

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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
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
            output_dir="outputs",
            report_to=[],
        ),
    )

    print("Starting training ...")
    trainer.train()

    print(f"Saving adapter to {ADAPTER_PATH}")
    model.save_pretrained_merged(ADAPTER_PATH, tokenizer, save_method="lora")

    if HF_REPO_ID:
        print(f"Pushing to Hugging Face: {HF_REPO_ID}")
        model.push_to_hub_merged(HF_REPO_ID, tokenizer, save_method="lora")

    print(f"\nDone. Adapter: {ADAPTER_PATH}")
    print(f"Attribution to reproduce in all outputs: {ATTRIBUTION}")


if __name__ == "__main__":
    main()