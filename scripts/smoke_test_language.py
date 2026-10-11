"""Smoke test: verify N-ATLaS can actually produce target-language output.

If the base model cannot produce Yoruba/Hausa/Igbo, self-distillation is not
viable and we must fall back to English-only training. This is a cheap check
(loads the 4-bit model once, generates 3 prompts) that prevents committing
hours of GPU time to a doomed approach.
"""
import json, os, re, sys, unicodedata
from datetime import datetime
import torch

MODEL_ID = "NCAIR1/N-ATLaS"

# This script is executed via exec() inside the Kaggle notebook, where
# __file__ is undefined. Resolve the project root from DATA_ROOT when the
# notebook sets it, and fall back to __file__ when run directly.
if "__file__" in dir():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    PROJECT_ROOT = globals().get("DATA_ROOT", "/kaggle/working")
MARKERS = {
    "Yoruba": ["ati","ni","ko","wa","fun","won","yi","naa","kini","won"],
    "Hausa": ["da","na","ta","ya","ba","sun","kuma","wannan","shi","cikin"],
    "Igbo":  ["na","ya","ihe","ndi","ka","gi","maka","nke","a","buru"],
}

def strip_accents(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c))

recs = [json.loads(l) for l in open(os.path.join(PROJECT_ROOT,"data","processed","train.jsonl"),encoding="utf-8")][:2]

from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(model_name=MODEL_ID, max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)

for lang in ["Yoruba","Hausa","Igbo"]:
    rec = recs[0]
    msgs=[{"role":"system","content":f"You are a Nigerian academic translator. Translate into {lang}. Output only {lang}."},
          {"role":"user","content":f"Translate into {lang}:\n\nTitle: {rec['title']}\nAbstract: {rec['abstract'][:700]}"}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       date_string=datetime.now().strftime("%d %b %Y"))
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=200, do_sample=False, repetition_penalty=1.15)
    gen = tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    toks = set(re.findall(r"[a-zàáèéìíòóùúṣọ]+", strip_accents(gen).lower()))
    hits = [m for m in MARKERS[lang] if m in toks]
    print("="*70)
    print(f"{lang}   marker_hits={len(hits)} {hits[:6]}")
    print("-"*70)
    print(gen[:420])
    print()