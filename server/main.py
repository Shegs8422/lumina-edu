"""
Lumina-Edu inference API.

Replaces the blueprint's server, which returned a hard-coded sentence instead
of running the model:

    response_text = (f"Lumina-Edu Analysis [{payload.target_language}]: "
                     f"Regarding '{payload.query}', institutional repository "
                     f"metrics show strong alignment ...")

That is not inference. It fabricates a scholarly-sounding answer with no model
in the loop, which is exactly what the NAIC's "not simulated or hypothetical
results" rule prohibits.

This module loads the real N-ATLaS base with the Lumina-Edu LoRA adapter and
generates every response.

Deployment note
---------------
An 8B model in 4-bit needs roughly 6 GB of memory. Render's free tier offers
512 MB, so the blueprint's "deploy to Render ($0)" plan cannot work. Deploy
to a Hugging Face Space with ZeroGPU, or run behind a Kaggle/Colab tunnel.

Licence: outputs must carry the attribution string required by the N-ATLaS
Terms of Use. See ATTRIBUTION.md.
"""

import os
import time
from datetime import datetime
from typing import Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_ID = "NCAIR1/N-ATLaS"
ADAPTER_ID = os.environ.get("LUMINA_ADAPTER", "lumina-edu-natlas-adapter-powered-by-awarri")
MAX_SEQ_LENGTH = 2048
NEW_TOKENS = int(os.environ.get("LUMINA_MAX_NEW_TOKENS", "512"))

ATTRIBUTION = ("N-ATLaS is an initiative of the Federal Ministry of "
               "Communications, Innovation and Digital Economy, and powered "
               "by Awarri Technologies.")

SYSTEM_PROMPT = (
    "You are Lumina-Edu, an academic research assistant for Nigerian higher "
    "education. Answer using the supplied context. If the context does not "
    "contain the answer, say so plainly rather than inventing a citation. "
    "When asked for a language other than English, reply in that language. "
    + ATTRIBUTION
)

app = FastAPI(
    title="Lumina-Edu API",
    description="N-ATLaS fine-tuned academic literature endpoint (NAIC 2026, PS3)",
    version="1.0.0",
)

# Restrict to the deployed frontend in production rather than "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_tokenizer = None


class QueryPayload(BaseModel):
    query: str = Field(..., min_length=1, description="Research question")
    context: str = Field("", description="Optional abstract / source passage")
    target_language: Literal["English", "Yoruba", "Hausa", "Igbo"] = "English"


def load_model():
    """Load base + adapter lazily so the app starts before weights are ready."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from unsloth import FastLanguageModel
    from peft import PeftModel

    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True)

    if os.path.exists(ADAPTER_ID):
        print(f"Applying LoRA adapter: {ADAPTER_ID}")
        model = PeftModel.from_pretrained(model, ADAPTER_ID)
    else:
        print(f"WARNING: adapter '{ADAPTER_ID}' not found. "
              "Serving BASE N-ATLaS. This is not Lumina-Edu.")

    model.eval()
    _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


@app.on_event("startup")
def startup():
    try:
        load_model()
    except Exception as exc:  # keep /health usable even if weights fail
        print(f"Model load failed: {exc}")


@app.get("/")
def root():
    return {
        "status": "online" if _model is not None else "loading",
        "system": "Lumina-Edu Academic Agent — Powered by Awarri",
        "base_model": MODEL_ID,
        "adapter": ADAPTER_ID if _model is not None else None,
        "languages": ["English", "Yoruba", "Hausa", "Igbo"],
        "attribution": ATTRIBUTION,
    }


@app.get("/health")
def health():
    return {"model_loaded": _model is not None}


@app.post("/api/v1/query")
async def query(payload: QueryPayload):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        model, tokenizer = load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")

    user = payload.query
    if payload.context.strip():
        user += f"\n\nContext:\n{payload.context}"
    if payload.target_language != "English":
        user += f"\n\nRespond in {payload.target_language}."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            date_string=datetime.now().strftime("%d %b %Y"))
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)

    started = time.time()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=NEW_TOKENS, do_sample=False,
            repetition_penalty=1.1, use_cache=True)
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    return {
        "success": True,
        "query": payload.query,
        "target_language": payload.target_language,
        "response": completion,
        "model_version": "N-ATLaS-Lumina-Edu-v1-LoRA",
        "base_model": MODEL_ID,
        "latency_ms": int((time.time() - started) * 1000),
        "attribution": ATTRIBUTION,
    }