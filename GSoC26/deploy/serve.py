"""Minimal extraction endpoint — DRAFT.

Wraps the frozen pipeline's extract_all_triples() behind one HTTP call:

    POST /extract   {"text": "The song Mermaid is by the band Train."}
    → {"triples": [{"sub": ..., "rel": ..., "obj": ..., "status": ...}, ...]}

Every triple carries its verification status (verified / faithful-unverified),
so downstream users can filter by trust level.
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

os.environ.setdefault("USE_TF", "0")
import autonomous_pipeline_v13 as pl

app = FastAPI(title="Neural Extraction Framework 2.0")


class ExtractRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True, "model": pl.TARGET_MODEL}


@app.post("/extract")
def extract(req: ExtractRequest):
    if not pl.OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured — set it in deploy/.env and restart.",
        )
    results = pl.extract_all_triples(req.text)
    return {"triples": results}
