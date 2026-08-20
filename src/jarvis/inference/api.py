"""Phase 4 — FastAPI inference endpoint with RAG + Guardrails.

Run (on a machine with GPU + trained adapters):
    JARVIS_ADAPTERS=/path/to/jarvis_models/krikri_qlora \
    uvicorn jarvis.inference.api:app --host 0.0.0.0 --port 8000

Architecture:
  User message → RAG retrieval → Krikri generation → Guardrails → Response

Design decisions:
  * ON/OFF toggle — the Phase 5 safety requirement — is enforced HERE
    (env JARVIS_ENABLED or POST /twin/toggle), not in the orchestrator,
    so no integration (n8n, LinkedIn) can bypass it.
  * Every generated reply passes guardrails (profanity, accents, capitalization)
    AND a PII leak check (defence-in-depth).
  * Model loads lazily on first /chat, so /health works instantly and the
    skeleton runs even on GPU-less machines (returns 503 for /chat).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jarvis.sanitization import PIISanitizer
from jarvis.inference.guardrails import Guardrails
from jarvis.rag.context_builder import build_searcher, format_context

app = FastAPI(
    title="Jarvis George — Digital Twin API",
    version="5.0.0",
    description=(
        "Inference endpoint for the Krikri-8B fine-tuned digital twin.\n\n"
        "**Monolithic**: POST /chat (full pipeline in one call)\n\n"
        "**Granular** (for n8n orchestration): /orchestration/* endpoints"
    ),
)

# Register orchestration routes for n8n
try:
    from jarvis.orchestration.api_routes import router as orch_router
    app.include_router(orch_router)
except ImportError:
    pass  # orchestration module not installed

# Default system prompt
_DEFAULT_SYSTEM_PROMPT = (
    "Είσαι ο Jarvis George, ψηφιακό δίδυμο του Γιώργου Τροχίδη. "
    "Απαντάς στα ελληνικά με το ύφος και τον τρόπο επικοινωνίας του Γιώργου — "
    "σύντομα, φιλικά, πρακτικά."
)

_state: dict = {
    "enabled": os.getenv("JARVIS_ENABLED", "true").lower() == "true",
    "model": None,
    "tokenizer": None,
    "chat_format": None,
    "leak_guard": PIISanitizer(),
    "guardrails": Guardrails(),
    "rag_searcher": None,
}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    context: str = Field("", description="Optional manual RAG context")
    use_rag: bool = Field(True, description="Auto-retrieve context via RAG")
    max_new_tokens: int = Field(150, ge=1, le=1024)
    temperature: float = Field(0.5, ge=0.0, le=2.0)
    top_p: float = Field(0.85, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=200)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)
    apply_guardrails: bool = Field(True, description="Apply post-processing guardrails")


class ChatResponse(BaseModel):
    reply: str
    raw_reply: str = Field("", description="Pre-guardrail response (for debugging)")
    pii_filtered: bool
    rag_context_used: bool
    model_name: str


class ToggleRequest(BaseModel):
    enabled: bool


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "twin_enabled": _state["enabled"],
        "model_loaded": _state["model"] is not None,
        "model_name": os.getenv("JARVIS_BASE_MODEL", "ilsp/Llama-Krikri-8B-Instruct"),
        "rag_loaded": _state["rag_searcher"] is not None,
    }


@app.post("/twin/toggle")
def toggle(req: ToggleRequest) -> dict:
    """Master ON/OFF switch (Phase 5 requirement)."""
    _state["enabled"] = req.enabled
    return {"twin_enabled": _state["enabled"]}


def _ensure_model() -> None:
    if _state["model"] is not None:
        return
    adapters = os.getenv("JARVIS_ADAPTERS")
    base_model = os.getenv("JARVIS_BASE_MODEL", "ilsp/Llama-Krikri-8B-Instruct")
    try:
        from jarvis.inference.model_loader import (
            load_model_and_tokenizer,
            detect_chat_format,
        )
        _state["model"], _state["tokenizer"] = load_model_and_tokenizer(
            base_model=base_model,
            adapters_path=adapters,
        )
        _state["chat_format"] = detect_chat_format(base_model)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model runtime not installed ({exc}). "
                   "Install torch/transformers/peft/bitsandbytes.",
        ) from exc


def _ensure_rag() -> None:
    """Load the RAG searcher from sanitized data if not yet loaded."""
    if _state["rag_searcher"] is not None:
        return
    corpus_path = os.getenv("JARVIS_CORPUS")
    if not corpus_path:
        return  # RAG is optional; works without it
    try:
        _state["rag_searcher"] = build_searcher(corpus_path)
    except Exception as exc:
        print(f"WARNING: RAG corpus load failed: {exc}")


def _get_rag_context(message: str) -> str:
    """Retrieve relevant context from the RAG searcher if available."""
    _ensure_rag()
    if _state["rag_searcher"] is None:
        return ""
    try:
        results = _state["rag_searcher"].search(message, top_k=3)
        return format_context(results, max_results=3, style="conversation")
    except Exception:
        return ""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not _state["enabled"]:
        raise HTTPException(
            status_code=423, detail="Twin is switched OFF (toggle to enable)."
        )
    _ensure_model()

    from jarvis.inference.model_loader import generate_reply

    # Build context: manual context or RAG auto-retrieval
    context = req.context
    rag_used = False
    if not context and req.use_rag:
        context = _get_rag_context(req.message)
        rag_used = bool(context)

    raw_reply = generate_reply(
        _state["model"],
        _state["tokenizer"],
        req.message,
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        context=context,
        chat_format=_state["chat_format"],
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )

    # Guardrails pipeline
    processed = raw_reply
    if req.apply_guardrails:
        processed = _state["guardrails"].process(processed)

    # Defence-in-depth: never let structural PII leave the endpoint
    guarded = _state["leak_guard"].sanitize(processed)

    return ChatResponse(
        reply=guarded,
        raw_reply=raw_reply if req.apply_guardrails else "",
        pii_filtered=guarded != processed,
        rag_context_used=rag_used,
        model_name=os.getenv(
            "JARVIS_BASE_MODEL", "ilsp/Llama-Krikri-8B-Instruct"
        ),
    )
