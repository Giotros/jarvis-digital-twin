"""Granular API routes for n8n orchestration.

These endpoints expose each pipeline step individually, so n8n can
orchestrate them as separate nodes in a visual workflow:

  Webhook → /orchestration/intent     → classify message
         → /orchestration/identity    → lookup static identity
         → /orchestration/rag         → search conversation corpus
         → /orchestration/email       → search Gmail for recent events
         → /orchestration/calendar    → check Google Calendar availability
         → /orchestration/generate    → call Ollama/Krikri
         → /orchestration/guardrails  → post-process response
         → /orchestration/fact-check  → validate against identity

The monolithic /chat endpoint (api.py) still works for direct use.
These routes add the granular alternative for n8n workflows.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from jarvis.orchestration.intent_classifier import classify_intent, Intent
from jarvis.inference.guardrails import Guardrails
from jarvis.inference.identity import load_identity, identity_to_prompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestration", tags=["orchestration"])

# ── Shared state (lazy-loaded) ──────────────────────────────────

_guardrails = Guardrails()
_identity: dict | None = None
_identity_prompt: str | None = None


def _get_identity() -> tuple[dict, str]:
    """Load identity data (cached)."""
    global _identity, _identity_prompt
    if _identity is None:
        _identity = load_identity()
        _identity_prompt = identity_to_prompt(_identity)
    return _identity, _identity_prompt


# ── Request/Response models ─────────────────────────────────────

class IntentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    matched_keywords: list[str]
    explanation: str


class IdentityRequest(BaseModel):
    query: str = Field(..., description="What to look up in identity")


class IdentityResponse(BaseModel):
    identity_prompt: str
    raw_identity: dict
    relevant_section: str = ""


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class RAGResponse(BaseModel):
    context: str
    num_results: int
    results: list[dict] = []
    #: "ok" | "no_corpus_configured" | "corpus_missing" | "search_failed"
    #: Never silently empty: downstream nodes and the thesis evaluation
    #: must be able to distinguish "no relevant match" from "RAG is broken".
    status: str = "ok"
    detail: str = ""


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class GenerateRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list, description="Previous conversation turns")
    system_prompt: str = Field("")
    context: str = Field("", description="RAG or identity context to inject")
    max_new_tokens: int = Field(150, ge=1, le=1024)
    temperature: float = Field(0.5, ge=0.0, le=2.0)
    top_p: float = Field(0.85, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=200)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)


class GenerateResponse(BaseModel):
    reply: str
    model: str
    tokens_generated: int = 0


class GuardrailsRequest(BaseModel):
    text: str = Field(..., min_length=1)


class GuardrailsResponse(BaseModel):
    processed: str
    original: str
    changed: bool


class FactCheckRequest(BaseModel):
    response: str = Field(..., description="Generated response to validate")
    identity_prompt: str = Field("", description="Identity to check against")


class FactCheckResponse(BaseModel):
    passed: bool
    issues: list[str] = []
    cleaned_response: str


# ── Endpoints ───────────────────────────────────────────────────

@router.post("/intent", response_model=IntentResponse)
def classify(req: IntentRequest) -> IntentResponse:
    """Classify message intent for routing."""
    result = classify_intent(req.message)
    return IntentResponse(
        intent=result.intent.value,
        confidence=result.confidence,
        matched_keywords=result.matched_keywords,
        explanation=result.explanation,
    )


@router.post("/identity", response_model=IdentityResponse)
def identity_lookup(req: IdentityRequest) -> IdentityResponse:
    """Look up George's identity for personal questions."""
    identity_data, prompt = _get_identity()

    # Try to find the most relevant section
    query_lower = req.query.lower()
    relevant = ""

    section_map = {
        "personal": ["ονομα", "όνομα", "ηλικια", "ηλικία", "χρονων", "χρονών",
                      "γεννη", "γιαννιτσα", "γιαννιτσά"],
        "education": ["σπουδ", "πτυχι", "πτυχί", "μεταπτυχ", "διπλωματ",
                       "πανεπιστημ", "κερκυρα", "κέρκυρα"],
        "career": ["δουλει", "δουλειά", "εργασ", "εταιρ", "εταιρία",
                    "manager", "nova", "e-avenue"],
        "military": ["στρατ", "κυπρο", "κύπρο"],
        "hobbies": ["hobby", "hobbies", "χομπ", "τρεξιμ", "τρέξιμ"],
        "technical_skills": ["skill", "python", "aws", "cloud", "docker",
                              "react", "javascript", "τεχνολογ"],
        "projects": ["project", "jarvis", "obsidian", "b2b"],
    }

    for section, keywords in section_map.items():
        if any(kw in query_lower for kw in keywords):
            section_data = identity_data.get(section, {})
            relevant = yaml.dump(section_data, allow_unicode=True, default_flow_style=False)
            break

    return IdentityResponse(
        identity_prompt=prompt,
        raw_identity=identity_data,
        relevant_section=relevant,
    )


@router.post("/rag", response_model=RAGResponse)
def rag_search(req: RAGRequest) -> RAGResponse:
    """Search the RAG corpus for relevant conversation context."""
    corpus_path = os.getenv("JARVIS_CORPUS")
    if not corpus_path:
        logger.error("RAG disabled: JARVIS_CORPUS is not set")
        return RAGResponse(
            context="", num_results=0,
            status="no_corpus_configured",
            detail="JARVIS_CORPUS environment variable is not set.",
        )

    if not Path(corpus_path).exists():
        logger.error("RAG corpus missing at %s — twin will hallucinate", corpus_path)
        return RAGResponse(
            context="", num_results=0,
            status="corpus_missing",
            detail=f"Corpus file not found: {corpus_path}",
        )

    try:
        from jarvis.rag.context_builder import build_searcher, format_context
        searcher = build_searcher(corpus_path)
        results = searcher.search(req.query, top_k=req.top_k)
        context = format_context(results, max_results=req.top_k, style="conversation")

        result_dicts = [
            {
                "score": r.score,
                "instruction": r.metadata.get("instruction", ""),
                "response": r.metadata.get("response", ""),
            }
            for r in results
        ]

        return RAGResponse(
            context=context,
            num_results=len(results),
            results=result_dicts,
            status="ok",
        )
    except Exception as exc:
        logger.exception("RAG search failed for query %r", req.query)
        return RAGResponse(
            context="", num_results=0,
            status="search_failed",
            detail=f"{type(exc).__name__}: {exc}",
        )


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate a response via Ollama API.

    In the Docker stack, this calls the Ollama container.
    The model must be pre-loaded (ollama pull krikri-8b).
    """
    primary_url = os.getenv("JARVIS_OLLAMA_URL", "http://localhost:11434")
    fallback_url = os.getenv("JARVIS_OLLAMA_FALLBACK", "http://ollama:11434")
    model_name = os.getenv("JARVIS_MODEL", "krikri-8b")
    fallback_model = os.getenv("JARVIS_FALLBACK_MODEL", "mistral")

    # Build the full prompt with context
    system = req.system_prompt or (
        "Είσαι ο Γιώργος Τροχίδης, φοιτητής στο Πανεπιστήμιο Πελοποννήσου στην Τρίπολη.\n"
        "Μιλάς ΠΑΝΤΑ σε πρώτο ενικό πρόσωπο — εσύ ΕΙΣΑΙ ο Γιώργος, ΟΧΙ AI ή βοηθός.\n"
        "ΥΦΟΣ: Φιλικό, casual, σαν να μιλάς σε φίλο. Χρησιμοποίεις greeklish και αργκό αν ταιριάζει.\n"
        "ΜΗΚΟΣ: Απάντα σε 1-3 προτάσεις. Μην είσαι μονολεκτικός — δώσε λίγη προσωπικότητα.\n"
        "ΚΑΝΟΝΕΣ: Μην επινοείς πληροφορίες. Αν δεν ξέρεις κάτι, πες 'δεν θυμάμαι' ή 'δεν ξέρω'.\n"
        "CONTEXT: Αν υπάρχει ιστορικό συνομιλίας, χρησιμοποίησέ το για να απαντήσεις σχετικά."
    )

    if req.context:
        system += f"\n\nΣΧΕΤΙΚΟ CONTEXT:\n{req.context}"

    try:
        # Build messages with conversation history
        messages = [{"role": "system", "content": system}]
        for msg in req.history[-10:]:  # Keep last 10 turns for context
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": req.message})

        # Try primary (Colab/ngrok), then fallback (local Ollama)
        used_model = model_name
        used_url = primary_url
        for url, model in [(primary_url, model_name), (fallback_url, fallback_model)]:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{url}/api/chat",
                        json={
                            "model": model,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": req.temperature,
                                "top_p": req.top_p,
                                "top_k": req.top_k,
                                "repeat_penalty": req.repetition_penalty,
                                "num_predict": req.max_new_tokens,
                            },
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return GenerateResponse(
                        reply=data.get("message", {}).get("content", ""),
                        model=model,
                        tokens_generated=data.get("eval_count", 0),
                    )
            except (httpx.ConnectError, httpx.HTTPStatusError):
                used_url = fallback_url
                used_model = fallback_model
                continue
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to any Ollama. Tried {primary_url} and {fallback_url}.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Feedback & Conversation Logging ────────────────────────────

FEEDBACK_LOG = Path(os.getenv("JARVIS_FEEDBACK_LOG", "/app/data/feedback_log.jsonl"))


class FeedbackRequest(BaseModel):
    message: str = Field(..., description="User's original message")
    reply: str = Field(..., description="Twin's response")
    intent: str = Field("")
    rating: int = Field(..., ge=-1, le=1, description="-1=bad, 0=neutral, 1=good")
    correction: str = Field("", description="User's preferred response (optional)")
    history: list[ChatMessage] = Field(default_factory=list)


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Log conversation feedback for future training."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": req.message,
        "reply": req.reply,
        "intent": req.intent,
        "rating": req.rating,
        "correction": req.correction if req.correction else None,
        "history": [{"role": m.role, "content": m.content} for m in req.history],
    }
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"status": "saved", "total": sum(1 for _ in open(FEEDBACK_LOG))}


@router.get("/feedback/export")
async def export_training_pairs():
    """Export positively-rated conversations as training pairs."""
    if not FEEDBACK_LOG.exists():
        return {"pairs": [], "total": 0}
    pairs = []
    with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["rating"] == 1:
                # Use correction if provided, otherwise use the original reply
                response = entry.get("correction") or entry["reply"]
                pairs.append({
                    "instruction": entry["message"],
                    "response": response,
                    "intent": entry["intent"],
                })
    return {"pairs": pairs, "total": len(pairs)}


# ── Email & Calendar endpoints ──────────────────────────────────

class EmailSearchRequest(BaseModel):
    query: str = Field(..., description="What to search for in emails")
    max_results: int = Field(5, ge=1, le=20)


class EmailResult(BaseModel):
    subject: str
    sender: str
    date: str
    snippet: str
    has_attachments: bool = False


class EmailSearchResponse(BaseModel):
    results: list[EmailResult]
    context: str = Field("", description="Formatted context for LLM prompt")
    num_results: int


class CalendarRequest(BaseModel):
    query: str = Field(..., description="Availability or meeting question")
    days_ahead: int = Field(7, ge=1, le=30, description="How many days to check")


class CalendarEvent(BaseModel):
    title: str
    start: str
    end: str
    location: str = ""
    is_all_day: bool = False


class CalendarResponse(BaseModel):
    events: list[CalendarEvent]
    free_slots: list[str] = Field(default_factory=list)
    context: str = Field("", description="Formatted context for LLM prompt")


@router.post("/email", response_model=EmailSearchResponse)
async def email_search(req: EmailSearchRequest) -> EmailSearchResponse:
    """Search Gmail for relevant emails.

    In production, this calls the Gmail API via n8n's Gmail node.
    This endpoint provides the API contract and can also call Gmail
    directly if GOOGLE_CREDENTIALS are configured.

    For n8n: this endpoint is optional — n8n can call Gmail directly
    via its built-in Gmail node. This exists for non-n8n deployments.
    """
    gmail_api_url = os.getenv("JARVIS_GMAIL_API_URL")

    if gmail_api_url:
        # Call Gmail proxy (n8n webhook or custom API)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    gmail_api_url,
                    json={"query": req.query, "max_results": req.max_results},
                )
                response.raise_for_status()
                data = response.json()

                results = [EmailResult(**r) for r in data.get("results", [])]
                context = _format_email_context(results)

                return EmailSearchResponse(
                    results=results,
                    context=context,
                    num_results=len(results),
                )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Gmail search failed: {exc}",
            )
    else:
        # No Gmail configured — return empty with explanation
        return EmailSearchResponse(
            results=[],
            context="[Email search not configured — connect Gmail in n8n]",
            num_results=0,
        )


@router.post("/calendar", response_model=CalendarResponse)
async def calendar_lookup(req: CalendarRequest) -> CalendarResponse:
    """Check Google Calendar for events and availability.

    In production, this calls the Google Calendar API via n8n's
    Calendar node. This endpoint provides the API contract.

    For n8n: this endpoint is optional — n8n can call Calendar directly
    via its built-in Google Calendar node. This exists for non-n8n use.
    """
    calendar_api_url = os.getenv("JARVIS_CALENDAR_API_URL")

    if calendar_api_url:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    calendar_api_url,
                    json={"query": req.query, "days_ahead": req.days_ahead},
                )
                response.raise_for_status()
                data = response.json()

                events = [CalendarEvent(**e) for e in data.get("events", [])]
                free_slots = data.get("free_slots", [])
                context = _format_calendar_context(events, free_slots)

                return CalendarResponse(
                    events=events,
                    free_slots=free_slots,
                    context=context,
                )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Calendar lookup failed: {exc}",
            )
    else:
        return CalendarResponse(
            events=[],
            free_slots=[],
            context="[Calendar not configured — connect Google Calendar in n8n]",
        )


def _format_email_context(results: list[EmailResult]) -> str:
    """Format email results as natural-language context for the LLM."""
    if not results:
        return ""
    lines = ["ΠΡΟΣΦΑΤΑ EMAILS:"]
    for r in results[:5]:
        lines.append(f"- {r.date}: {r.subject} (από {r.sender}) — {r.snippet}")
    return "\n".join(lines)


def _format_calendar_context(
    events: list[CalendarEvent], free_slots: list[str]
) -> str:
    """Format calendar data as natural-language context for the LLM."""
    lines = []
    if events:
        lines.append("ΠΡΟΓΡΑΜΜΑ:")
        for e in events[:10]:
            loc = f" ({e.location})" if e.location else ""
            lines.append(f"- {e.start} → {e.end}: {e.title}{loc}")
    if free_slots:
        lines.append("ΔΙΑΘΕΣΙΜΟΤΗΤΑ:")
        for slot in free_slots[:5]:
            lines.append(f"- {slot}")
    return "\n".join(lines) if lines else ""


@router.post("/guardrails", response_model=GuardrailsResponse)
def apply_guardrails(req: GuardrailsRequest) -> GuardrailsResponse:
    """Apply post-processing guardrails to generated text."""
    processed = _guardrails.process(req.text)
    return GuardrailsResponse(
        processed=processed,
        original=req.text,
        changed=processed != req.text,
    )


@router.post("/fact-check", response_model=FactCheckResponse)
def fact_check(req: FactCheckRequest) -> FactCheckResponse:
    """Validate a generated response against identity data.

    Checks for common hallucination patterns:
    - Programming languages not in George's skills
    - Wrong age
    - Wrong location claims
    - Made-up company names
    """
    identity_data, prompt = _get_identity()
    response_lower = req.response.lower()
    issues: list[str] = []

    # Known skills from identity
    known_skills = identity_data.get("technical_skills", {})
    all_skills_text = " ".join(str(v) for v in known_skills.values()).lower()

    # Languages George does NOT know — common hallucinations
    hallucinated_langs = ["golang", "go lang", "ruby", "rust", "swift",
                          "kotlin", "scala", "r ", "matlab", "c++", "c#",
                          "java ", "php"]
    for lang in hallucinated_langs:
        if lang in response_lower and lang not in all_skills_text:
            issues.append(f"Hallucinated skill: '{lang.strip()}' not in identity")

    # Age check
    correct_age = str(identity_data.get("personal", {}).get("age", 26))
    age_match = re.search(r"(\d{2})\s*(χρονων|χρονών|ετών|ετων)", response_lower)
    if age_match and age_match.group(1) != correct_age:
        issues.append(
            f"Wrong age: said {age_match.group(1)}, correct is {correct_age}"
        )

    # Company check — known companies
    known_companies = ["e-avenue", "isaakidis", "vat-group", "nova", "εφετείο"]
    company_patterns = re.findall(
        r"(?:δουλεύω|δούλευα|εργάζομαι|εργαζόμουν)\s+(?:στην?|σε)\s+(\w+)",
        response_lower,
    )
    for company in company_patterns:
        if not any(known in company.lower() for known in known_companies):
            issues.append(f"Unknown company mentioned: '{company}'")

    # Clean: remove hallucinated skills from response
    cleaned = req.response
    for lang in hallucinated_langs:
        if lang in response_lower and lang not in all_skills_text:
            # Remove the sentence containing the hallucination
            pattern = rf"[^.]*\b{re.escape(lang.strip())}\b[^.]*\."
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    return FactCheckResponse(
        passed=len(issues) == 0,
        issues=issues,
        cleaned_response=cleaned if issues else req.response,
    )
