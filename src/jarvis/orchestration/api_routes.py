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

from jarvis.orchestration import persona
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
    # Collected once per session by the client. Held for the lifetime of the
    # request only — never logged, never persisted, never added to the corpus.
    speaker_name: str = Field("", max_length=80, description="Interlocutor's first name")
    speaker_role: str = Field("", max_length=80, description="Relationship to George")
    max_new_tokens: int = Field(150, ge=1, le=1024)
    temperature: float = Field(0.5, ge=0.0, le=2.0)
    top_p: float = Field(0.85, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=200)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)


class GenerateResponse(BaseModel):
    reply: str
    model: str
    tokens_generated: int = 0
    #: Which register produced this reply. Reported so the behaviour is
    #: visible during a demo instead of being an unexplained tone shift.
    speaker_register: str = "neutral"
    #: True when the first draft contradicted the project facts and was
    #: regenerated. Surfaced rather than hidden so the evaluation chapter can
    #: count how often grounding alone fails — the number is a result, and a
    #: silent retry would make it unmeasurable.
    regenerated: bool = False


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


class PersonaRequest(BaseModel):
    speaker_name: str = Field("", max_length=80)
    speaker_role: str = Field("", max_length=80)


class PersonaResponse(BaseModel):
    speaker_register: str
    label: str
    target_words: int
    system_prompt: str


@router.post("/persona", response_model=PersonaResponse)
def persona_preview(req: PersonaRequest) -> PersonaResponse:
    """Show which register a given ιδιότητα selects, and the prompt it builds.

    Exists because the first version of this feature silently did nothing:
    the register was chosen correctly, the fragment was appended, and the
    replies came back identical, with no way to see where the instruction was
    lost. A feature whose only output is a tone shift needs somewhere to
    inspect its input.
    """
    prompt, register = persona.build_system_prompt(
        name=req.speaker_name, role=req.speaker_role
    )
    return PersonaResponse(
        speaker_register=register.name,
        label=register.label,
        target_words=register.target_words,
        system_prompt=prompt,
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


async def _reject_confabulation(
    *,
    client: "httpx.AsyncClient",
    url: str,
    model: str,
    messages: list[dict],
    reply: str,
    register: "persona.Register",
    options: dict,
) -> tuple[str, bool]:
    """Regenerate once when the reply contradicts the project facts.

    Why this exists at all: ``check_technical_claims`` was written, tested,
    and then wired only into ``/fact-check`` — an endpoint nothing in the
    workflow calls. The detector was correct and unreachable, so every
    confabulation it could recognise was served to the user anyway. Running
    the diagnostic script surfaced six invented answers in six attempts, all
    of which the detector flagged the moment it was pointed at them.

    The correction is one retry, not a rewrite. Editing a technical answer
    by rule would mean guessing what the sentence meant to say; asking the
    model again with the specific contradiction named is both safer and
    honest about what happened. If the second attempt also fails, the reply
    is replaced with a refusal rather than a third roll of the dice: an
    examiner hearing "δεν το θυμάμαι ακριβώς" loses a point, and an examiner
    hearing about a Kubernetes cluster that does not exist loses the thesis.

    Returns ``(reply, regenerated)``.
    """
    from jarvis.inference.thesis_facts import (
        check_acronym_expansions,
        check_corrupted_names,
        check_technical_claims,
        unsupported_technologies,
    )

    def _problems(text: str) -> list[str]:
        """Four checks, because each one is blind where the others see.

        * **Denylist** knows *why* a claim is wrong, and cannot see anything
          it has not been shown. Alone, it passed "Rust σε συνδυασμό με
          WebAssembly μέσω του actix-web".
        * **Allowlist** sees anything outside the facts, and cannot explain
          it. Alone, it passed "PEFT (PyTorch Elastic Framework)" — the tool
          is real and the gloss is invented.
        * **Acronyms** catch a wrong expansion of a right name.
        * **Near-miss** catches a right tool with wrong letters: "ChromeDB".

        Each was added after the previous set reported clean on an answer
        that was wrong. That progression is the argument for having four.
        """
        found = check_technical_claims(text)
        found.extend(check_acronym_expansions(text))
        found.extend(check_corrupted_names(text))
        unsupported = unsupported_technologies(text)
        if unsupported:
            found.append(
                "Δεν αναφέρονται στα στοιχεία της εργασίας: "
                + ", ".join(unsupported)
            )
        return found

    issues = _problems(reply)
    if not issues:
        return reply, False

    correction = (
        "Η προηγούμενη απάντησή σου περιείχε λάθος τεχνικά στοιχεία:\n"
        + "\n".join(f"— {issue}" for issue in issues)
        + "\n\nΞαναγράψε την απάντηση χωρίς αυτά, χρησιμοποιώντας ΜΟΝΟ όσα "
        "αναφέρονται στα στοιχεία της διπλωματικής. Αν δεν ξέρεις κάτι, πες "
        "το. Κράτα το ίδιο ύφος και το ίδιο μήκος."
    )

    try:
        response = await client.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [
                    *messages,
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": correction},
                ],
                "stream": False,
                "options": options,
            },
        )
        response.raise_for_status()
        second = _guardrails.sanitise_output(
            response.json().get("message", {}).get("content", ""),
            register.name,
        )
    except Exception:  # noqa: BLE001 — a failed retry must not lose the turn
        second = ""

    if second and not _problems(second):
        return second, True

    # Both attempts contradicted the facts. Refuse the technical claim rather
    # than serve either draft.
    return (
        "Δεν θέλω να πω κάτι λάθος για την υλοποίηση — προτιμώ να το "
        "κοιτάξω και να σου απαντήσω με ακρίβεια.",
        True,
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

    # The style section is generated from the register, not appended after a
    # hard-coded one. A caller-supplied system_prompt (the Memory and
    # Schedule nodes send one) becomes the *identity* half, so a routing
    # instruction and a register can coexist without contradicting each other.
    system, register = persona.build_system_prompt(
        name=req.speaker_name,
        role=req.speaker_role,
        identity=req.system_prompt,
        message=req.message,
    )

    if req.context:
        from jarvis.rag.context_builder import frame_context

        system += "\n\n" + frame_context(req.context)

    # Optional ablation: serve the academic register from the base model with
    # no adapter attached. Unset by default, so nothing changes unless the
    # experiment is deliberately switched on.
    #
    # The adapter learned to write like George on Viber, which is exactly what
    # the close register wants and exactly what fights a technical question:
    # it pulls towards short, loose and improvised, and it outweighs the
    # system prompt. Both models share one GGUF on disk, so the comparison
    # costs no extra download and no extra 5GB.
    academic_model = os.getenv("JARVIS_MODEL_ACADEMIC", "").strip()
    if academic_model and register is persona.ACADEMIC:
        model_name = academic_model

    # The register raises the token ceiling but never lowers it. Asking for
    # "2-4 πλήρεις προτάσεις" inside a 150-token budget produces a reply cut
    # off mid-sentence, which in front of an examiner reads as a crash. The
    # instruction and the budget have to agree, and the safe direction is up:
    # brevity is enforced by the prompt, not by truncation.
    temperature = req.temperature
    max_new_tokens = max(req.max_new_tokens, register.max_new_tokens)

    try:
        # Build messages with conversation history
        messages = [{"role": "system", "content": system}]

        # Register demonstrations go before the real history, so the model
        # reads them as how this conversation has been going. Placing them
        # after would make the user's own last turn the nearest example and
        # undo the effect.
        for q, a in register.examples:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

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
                                "temperature": temperature,
                                "top_p": req.top_p,
                                "top_k": req.top_k,
                                "repeat_penalty": req.repetition_penalty,
                                "num_predict": max_new_tokens,
                            },
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Register enforcement happens here rather than in the
                    # /guardrails node, which has no way of knowing who is
                    # being spoken to. Doing it here needs no workflow change
                    # and is idempotent, so the later node stays correct.
                    reply = _guardrails.sanitise_output(
                        data.get("message", {}).get("content", ""),
                        register.name,
                    )

                    reply, retried = await _reject_confabulation(
                        client=client,
                        url=url,
                        model=model,
                        messages=messages,
                        reply=reply,
                        register=register,
                        options={
                            "temperature": temperature,
                            "top_p": req.top_p,
                            "top_k": req.top_k,
                            "repeat_penalty": req.repetition_penalty,
                            "num_predict": max_new_tokens,
                        },
                    )

                    return GenerateResponse(
                        reply=reply,
                        model=model,
                        tokens_generated=data.get("eval_count", 0),
                        speaker_register=register.name,
                        regenerated=retried,
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


class BriefingRequest(BaseModel):
    #: Identities the caller has already delivered. Sent back by the client
    #: so the twin does not report the same unread email every hour.
    already_said: list[str] = Field(default_factory=list)
    threshold: float = Field(3.0, ge=0.0, le=20.0)


class BriefingResponse(BaseModel):
    #: Empty when there was nothing worth saying. The caller sends nothing.
    text: str
    #: "spoke" | "spoke_degraded" | "silent_below_threshold"
    #: | "silent_nothing_observed"
    status: str
    #: Identities to remember, so the next run can suppress them.
    identities: list[str] = []
    unavailable: list[str] = []


@router.post("/briefing", response_model=BriefingResponse)
async def briefing(req: BriefingRequest) -> BriefingResponse:
    """Assemble an unprompted brief, or decline to send one.

    Called on a schedule rather than by a user. The interesting output is
    often the empty one: a system that reports every morning regardless of
    whether anything happened gets muted, and once muted it fails silently
    while continuing to work.

    Tool failures degrade rather than abort. A brief that says "I could not
    check your email" is useful; one that says nothing because Gmail was
    down is indistinguishable from a quiet day, and those are opposite
    claims.
    """
    from jarvis.agency.briefing import build_briefing
    from jarvis.agency.signals import Signal, Source, Urgency

    signals: list[Signal] = []
    unavailable: list[Source] = []

    # Calendar
    try:
        events = await _upcoming_events()
        for event in events:
            signals.append(Signal(
                source=Source.CALENDAR,
                summary=event["summary"],
                urgency=Urgency.TIME_BOUND,
                occurs_at=event.get("starts_at"),
                key=event.get("id", ""),
            ))
    except Exception as exc:  # noqa: BLE001 — degrade, do not abort
        logger.warning("Briefing: calendar unavailable (%s)", exc)
        unavailable.append(Source.CALENDAR)

    # Email
    try:
        pending = await _pending_email()
        for item in pending:
            signals.append(Signal(
                source=Source.EMAIL,
                summary=item["summary"],
                urgency=Urgency.BLOCKING if item.get("awaiting_reply")
                else Urgency.NOTABLE,
                key=item.get("id", ""),
            ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Briefing: email unavailable (%s)", exc)
        unavailable.append(Source.EMAIL)

    # Feedback backlog. Unlike the two above this needs no credentials, so
    # the briefing has something real to say on a machine where Gmail and
    # Calendar were never connected — which is the machine it will be
    # demonstrated on. It is also the loop closing: corrections George marked
    # while talking to the twin are what the next fine-tune trains on, and
    # they are worth nothing sitting unreviewed in a log.
    try:
        pending_corrections = _unreviewed_corrections()
        if pending_corrections:
            signals.append(Signal(
                source=Source.CONVERSATION,
                summary=f"{pending_corrections} διορθώσεις περιμένουν έλεγχο",
                urgency=Urgency.NOTABLE,
                detail="από τη συνομιλία με το twin",
                key=f"corrections-{pending_corrections}",
            ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Briefing: feedback log unreadable (%s)", exc)
        unavailable.append(Source.CONVERSATION)

    result = build_briefing(signals, unavailable, set(req.already_said))

    return BriefingResponse(
        text=result.text,
        status=result.status,
        identities=[s.identity() for s in signals],
        unavailable=[s.value for s in unavailable],
    )


async def _upcoming_events() -> list[dict]:
    """Calendar events for the next 24 hours.

    Raises when the calendar cannot be reached, so the caller can say so.
    Returning an empty list on failure would be indistinguishable from a
    genuinely empty day — the distinction this whole endpoint turns on.
    """
    raise RuntimeError("Google Calendar credentials not configured")


async def _pending_email() -> list[dict]:
    """Messages that look like they are waiting on George.

    Same contract as :func:`_upcoming_events`: raise, never return empty on
    failure.
    """
    raise RuntimeError("Gmail credentials not configured")


def _unreviewed_corrections() -> int:
    """How many corrections are sitting in the feedback log.

    A missing log means the twin has never been rated, which is a real zero
    and not a failure — so this returns 0 rather than raising. An
    *unreadable* log is different and does raise, because "I could not check"
    and "there is nothing" must not collapse into the same answer.
    """
    if not FEEDBACK_LOG.exists():
        return 0

    count = 0
    with open(FEEDBACK_LOG, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # One malformed line should not hide the rest of the file.
                continue
            if entry.get("correction") or entry.get("rating", 0) < 0:
                count += 1
    return count


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

    # Technical confabulation about the thesis itself. Checked first because
    # it is the costliest kind: an examiner asking about the method gets a
    # fluent, specific, wrong answer, and fluency makes it harder to catch
    # than a vague one.
    from jarvis.inference.thesis_facts import check_technical_claims

    issues.extend(check_technical_claims(req.response))

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
