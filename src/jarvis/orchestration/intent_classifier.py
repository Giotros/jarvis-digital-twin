"""Intent classification for n8n routing.

Classifies incoming messages into categories so the n8n orchestrator
can route them to the appropriate pipeline:

  - PERSONAL  → answered from static identity (config/identity.yaml)
  - KNOWLEDGE → answered via RAG retrieval from conversation corpus
  - CASUAL    → direct generation (greeting, small talk)
  - SENSITIVE → flagged for human review (money, legal, health)
  - MEMORY    → answered via email search (recent events, orders, receipts)
  - SCHEDULE  → answered via calendar lookup (availability, meetings)

The classifier uses a lightweight keyword/pattern approach — no LLM call
needed for routing. This keeps latency low and makes the orchestration
layer independent of model availability.

Architecture note: this runs as a Code Node inside n8n, or as a
standalone FastAPI endpoint that n8n calls via HTTP Request node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class Intent(str, Enum):
    """Message intent categories for routing.

    9 categories — each maps to a different n8n branch:
      Core:     PERSONAL, KNOWLEDGE, CASUAL, SENSITIVE
      Live:     MEMORY (email), SCHEDULE (calendar)
      External: DEVOPS (GitHub), WEATHER, NEWS
    """
    PERSONAL = "personal"
    KNOWLEDGE = "knowledge"
    CASUAL = "casual"
    SENSITIVE = "sensitive"
    MEMORY = "memory"
    SCHEDULE = "schedule"
    DEVOPS = "devops"
    WEATHER = "weather"
    NEWS = "news"


@dataclass
class ClassificationResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float  # 0.0–1.0
    matched_keywords: list[str] = field(default_factory=list)
    explanation: str = ""


# ── Keyword dictionaries ────────────────────────────────────────

# Personal questions — answered from identity.yaml
_PERSONAL_PATTERNS: list[tuple[str, float]] = [
    # Direct personal questions
    (r"\b(ποιος|ποια)\s+(εισαι|είσαι)\b", 0.95),
    (r"\bπως\s+(σε\s+)?λενε\b", 0.95),
    (r"\bπώς\s+(σε\s+)?λένε\b", 0.95),
    (r"\b(ποσο|πόσο)\s+(χρονων|χρονών)\b", 0.95),
    (r"\bηλικια\b", 0.90),
    (r"\bηλικία\b", 0.90),
    (r"\b(απο\s+που|από\s+πού)\s+(εισαι|είσαι)\b", 0.95),
    (r"\bγεννη(θηκες|θήκες)\b", 0.90),
    (r"\b(τι|τί)\s+(δουλεια|δουλειά)\s+(κανεις|κάνεις)\b", 0.95),
    (r"\bπου\s+(δουλευεις|δουλεύεις)\b", 0.95),
    (r"\b(τι|τί)\s+(σπουδ(ασες|ασεις|αζεις)|σπούδ(ασες|ασεις|άζεις))\b", 0.90),
    (r"\bπτυχιο\b", 0.85),
    (r"\bπτυχίο\b", 0.85),
    (r"\bμεταπτυχιακ[οό]\b", 0.90),
    (r"\bδιπλωματικ[ηή]\b", 0.90),
    (r"\bστρατ[οό]\b", 0.80),
    (r"\bhobb(y|ies)\b", 0.85),
    (r"\bχομπ[ιυ]\b", 0.85),
    (r"\bβιογραφικ[οό]\b", 0.85),
    (r"\b(cv|resume)\b", 0.85),
    (r"\bπες\s+μου\s+για\s+(σενα|εσένα|τον εαυτο σου)\b", 0.90),
    (r"\bπες\s+μου\s+για\s+(σένα|εσένα|τον εαυτό σου)\b", 0.90),
]

# Knowledge/technical — answered via RAG
_KNOWLEDGE_PATTERNS: list[tuple[str, float]] = [
    (r"\b(πως|πώς)\s+(να|θα)\b", 0.70),
    (r"\b(εξηγησε|εξήγησε)\b", 0.75),
    (r"\b(τι\s+ειναι|τί\s+είναι)\s+\w+", 0.70),
    (r"\bβοηθ(εια|ησε|ήσε)\b", 0.65),
    (r"\bπροβλημα\b", 0.70),
    (r"\bπρόβλημα\b", 0.70),
    (r"\b(error|bug|crash|fix)\b", 0.75),
    (r"\bδεν\s+(δουλευει|δουλεύει|λειτουργ)\b", 0.75),
    (r"\b(internet|ιντερνετ|ίντερνετ|wifi|router)\b", 0.70),
    (r"\b(κωδικ[οό]|password)\b", 0.65),
    (r"\bτεχνικ[οόηή]\b", 0.65),
]

# Casual — direct generation, no retrieval needed
_CASUAL_PATTERNS: list[tuple[str, float]] = [
    (r"^(γεια|γειά|γεια\s+σου|γειά\s+σου|hey|hi|hello)\b", 0.90),
    (r"^(καλημερα|καλημέρα|καλησπερα|καλησπέρα|καληνυχτα|καληνύχτα)\b", 0.90),
    (r"\b(τι\s+κανεις|τί\s+κάνεις|τι\s+κάνεις|τι\s+λεει|τί\s+λέει)\b", 0.85),
    (r"\b(τι\s+γινεται|τί\s+γίνεται|τι\s+γίνεται)\b", 0.85),
    (r"\b(παμε|πάμε)\s+(για|να)\s+(καφε|καφέ|μπυρα|μπύρα|φαγητο|φαγητό)\b", 0.85),
    (r"\b(ευχαριστω|ευχαριστώ|thanks|thank you)\b", 0.80),
    (r"\b(οκ|οκει|ωραια|ωραία|εντάξει|ενταξει)\b", 0.75),
    (r"^(ναι|όχι|οχι)[\s!.]*$", 0.80),
    (r"\b(χαχα|lol|haha|xaxa)\b", 0.85),
]

# Sensitive — requires human review
_SENSITIVE_PATTERNS: list[tuple[str, float]] = [
    (r"\b(λεφτα|λεφτά|χρηματα|χρήματα|πληρωμ[ηή]|μεταφορ[αά])\b", 0.80),
    (r"\b(τραπεζ[αά]|iban|λογαριασμ[οό])\b", 0.85),
    (r"\b(δικηγορ[οό]|νομικ[αάοό]|αγωγ[ηή])\b", 0.80),
    (r"\b(γιατρ[οό]|νοσοκομει[οό]|αρρωστ|φαρμακ)\b", 0.75),
    (r"\b(αφμ|αμκα|ταυτοτητ[αά]ς?|ταυτότητ[αά]ς?)\b", 0.90),
    (r"\b(κωδικ[οό]ς?\s+(τραπεζ|πιστωτικ))\b", 0.95),
    (r"\bpassword\b", 0.70),
]

# Memory — answered via email search (recent events, orders, tracking)
_MEMORY_PATTERNS: list[tuple[str, float]] = [
    # Orders and purchases
    (r"\b(παραγγελ[ιία]|παράγγειλ|order)\b", 0.90),
    (r"\b(ρουχα|ρούχα|παπουτσ|παπούτσ)\b", 0.75),
    (r"\b(tracking|αποστολ[ηή]|courier|acs|speedex|ελτα|elta)\b", 0.85),
    (r"\b(αγορ[αά]|αγόρασ)\b", 0.80),
    (r"\b(τιμολογ[ιί]ο?|invoice|αποδειξ[ηή]|receipt)\b", 0.85),
    # Recent events / "what happened with X"
    (r"\bτι\s+(εγινε|έγινε)\s+(με|για)\b", 0.85),
    (r"\bτι\s+(γινεται|γίνεται)\s+(με|για)\b", 0.80),
    (r"\b(θυμασαι|θυμάσαι)\s+(αν|τι|ποτε|πότε)\b", 0.80),
    (r"\b(ηρθε|ήρθε|εφτασε|έφτασε)\s+(το|η|τα)\b", 0.80),
    (r"\b(email|mail|μειλ|μηνυμα|μήνυμα)\s+(απο|από)\b", 0.85),
    (r"\b(απαντησ[αε]|απάντησ[αε])\s+(στο|στον|στην)\b", 0.75),
    # Subscriptions / services
    (r"\b(συνδρομ[ηή]|subscription)\b", 0.80),
    (r"\b(λογαριασμ[οό]ς?\s+(ρεύμα|νερ[οό]|κινητ))\b", 0.80),
    (r"\b(πληρω[σσ]|πλήρωσ)\b", 0.70),
]

# Schedule — answered via calendar lookup (availability, meetings)
_SCHEDULE_PATTERNS: list[tuple[str, float]] = [
    # Availability questions
    (r"\b(διαθεσιμ[οό]ς?|διαθέσιμ[οό]ς?)\b", 0.90),
    (r"\b(ελευθερ[οό]ς?|ελεύθερ[οό]ς?)\b", 0.85),
    (r"\bεχεις\s+(χρονο|χρόνο|ωρα|ώρα)\b", 0.85),
    (r"\bέχεις\s+(χρόνο|ώρα)\b", 0.85),
    (r"\b(ποτε|πότε)\s+(μπορ[εεί]ς?|εισαι|είσαι|θα|εχεις|έχεις)\b", 0.90),
    (r"\b(ποτε|πότε)\s+(μπορεις|μπορείς)\b", 0.90),
    # Meeting / appointment references
    (r"\b(ραντεβου|ραντεβού|meeting|συναντηση|συνάντηση)\b", 0.85),
    (r"\b(calendar|ημερολογι[οό]|ημερολόγι[οό])\b", 0.90),
    (r"\b(προγραμμα|πρόγραμμα)\s+(σου|εχεις|έχεις|σημερα|σήμερα|αυριο|αύριο)\b", 0.85),
    # Time-specific availability
    (r"\b(αυριο|αύριο|μεθαυριο|μεθαύριο)\s+.*(βολευει|βολεύει|κανει|κάνει)\b", 0.85),
    (r"\b(σημερα|σήμερα)\s+(το\s+)?(απογευμα|απόγευμα|βραδυ|βράδυ|μεσημερ)\b", 0.80),
    (r"\b(δευτερα|τριτη|τεταρτη|πεμπτη|παρασκευη|σαββατο|κυριακη)\b", 0.75),
    (r"\b(δευτέρα|τρίτη|τετάρτη|πέμπτη|παρασκευή|σάββατο|κυριακή)\b", 0.75),
    # Invitations
    (r"\b(παμε|πάμε)\s+(να|για)\b.*\b(αυριο|αύριο|σημερα|σήμερα|βραδυ|βράδυ)\b", 0.85),
    (r"\bθες\s+να\s+(βρεθουμε|βρεθούμε|συναντηθ)\b", 0.85),
    (r"\b(ποτε|πότε)\s+.*\b(βρεθουμε|βρεθούμε)\b", 0.85),
]

# DevOps — answered via GitHub API (commits, issues, PRs)
_DEVOPS_PATTERNS: list[tuple[str, float]] = [
    (r"\b(commit|commits|pushed|push)\b", 0.90),
    (r"\b(pull\s*request|pr|merge)\b", 0.85),
    (r"\b(issue|issues|bug|bugs)\b", 0.80),
    (r"\b(github|gitlab|repo|repository)\b", 0.90),
    (r"\b(deploy|deployment|release)\b", 0.80),
    (r"\b(branch|branches|main|master)\b", 0.75),
    (r"\b(κωδικ[αά]|κώδικ[αά])\s+(σημερα|σήμερα|χτες|χθες)\b", 0.85),
    (r"\b(τι|τί)\s+(εκανες|έκανες)\s+.*(code|coding|κωδικ)\b", 0.80),
    (r"\b(ανοιχτ[αάοό])\s+(issue|pr|ticket)\b", 0.85),
    (r"\b(pipeline|ci|cd|build)\b", 0.75),
]

# Weather — answered via weather API (OpenWeatherMap)
_WEATHER_PATTERNS: list[tuple[str, float]] = [
    (r"\b(καιρ[οό]ς?|καιρός?)\b", 0.95),
    (r"\b(weather)\b", 0.90),
    (r"\b(βρ[εέ]ξ[εη]ι?|βροχ[ηή]|χιον[ιί])\b", 0.90),
    (r"\b(ζεστ[αάηή]|κρυ[οό]|κρύο)\b", 0.75),
    (r"\b(θερμοκρασ[ιί]α|βαθμ[οό]υ?ς?)\b", 0.85),
    (r"\b(ηλι[οό]ς?|ήλι[οό]ς?|συννεφ)\b", 0.85),
    (r"\b(ομπρελα|ομπρέλα)\b", 0.90),
    (r"\b(τι\s+καιρο|τί\s+καιρό)\b", 0.95),
]

# News — answered via RSS feeds or news API
_NEWS_PATTERNS: list[tuple[str, float]] = [
    (r"\b(νεα|νέα|ειδησ[εη]ις?|ειδήσ[εη]ις?)\b", 0.90),
    (r"\b(news)\b", 0.85),
    (r"\bτι\s+(γινεται|γίνεται)\s+(στ[ηο]ν?|στην?)\s+(ελλαδα|ελλάδα|κοσμο|κόσμο)\b", 0.90),
    (r"\b(τι\s+γινεται|τί\s+γίνεται)\s+(σημερα|σήμερα)\b", 0.80),
    (r"\b(πολιτικ[αάηή]|οικονομι[αά]|αθλητικ[αά])\b", 0.75),
    (r"\b(tech\s*news|τεχνολογ[ιί]α\s+νεα)\b", 0.85),
    (r"\b(τι\s+εγινε|τί\s+έγινε)\s+(στ[ηο]ν?|με\s+τ)\b", 0.75),
]

_ALL_CATEGORIES: list[tuple[Intent, list[tuple[str, float]]]] = [
    (Intent.SENSITIVE, _SENSITIVE_PATTERNS),    # check first (safety)
    (Intent.SCHEDULE, _SCHEDULE_PATTERNS),      # before casual (catches "πάμε για καφέ αύριο")
    (Intent.MEMORY, _MEMORY_PATTERNS),          # before knowledge (catches "τι έγινε με X")
    (Intent.DEVOPS, _DEVOPS_PATTERNS),          # before knowledge (catches "commits")
    (Intent.WEATHER, _WEATHER_PATTERNS),        # specific topic
    (Intent.NEWS, _NEWS_PATTERNS),              # specific topic
    (Intent.PERSONAL, _PERSONAL_PATTERNS),
    (Intent.KNOWLEDGE, _KNOWLEDGE_PATTERNS),
    (Intent.CASUAL, _CASUAL_PATTERNS),
]


def classify_intent(message: str) -> ClassificationResult:
    """Classify a user message into an intent category.

    The classifier checks patterns in priority order:
    SENSITIVE > SCHEDULE > MEMORY > PERSONAL > KNOWLEDGE > CASUAL.

    If no patterns match, defaults to KNOWLEDGE (safest fallback:
    uses RAG, which either finds relevant context or returns nothing).

    Parameters
    ----------
    message : str
        The raw user message (Greek or English).

    Returns
    -------
    ClassificationResult
        The classified intent with confidence and matched keywords.
    """
    text = message.lower().strip()

    if not text:
        return ClassificationResult(
            intent=Intent.CASUAL,
            confidence=1.0,
            explanation="Empty message → casual",
        )

    best_intent = Intent.KNOWLEDGE  # safe default
    best_confidence = 0.0
    best_keywords: list[str] = []

    for intent, patterns in _ALL_CATEGORIES:
        intent_score = 0.0
        intent_keywords: list[str] = []

        for pattern, weight in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                intent_keywords.append(match.group(0))
                # Take highest matching weight for this intent
                if weight > intent_score:
                    intent_score = weight

        # SENSITIVE gets a boost if ANY pattern matches (safety-first)
        if intent == Intent.SENSITIVE and intent_score > 0:
            intent_score = min(intent_score + 0.1, 1.0)

        if intent_score > best_confidence:
            best_confidence = intent_score
            best_intent = intent
            best_keywords = intent_keywords

    # If nothing matched with decent confidence, default to KNOWLEDGE
    if best_confidence < 0.5:
        return ClassificationResult(
            intent=Intent.KNOWLEDGE,
            confidence=0.5,
            matched_keywords=[],
            explanation="No strong pattern match → defaulting to RAG retrieval",
        )

    return ClassificationResult(
        intent=best_intent,
        confidence=best_confidence,
        matched_keywords=best_keywords,
        explanation=f"Matched {best_intent.value} patterns: {', '.join(best_keywords[:3])}",
    )
