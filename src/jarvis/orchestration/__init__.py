"""Phase 5 — Orchestration: intent classification + n8n workflow support.

Supports 6 intent categories:
  PERSONAL, KNOWLEDGE, CASUAL, SENSITIVE, MEMORY (email), SCHEDULE (calendar)
"""

from jarvis.orchestration.intent_classifier import classify_intent, Intent

__all__ = ["classify_intent", "Intent"]
