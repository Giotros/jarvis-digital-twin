"""Static identity for the digital twin — who is George Trochidis.

This module loads config/identity.yaml and converts it to a natural-language
prompt block that gets injected into EVERY system prompt. It is the twin's
"permanent memory" — facts that never change and are always available,
regardless of what RAG retrieves.

The identity is kept in YAML (not hardcoded) so George can update it
without touching code, and so the thesis can reference it as a
configurable component of the architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_identity(path: str | Path | None = None) -> dict[str, Any]:
    """Load the identity YAML file."""
    if path is None:
        path = Path(__file__).resolve().parents[3] / "config" / "identity.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def identity_to_prompt(identity: dict[str, Any] | None = None) -> str:
    """Convert identity data to a natural-language prompt block.

    Returns a string like:
        Με λένε Γιώργο Τροχίδη. Είμαι 26 χρονών, από τα Γιαννιτσά.
        Σπούδασα πληροφορική στο Ιόνιο Πανεπιστήμιο, τώρα κάνω μεταπτυχιακό...
    """
    if identity is None:
        identity = load_identity()

    p = identity.get("personal", {})
    e = identity.get("education", {})
    c = identity.get("career", {})
    m = identity.get("military", {})
    h = identity.get("hobbies", [])

    current_job = c.get("current", {})

    lines = [
        f"Με λένε {p.get('nickname', 'Γιώργο')} {p.get('full_name', 'Τροχίδη').split()[-1]}.",
        f"Είμαι {p.get('age', '26')} χρονών, γεννημένος στα {p.get('birthplace', 'Γιαννιτσά')}.",
        f"Σπούδασα πληροφορική στην Κέρκυρα ({e.get('bachelor', '')}).",
        f"Τώρα κάνω μεταπτυχιακό: {e.get('current', '')}.",
        f"Η διπλωματική μου είναι το Jarvis George — AI digital twin.",
    ]

    if current_job:
        lines.append(
            f"Δουλεύω ως {current_job.get('title', '')} "
            f"στην {current_job.get('company', '')}."
        )

    previous = c.get("previous", [])
    if previous:
        prev_summary = ", ".join(
            f"{j.get('title', '')} στην {j.get('company', '')}"
            for j in previous[:2]  # top 2 most recent
        )
        lines.append(f"Πριν δούλεψα: {prev_summary}.")

    if m.get("served"):
        lines.append(f"Έκανα στρατό στην {m.get('location', 'Κύπρο')}.")

    if h:
        lines.append(f"Hobbies: {', '.join(h)}.")

    lines.append(
        f"Μιλάω {' και '.join(p.get('languages', ['Ελληνικά', 'Αγγλικά']))}."
    )

    return " ".join(lines)


# Pre-built prompt for convenience
IDENTITY_PROMPT = identity_to_prompt()
