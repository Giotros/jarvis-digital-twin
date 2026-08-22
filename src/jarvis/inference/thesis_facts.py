"""Authoritative project facts for questions an examiner would ask.

A fine-tuned model does not know what you built. Asked "με τι τεχνολογίες
δούλεψες", it produces something that *sounds* like a plausible answer for a
student of this kind — and the live system produced "Krikri-12B", "QLoRA πάνω
στο BERTweet", "ανάλυση συναισθημάτων", MongoDB, Django, TensorFlow. None of
those exist in the project. The parameters were wrong, the base model was
wrong, and the task itself was wrong, delivered fluently and with confidence.

That failure mode is worse than an awkward tone. A committee can forgive a
twin that sounds too casual; it cannot ignore one that misstates the method
under examination. So technical answers are grounded in a file rather than
generated: :mod:`config/thesis_facts.yaml` is loaded and rendered into the
prompt whenever the academic register is active.

The trade-off is stated plainly: this makes the twin *recite* on technical
questions rather than improvise. For a viva that is the correct trade.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Searched in order. The first existing file wins, so a deployment can
#: override the checked-in defaults without editing the package.
_SEARCH_PATHS: tuple[Path, ...] = (
    Path("/app/config/thesis_facts.yaml"),
    Path(__file__).resolve().parents[3] / "config" / "thesis_facts.yaml",
    Path("config/thesis_facts.yaml"),
)

_cache: str | None = None


def _render(data: Any, indent: int = 0) -> list[str]:
    """Flatten the YAML into readable lines.

    Rendered as prose rather than dumped as YAML: the model reproduces the
    shape of what it is given, and a reply formatted as a YAML tree is not an
    answer to a spoken question.
    """
    pad = "  " * indent
    lines: list[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            label = str(key).replace("_", " ")
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{label}:")
                lines.extend(_render(value, indent + 1))
            else:
                lines.append(f"{pad}{label}: {str(value).strip()}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.extend(_render(item, indent))
            else:
                lines.append(f"{pad}- {str(item).strip()}")
    else:
        lines.append(f"{pad}{data}")

    return lines


def load_thesis_facts(force_reload: bool = False) -> str:
    """Return the project facts as a prompt block, or "" if unavailable.

    Missing or malformed files degrade to an empty string rather than
    raising. The grounding is important, but a broken YAML file should make
    the twin vaguer, not make it stop answering mid-presentation.
    """
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    for path in _SEARCH_PATHS:
        if not path.exists():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 - see docstring
            logger.error("Could not read thesis facts at %s: %s", path, exc)
            continue

        body = "\n".join(_render(data))
        _cache = (
            "ΣΤΟΙΧΕΙΑ ΤΗΣ ΔΙΠΛΩΜΑΤΙΚΗΣ ΣΟΥ — αυτά ΙΣΧΥΟΥΝ, είναι η δουλειά σου:\n"
            f"{body}\n\n"
            "Σε τεχνικές ερωτήσεις απάντα ΜΟΝΟ από τα παραπάνω. "
            "Μην αναφέρεις εργαλεία, μοντέλα ή αριθμούς που δεν γράφονται εδώ. "
            "Αν σε ρωτήσουν κάτι που δεν καλύπτεται, πες «δεν το έχω μετρήσει» "
            "ή «δεν το κάλυψα αυτό» — μην το συμπληρώσεις."
        )
        logger.info("Loaded thesis facts from %s", path)
        return _cache

    logger.warning(
        "No thesis_facts.yaml found; technical answers will be ungrounded"
    )
    _cache = ""
    return _cache


#: Claims observed from the live model that contradict the project.
#:
#: A curated list rather than "anything not in the facts file", because the
#: latter flags every ordinary word. These are the actual confabulations
#: recorded on 2026-08-22, when the twin was asked what technologies it used:
#: it named a base model it was not built on, a task it does not perform, and
#: a stack it does not run. Each entry carries the correction, so the check
#: reports what is wrong rather than only that something is.
# NOTE ON WORD BOUNDARIES
# ------------------------
# Every product name below ends in ``\w*``, never a bare ``\b``. The first
# version used ``\btensorflow\b`` and the model wrote "TensorFlow2", which
# does not match: a digit is a word character, so there is no boundary after
# "TensorFlow". The claim sailed through a check written specifically to
# catch it. Version suffixes, plural forms and glued-on words are the normal
# case in this text, not the exception.
_CONTRADICTIONS: tuple[tuple[str, str], ...] = (
    (r"krikri[\s-]*(?!8)\d+\s*b", "Το Krikri είναι 8B, όχι άλλο μέγεθος"),
    (r"\bbert\w*", "Δεν χρησιμοποιήθηκε BERT — η βάση είναι το Krikri-8B"),
    # Mistral and GPT are legitimate to *mention* — the thesis compares
    # against both and explains why neither was used. Only a claim of having
    # used them is wrong, so a usage verb has to appear nearby. Flagging the
    # bare name would mark the correct answer as a hallucination.
    (r"(?:χρησιμοπο\w+|δούλεψα|δουλεψα|έτρεξα|ετρεξα|βασίστηκα|βασιστηκα"
     r"|επέλεξα|επελεξα|διάλεξα|διαλεξα|εκπαίδευσα|εκπαιδευσα)"
     r"[^.!?;]{0,90}\bgpt\b",
     "Δεν χρησιμοποιήθηκε GPT — το μοντέλο τρέχει τοπικά"),
    (r"(?:χρησιμοπο\w+|δούλεψα|δουλεψα|έτρεξα|ετρεξα|βασίστηκα|βασιστηκα"
     r"|επέλεξα|επελεξα|διάλεξα|διαλεξα|εκπαίδευσα|εκπαιδευσα)"
     r"[^.!?;]{0,90}\bmistral\b",
     "Το Mistral εξετάστηκε αλλά απορρίφθηκε — δεν χρησιμοποιήθηκε"),
    (r"συναισθημ\w*\s+αναλ|ανάλυση\s+συναισθ", "Η εργασία δεν κάνει ανάλυση συναισθήματος"),
    (r"computer\s+vision|υπολογιστικ\w*\s+όραση", "Δεν υπάρχει computer vision στην εργασία"),
    (r"\bmongo\w*", "Η αποθήκευση είναι Delta Lake και ChromaDB, όχι MongoDB"),
    (r"\bdjango\w*", "Το API είναι FastAPI, όχι Django"),
    (r"\btensorflow\w*|\btensor\s?flow\w*", "Η εκπαίδευση έγινε με PyTorch και Ray, όχι TensorFlow"),
    (r"\bkeras\w*", "Δεν χρησιμοποιήθηκε Keras"),
    (r"\bflax\w*", "Δεν χρησιμοποιήθηκε Flax"),
    (r"\bkubernetes\w*|\bk8s\b", "Δεν χρησιμοποιήθηκε Kubernetes"),
    (r"\bredux\w*", "Δεν χρησιμοποιήθηκε Redux"),
    (r"\bopencv\w*|\bopen\s?cv\w*", "Δεν χρησιμοποιήθηκε OpenCV"),
    (r"\bselenium\w*", "Δεν χρησιμοποιήθηκε Selenium"),
    (r"\barduino\w*", "Δεν υπάρχει υλικό/Arduino στην εργασία"),
    # Invented tool names. These are not real products — the model produced
    # them by analogy from names that are, which is the same mechanism that
    # produced "Krikri-12B" and is invisible to a reader who does not already
    # know the ecosystem.
    (r"\brayhub\w*|\bray\s?hub\w*", "Δεν υπάρχει «RayHub» — το εργαλείο είναι το Ray"),
    (r"\bn8x\b", "Δεν υπάρχει «n8x» — το εργαλείο είναι το n8n"),
    # Third round of observations. Each was produced *after* the facts were
    # already in the prompt, which is the useful thing about them: grounding
    # reduces confabulation, it does not end it. The model paraphrases the
    # facts it was given and fills the gaps between them.
    (r"google\s?cloud\w*|\bgcp\b|\bazure\w*",
     "Η εκπαίδευση έγινε σε Colab και η εκτέλεση τοπικά — όχι σε GCP ή Azure"),
    (r"edge\s+(?:συσκευ\w*|devices?)|σε\s+edge\b",
     "Το QLoRA έγινε για να χωρέσει η εκπαίδευση σε ένα GPU, όχι για edge συσκευές"),
    (r"\bsolidity\w*|\bblockchain\w*", "Δεν υπάρχει blockchain στην εργασία"),
    (r"\bpostgres\w*|\bpostgresql\w*",
     "Η αποθήκευση είναι Delta Lake και ChromaDB, όχι PostgreSQL"),
    # Λάθος αναπτύγματα του ακρωνυμίου. Το base μοντέλο έγραψε
    # «RAG (Retriever-Adapter-Generator)» και, σε άλλη εκτέλεση,
    # «RAG (Retrovirus Activation Gene) — τμήμα του DNA». Και τα δύο
    # ακούγονται τεχνικά, και το δεύτερο είναι από άλλο επιστημονικό πεδίο.
    (r"\bRAG\b[^.!?;(]{0,25}\((?![^)]*[Rr]etrieval[- ][Aa]ugmented)[^)]{3,60}\)",
     "Το RAG είναι Retrieval-Augmented Generation — όχι κάτι άλλο"),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), msg) for pattern, msg in _CONTRADICTIONS
)


#: Negations that turn a flagged claim into a correct one.
#:
#: "Δεν χρησιμοποίησα GPT γιατί τα δεδομένα είναι προσωπικές συνομιλίες" is
#: the *right* answer, and a check that flags it would fire hardest exactly
#: when the twin is at its best. Applies to every pattern, not just the model
#: names: "δεν χρησιμοποιήσαμε MongoDB" is equally true.
_NEGATION = re.compile(r"\b(?:δεν?|όχι|οχι|χωρίς|χωρις)\b", re.IGNORECASE)

#: How far back to look for a negation. One clause, roughly — far enough to
#: catch "δεν χρησιμοποίησα X", short enough not to reach the previous
#: sentence, which is why clause punctuation also stops the search.
_NEGATION_WINDOW = 60


def _is_negated(text: str, start: int) -> bool:
    """True when the matched claim is preceded by a negation in the clause."""
    window = text[max(0, start - _NEGATION_WINDOW):start]
    # Do not look past the end of the previous clause.
    for boundary in (".", "!", "?", ";", "·"):
        window = window.rsplit(boundary, 1)[-1]
    return bool(_NEGATION.search(window))


def check_technical_claims(text: str) -> list[str]:
    """Report statements that contradict the project.

    Detection only — the caller decides whether to warn, log, or regenerate.
    Silently rewriting a technical answer would hide the very failure the
    evaluation chapter needs to be able to count.
    """
    if not text:
        return []
    issues: list[str] = []
    for pattern, msg in _COMPILED:
        match = pattern.search(text)
        if match and not _is_negated(text, match.start()):
            issues.append(msg)
    return issues
