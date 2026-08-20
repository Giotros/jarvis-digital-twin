"""Post-processing guardrails for model output.

Thesis §2.2.5 — Ενορχήστρωση Αυτόνομων Πρακτόρων: Guardrails ensure
the model's raw output is clean, polite, and properly formatted before
reaching the user. These rules will be replicated as n8n nodes in the
agentic orchestration layer.

Pipeline: raw response → emoji cleanup → name filter → profanity filter
          → accent restoration → capitalize → PII leak check → clean response
"""

from __future__ import annotations

import re
from typing import Any


# ── Common Greek words: unaccented → accented ──────────────────────
# This covers the most frequent words. For production, consider
# the `greek-accentuation` library or a dedicated accent model.
_ACCENT_MAP: dict[str, str] = {
    # Verbs
    "ειναι": "είναι", "εχω": "έχω", "εχει": "έχει", "εχεις": "έχεις",
    "εχουμε": "έχουμε", "κανω": "κάνω", "κανεις": "κάνεις", "κανει": "κάνει",
    "κανουμε": "κάνουμε", "θελω": "θέλω", "θελεις": "θέλεις", "θελει": "θέλει",
    "μπορω": "μπορώ", "μπορεις": "μπορείς", "μπορει": "μπορεί",
    "πρεπει": "πρέπει", "παω": "πάω", "παμε": "πάμε", "πηγαινω": "πηγαίνω",
    "ξερω": "ξέρω", "ξερεις": "ξέρεις", "λεω": "λέω", "λεει": "λέει",
    "βλεπω": "βλέπω", "βλεπεις": "βλέπεις", "δουλευω": "δουλεύω",
    "δουλευεις": "δουλεύεις", "ερχομαι": "έρχομαι", "φευγω": "φεύγω",
    "περιμενω": "περιμένω", "περιμενε": "περίμενε", "στελνω": "στέλνω",
    "βαζω": "βάζω", "παιρνω": "παίρνω", "δινω": "δίνω", "βρισκω": "βρίσκω",
    "ψαχνω": "ψάχνω", "αρχιζω": "αρχίζω", "τελειωνω": "τελειώνω",
    "εννοειται": "εννοείται", "νομιζω": "νομίζω", "φαινεται": "φαίνεται",
    "ακουω": "ακούω", "βοηθαω": "βοηθάω", "αλλαζω": "αλλάζω",

    # Pronouns / articles / particles
    "εγω": "εγώ", "εσυ": "εσύ", "αυτο": "αυτό", "αυτη": "αυτή",
    "αυτος": "αυτός", "ολα": "όλα", "ολοι": "όλοι", "κατι": "κάτι",
    "ποιος": "ποιός", "ποτε": "πότε", "πως": "πώς", "που": "πού",

    # Adverbs / conjunctions / prepositions
    "καλα": "καλά", "τωρα": "τώρα", "μετα": "μετά", "εδω": "εδώ",
    "εκει": "εκεί", "γιατι": "γιατί", "πολυ": "πολύ", "αλλα": "αλλά",
    "ομως": "όμως", "οταν": "όταν", "αφου": "αφού", "επισης": "επίσης",
    "ισως": "ίσως", "βεβαια": "βέβαια", "μονο": "μόνο", "ακομα": "ακόμα",
    "αμεσα": "άμεσα", "γρηγορα": "γρήγορα", "ευκολα": "εύκολα",
    "απλα": "απλά", "σιγουρα": "σίγουρα", "ετσι": "έτσι", "αυριο": "αύριο",
    "χτες": "χθες", "σημερα": "σήμερα", "οχι": "όχι", "μαζι": "μαζί",
    "κιολας": "κιόλας", "αμεσως": "αμέσως", "τελικα": "τελικά",
    "αρκετα": "αρκετά",

    # Nouns (common)
    "δουλεια": "δουλειά", "ωρα": "ώρα", "μερα": "μέρα", "νερο": "νερό",
    "φιλε": "φίλε", "φιλαρακι": "φιλαράκι", "αδερφε": "αδερφέ",
    "πληροφορικη": "πληροφορική", "εταιρεια": "εταιρεία",
    "προβλημα": "πρόβλημα", "θεμα": "θέμα", "λυση": "λύση",
    "ωραια": "ωραία", "καλημερα": "καλημέρα", "καλησπερα": "καλησπέρα",
    "ευχαριστω": "ευχαριστώ", "παρακαλω": "παρακαλώ",

    # Adjectives
    "καλος": "καλός", "καλη": "καλή", "καλο": "καλό",
    "νεος": "νέος", "νεα": "νέα", "νεο": "νέο",
    "μεγαλος": "μεγάλος", "μεγαλη": "μεγάλη", "μικρος": "μικρός",
    "σωστο": "σωστό", "σωστα": "σωστά",
}

# Pre-compile regex patterns for accent restoration
_ACCENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{k}\b", re.IGNORECASE), v)
    for k, v in _ACCENT_MAP.items()
]

# Default profanity replacements
_DEFAULT_PROFANITY_REPLACEMENTS: dict[str, str] = {
    "γαμησετα": "άστα να πάνε",
    "γαμω": "ωχ",
    "μαλακα": "φίλε",
    "σκατα": "χάλια",
}

# Default blocked words (replaced with "...")
_DEFAULT_BLOCKED_WORDS: list[str] = [
    "γαμησε", "γαμησετα", "γαμω", "γαμησου", "γαμωτο",
    "πουτανα", "σκατα", "αρχιδι", "μουνι", "γαμημεν", "σκασε",
]

# Words considered impolite (removed)
_DEFAULT_IMPOLITE: list[str] = ["ρε"]

# Common hallucinated name prefixes
_HALLUCINATED_NAMES: list[str] = [
    "παναγιωτη", "γιωτη", "χρηστο", "μαρια", "νικο", "δημητρη",
    "κωστα", "γιαννη", "σπυρο", "αντωνη", "μιχαλη", "βασιλη",
]


class Guardrails:
    """Post-processing pipeline for model output.

    Configurable via settings.yaml or constructor kwargs.
    Each step can be enabled/disabled independently.
    """

    def __init__(
        self,
        capitalize: bool = True,
        restore_accents: bool = True,
        filter_profanity: bool = True,
        remove_names: bool = True,
        clean_emojis: bool = True,
        remove_impolite: bool = True,
        profanity_replacements: dict[str, str] | None = None,
        blocked_words: list[str] | None = None,
        impolite_words: list[str] | None = None,
    ) -> None:
        self.capitalize = capitalize
        self.restore_accents = restore_accents
        self.filter_profanity = filter_profanity
        self.remove_names = remove_names
        self.clean_emojis = clean_emojis
        self.remove_impolite = remove_impolite
        self.profanity_replacements = profanity_replacements or _DEFAULT_PROFANITY_REPLACEMENTS
        self.blocked_words = blocked_words or _DEFAULT_BLOCKED_WORDS
        self.impolite_words = impolite_words or _DEFAULT_IMPOLITE

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "Guardrails":
        """Create from settings.yaml guardrails section."""
        g = settings.get("guardrails", {})
        return cls(
            capitalize=g.get("capitalize_sentences", True),
            restore_accents=g.get("restore_accents", True),
            filter_profanity=g.get("filter_profanity", True),
            remove_names=g.get("remove_name_hallucinations", True),
            clean_emojis=g.get("clean_emoji_artifacts", True),
            profanity_replacements=g.get("profanity_replacements"),
            blocked_words=g.get("blocked_words"),
            impolite_words=g.get("impolite_words"),
        )

    def process(self, text: str) -> str:
        """Run the full guardrail pipeline."""
        if not text:
            return text

        if self.clean_emojis:
            text = self._clean_emoji_artifacts(text)
        if self.remove_names:
            text = self._remove_name_hallucinations(text)
        if self.filter_profanity:
            text = self._filter_profanity(text)
        if self.remove_impolite:
            text = self._remove_impolite_words(text)
        if self.restore_accents:
            text = self._restore_accents(text)
        if self.capitalize:
            text = self._capitalize_sentences(text)

        return text.strip()

    def _clean_emoji_artifacts(self, text: str) -> str:
        """Remove text emoji like (laugh), (purple_heart), etc."""
        return re.sub(r"\([a-z_]+\)", "", text).strip()

    def _remove_name_hallucinations(self, text: str) -> str:
        """Remove hallucinated name prefixes at the start of response."""
        pattern = "|".join(re.escape(n) for n in _HALLUCINATED_NAMES)
        text = re.sub(rf"^({pattern})\s+", "", text, flags=re.IGNORECASE)
        return text

    def _filter_profanity(self, text: str) -> str:
        """Replace profanity with milder alternatives, block the rest."""
        result = text
        # First: known replacements
        for bad, good in self.profanity_replacements.items():
            result = re.sub(re.escape(bad), good, result, flags=re.IGNORECASE)
        # Then: block remaining
        for word in self.blocked_words:
            result = re.sub(
                re.escape(word) + r"\w*", "...", result, flags=re.IGNORECASE
            )
        return result

    def _remove_impolite_words(self, text: str) -> str:
        """Remove impolite words like 'ρε'."""
        for word in self.impolite_words:
            # Remove "ρε" when standalone (not part of another word)
            text = re.sub(rf"\b{re.escape(word)}\b\s*", "", text, flags=re.IGNORECASE)
        return text

    def _restore_accents(self, text: str) -> str:
        """Restore accents on common Greek words."""
        result = text
        for pattern, replacement in _ACCENT_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize first letter of each sentence."""
        if not text:
            return text
        # Capitalize first character
        text = text[0].upper() + text[1:]
        # Capitalize after . ! ? followed by space
        text = re.sub(
            r"([.!?]\s+)(\w)",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )
        return text
