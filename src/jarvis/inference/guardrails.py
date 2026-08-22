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
import unicodedata
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
#
# Stored as STEMS, matched accent- and case-insensitively against a
# normalised copy of the text. An exact word list fails twice over in
# Greek: "πούστη" (accented) misses "πουστη", and "πουστης / πουστες /
# πουστη" are three separate surface forms of one stem. The corpus is
# casual chat between friends, so the model produces these fluently —
# observed in evaluation as "Ναι εννοείται τρελε πουστη μ", which passed
# the original filter untouched.
_DEFAULT_BLOCKED_WORDS: list[str] = [
    # sexual / obscene
    "γαμησ", "γαμω", "γαμωτ", "γαμημ", "πουταν", "μουν", "αρχιδ",
    "πουστ", "καριολ", "μαλακισμεν", "μπινε",
    # scatological
    "σκατ", "χεσ", "κωλοπαιδ",
    # insults likely in friendly banter but wrong in a demo
    "βλαμμεν", "ηλιθι", "κωλο",
]

# Words considered impolite (removed)
_DEFAULT_IMPOLITE: list[str] = ["ρε"]

# Familiar vocatives, stripped in the professional and academic registers.
#
# Prompting alone does not remove these. The adapter was trained on 13k
# casual messages where they appear constantly, and asked to address a
# professor the model still produced "Ειμαι καλά αγορι μ να ξερς". A
# demonstration in the prompt improves the odds; it does not make them zero,
# and one "αγόρι μου" to an examiner is one too many. Deterministic removal
# is the only version of this that can be relied on during a viva.
#
# Matched as *vocative forms*, not stems. Stems are too blunt in Greek:
# "μεγαλ" would also delete "μεγάλο πρόβλημα", and "τρελ" would delete
# "τρελό". The masculine vocative ending -ε is distinctive enough to match
# safely, and the neuter familiar forms are only ever addresses when they
# carry a possessive ("αγόρι μου"), so the possessive is required.
_FAMILIAR_VOCATIVE_PATTERNS: list[str] = [
    r"φιλαρακι(?:\s+μ(?:ου)?)?",
    r"φιλε",
    r"φιλαρα",
    r"αδερφε",
    r"αδελφε",
    r"μεγαλε",
    r"αρχηγε",
    r"μαστορα",
    r"τρελε",
    r"αγορι\s+μ(?:ου)?",
    r"κουκλα\s+μ(?:ου)?",
    r"μανα\s+μ(?:ου)?",
    r"ψυχη\s+μ(?:ου)?",
]

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

    #: Registers where familiar address is wrong. Kept as a set rather than a
    #: boolean so a future register can opt in without changing callers.
    FORMAL_REGISTERS = frozenset({"professional", "academic"})

    _VOCATIVE_RE = re.compile(
        r",?\s*\b(?:" + "|".join(_FAMILIAR_VOCATIVE_PATTERNS) + r")\b",
        re.IGNORECASE,
    )

    def process(self, text: str, register: str = "") -> str:
        """Run the full guardrail pipeline.

        ``register`` is the relationship register the reply was generated
        for. In the formal registers, familiar vocatives are removed after
        generation, because the model does not reliably drop them when asked.
        """
        if not text:
            return text

        # Always first: anonymisation placeholders must never reach a reader.
        text = self._strip_anonymisation_placeholders(text)

        if register in self.FORMAL_REGISTERS:
            text = self._strip_familiar_vocatives(text)

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

    #: Placeholders the sanitiser writes into the training corpus. The model
    #: sees ~4,700 of them during fine-tuning and learns to emit them as if
    #: they were words — observed in production as the reply "Ναι [NAME]".
    #: They are an artefact of the privacy pipeline leaking into the output
    #: surface, so they are stripped unconditionally, before any other rule.
    _PLACEHOLDER_RE = re.compile(
        r"\s*\[(?:NAME|PERSON(?:_\d+)?|PHONE|EMAIL|IBAN|AFM|AMKA|URL|ID_CARD)\]\s*",
        re.IGNORECASE,
    )

    def _strip_anonymisation_placeholders(self, text: str) -> str:
        """Remove privacy placeholders and tidy the seam they leave behind.

        Deleting a token mid-sentence can strand punctuation ("Ναι ,") or
        double a space, so the gap is closed rather than merely blanked.
        """
        cleaned = self._PLACEHOLDER_RE.sub(" ", text)
        cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def sanitise_output(self, text: str, register: str = "") -> str:
        """The subset of the pipeline that must run at generation time.

        Two rules cannot wait for the downstream guardrails node. Register
        enforcement needs to know who is being addressed, which only the
        generation step knows. Placeholder stripping is here because it must
        never be skipped, and it *was* being skipped: callers that hit
        /generate directly bypassed :meth:`process` entirely, and "[NAME]"
        reappeared in output that had supposedly been cleaned. A rule whose
        whole point is that it always applies should not live only on one
        code path.

        Both are idempotent, so the later node stays correct.
        """
        if not text:
            return text
        text = self._strip_anonymisation_placeholders(text)
        text = self._strip_surnames(text)
        if register in self.FORMAL_REGISTERS:
            text = self._strip_familiar_vocatives(text)
        return text

    def _strip_surnames(self, text: str) -> str:
        """Last line of defence against a third party's surname in the output.

        The corpus was believed clean and was not: two real surnames survived
        34 occurrences of sanitisation, were trained into the adapter, and
        came out of the deployed model when it was asked about a meeting.
        Retraining removes them from the weights, but retraining takes a day
        and the presentation does not wait for it.

        Cheap enough to run on every reply, and it costs nothing when the
        gazetteer is empty.
        """
        from jarvis.sanitization.greek_surnames import redact_surnames

        cleaned, count = redact_surnames(text, placeholder="")
        if not count:
            return text
        return self._close_gap(cleaned)

    #: Articles and prepositions that become ungrammatical once the noun they
    #: govern is deleted.
    #:
    #: Removing the name alone produced "και ο είναι πολύ καλός καθηγητής"
    #: and "το δουλεύω με τον από τη σχολή" in live output — sentences a
    #: reader stops at. A privacy filter that leaves visibly broken Greek
    #: draws attention to exactly the sentence it was trying to make
    #: unremarkable.
    _ORPHAN_ARTICLE = re.compile(
        r"(?:\b(?:στον|στην|στο|στου|στης|από\s+τον|απο\s+τον|με\s+τον|"
        r"με\s+την|για\s+τον|για\s+την)|\b(?:ο|η|το|τον|την|του|της|τους|τις))"
        r"\s+(?=[,.;!?·]|$|\s)",
        re.IGNORECASE,
    )

    def _close_gap(self, text: str) -> str:
        """Tidy the hole a deleted word leaves behind.

        Runs repeatedly because deletions cascade: "με τον Παπαδόπουλο" first
        loses the name, then the article, and the preposition only becomes
        orphaned once the article is gone.
        """
        previous = None
        cleaned = text
        while cleaned != previous:
            previous = cleaned
            cleaned = self._ORPHAN_ARTICLE.sub("", cleaned)
            cleaned = re.sub(r"\s+([,.;!?·])", r"\1", cleaned)
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            cleaned = re.sub(r"([,·])\s*([,.;!?·])", r"\2", cleaned)
        return re.sub(r"^[\s,·]+", "", cleaned).strip()

    def enforce_register(self, text: str, register: str) -> str:
        """Deprecated alias for :meth:`sanitise_output`."""
        return self.sanitise_output(text, register)

    def _strip_familiar_vocatives(self, text: str) -> str:
        """Remove "φίλε", "αγόρι μου" and friends from a formal reply.

        Matching folds accents — "φιλαράκι" and "φιλαρακι" are one word — and
        the cut is made on the original string by offset, so the rest of the
        reply keeps its diacritics. Same technique as the profanity filter,
        and for the same reason it was needed there.
        """
        folded = self._strip_accents(text)
        result = text
        for match in reversed(list(self._VOCATIVE_RE.finditer(folded))):
            result = result[: match.start()] + result[match.end() :]

        result = re.sub(r"\s+([,.;!?·])", r"\1", result)
        result = re.sub(r"\s{2,}", " ", result)
        # A reply that opened with the vocative now starts with a comma.
        return re.sub(r"^[\s,·]+", "", result).strip()

    def _clean_emoji_artifacts(self, text: str) -> str:
        """Remove text emoji like (laugh), (purple_heart), etc."""
        return re.sub(r"\([a-z_]+\)", "", text).strip()

    def _remove_name_hallucinations(self, text: str) -> str:
        """Remove hallucinated name prefixes at the start of response."""
        pattern = "|".join(re.escape(n) for n in _HALLUCINATED_NAMES)
        text = re.sub(rf"^({pattern})\s+", "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Fold diacritics so 'πούστη' and 'πουστη' are the same token.

        Length-preserving for Greek: each precomposed letter decomposes to
        one base plus one combining mark, and removing the mark restores the
        original length. This lets the normalised copy be used purely for
        *locating* matches, while replacement happens on the original text.
        """
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(c for c in decomposed if not unicodedata.combining(c))

    def _substitute_folded(self, text: str, rules: dict[str, str]) -> str:
        """Apply stem→replacement rules, matching without regard to accents.

        Both loops here originally used plain ``re.sub`` on the raw text.
        That silently failed on every accented form: the corpus is written
        with accents, so "μαλάκα" never matched the key "μαλακα" and passed
        through untouched. Matching therefore happens on an accent-folded
        copy, which is length-preserving for Greek, and the edit is applied
        to the *original* string by offset so the surviving text keeps its
        accents.

        Each key is treated as a stem (``\\w*`` suffix) because Greek inflects:
        one entry has to cover "μαλάκα", "μαλάκας", "μαλάκες".
        """
        if not rules:
            return text

        stems = "|".join(re.escape(k) for k in rules)
        pattern = re.compile(rf"\b(?:{stems})\w*", re.IGNORECASE)
        folded = self._strip_accents(text)

        result = text
        # Right to left, so offsets computed on the folded copy stay valid
        # as the string changes length.
        for match in reversed(list(pattern.finditer(folded))):
            key = self._strip_accents(match.group(0)).lower()
            replacement = next(
                (v for k, v in rules.items() if key.startswith(k.lower())), ""
            )
            result = result[: match.start()] + replacement + result[match.end() :]
        return result

    def _filter_profanity(self, text: str) -> str:
        """Soften what can be softened, delete what cannot.

        Blocked words are *removed* rather than replaced with "...". The
        ellipsis was worse than the problem it solved: it announced that
        something had been censored, and its full stop made the capitaliser
        upper-case the next word, producing "τρελε ... Μ" — an artefact no
        reader can interpret as anything but a bug.
        """
        result = self._substitute_folded(text, self.profanity_replacements)
        result = self._substitute_folded(
            result, {w: "" for w in self.blocked_words}
        )

        # Close the seam left by a deletion: stranded punctuation, orphaned
        # spaces before it, and doubled whitespace.
        result = re.sub(r"\s+([,.;!?·])", r"\1", result)
        return re.sub(r"\s{2,}", " ", result).strip()

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
