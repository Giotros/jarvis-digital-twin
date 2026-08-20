"""Greek given-name detection for PII sanitisation.

Why this module exists
----------------------
The v3 sanitiser detected personal names with :data:`patterns.GREEK_FULL_NAME`,
which requires *two consecutive capitalised* Greek words — the "Μαρία
Παπαδοπούλου" shape. An audit of the 13,289-pair corpus found that rule missed
1,070 records (8.1%), because instant-messaging Greek does not look like that:
people write lowercase, unaccented, and address each other by **bare first
name in the vocative** — ``γιωτα μολις ειδα...``, ``παναγιωτη ελα``.

A capitalisation-dependent rule is structurally blind to the dominant real
form, so no amount of tuning fixes it. This module detects names
*morphologically* instead: a curated stem gazetteer matched case- and
diacritic-insensitively, allowing the inflectional endings Greek nouns take.

Design decisions
----------------
* **Stems, not surface forms.** ``Παναγιώτης / Παναγιώτη / Παναγιώτη μου``
  all reduce to the stem ``παναγιωτ``. Listing surface forms would need
  4-6 entries per name and would still miss diminutives.
* **Explicit ending set.** Matching ``stem + .*`` would over-fire; matching
  a closed set of nominal endings keeps precision high.
* **Blocklist over cleverness.** A few stems collide with ordinary vocabulary
  (``νικ-`` → *νίκη* "victory", ``χαρ-`` → *χαρά* "joy"). Those are listed
  explicitly in :data:`NAME_LIKE_COMMON_WORDS` and never redacted.
* **The data subject is preserved.** George's own name is not third-party PII
  and is a structural marker in the corpus (``George: ...``), so it is passed
  in via ``self_names`` and skipped.

Recall is deliberately favoured over precision: a false positive costs one
redacted common word in a training pair; a false negative leaks a real
person's name into a model that will be demonstrated publicly.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

__all__ = [
    "GIVEN_NAME_STEMS",
    "NAME_LIKE_COMMON_WORDS",
    "strip_accents",
    "build_name_pattern",
    "find_given_names",
    "redact_given_names",
]


def strip_accents(text: str) -> str:
    """Lowercase and remove diacritics: 'Γιώτα' → 'γιωτα'.

    Chat Greek is written with and without accents interchangeably, so all
    matching happens in this normalised space.
    """
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _normalised_set(words: Iterable[str]) -> frozenset[str]:
    """Normalise a gazetteer at import time.

    Every lookup happens against accent-stripped, casefolded text. Entries
    written naturally — with accents, with final sigma ``ς`` — would therefore
    never match. That is a silent failure: a blocklist that never fires just
    looks like a stricter filter, and a stem that never fires looks like a
    name the gazetteer does not cover. Normalising here makes the collections
    correct by construction rather than by author discipline.
    """
    return frozenset(strip_accents(w) for w in words)


#: Stems of common Greek given names, accent-stripped and lowercased.
#: Sourced from frequency of Greek first names; extend as the corpus demands.
#: Keep stems >= 4 characters — shorter ones collide with ordinary words.
GIVEN_NAME_STEMS: frozenset[str] = _normalised_set({
    # ── male ──
    "παναγιωτ", "δημητρ", "γιανν", "ιωανν", "κωνσταντιν", "κωστ",
    "αποστολ", "αθανασ", "θαναση", "βασιλ", "νικολ", "χρηστ",
    "μιχαλ", "μιχαηλ", "αντων", "στελ", "στυλιαν", "πετρ",
    "θεοδωρ", "σπυρ", "ανδρε", "αλεξανδρ", "εμμανου", "μανωλ",
    "ηλια", "λευτερ", "ελευθερ", "σωτηρ", "χαραλαμπ", "λαμπρ",
    "μανθ", "θωμα", "στεφαν", "γρηγορ", "παυλ", "μαρκ",
    "αργυρ", "βαγγελ", "ευαγγελ", "ζαχαρ", "κυριακ", "λεωνιδ",
    "ορεστ", "περικλ", "σαββ", "φιλιππ", "χριστοφ",
    # ── female ──
    "μαρι", "ελεν", "κατεριν", "αικατεριν", "σοφι",
    "γιωτ", "παναγιωτα", "δημητρα", "βασιλικ", "αναστασ", "αννα",
    "χριστιν", "ευαγγελι", "ιωανν", "δεσποιν", "ελισαβετ", "ευτυχ",
    "θεοδωρα", "ιουλι", "καλλιοπ", "κυριακη", "λαμπριν", "μαγδαλ",
    "νικολετ", "ολγα", "παρασκευ", "πηνελοπ", "ραφαηλ", "σταυρ",
    "στεργ", "τριανταφυλλ", "φανη", "φωτειν", "χαρικλ", "χρυσ",
    "αγγελικ", "αδαμαντ", "αλεξανδρα", "αμαλι", "ανδρομαχ",
    "αρετ", "αρτεμ", "ασπασ", "αφροδιτ", "βαρβαρ", "γεωργι",
    "ειρην", "ελπιδ", "ερασμ", "ευαγγελια", "ευανθ",
    "θαλει", "θεοφαν", "κλεοπατρ", "κωνσταντιν", "λεμον",
    "μαργαριτ", "μελπομεν", "ξανθ", "ουρανι", "πολυξεν",
    "σμαραγδ", "σταματ", "τερψ", "υπαπαντ", "φιλοθε",
    # "ροδ" deliberately excluded: 3 letters, collides with ρόδα (wheel),
    # ρόδι (pomegranate) and Ρόδος (Rhodes). Ροδούλα is rare enough that
    # the false-positive cost outweighs the recall gain.
})

#: Inflectional endings a Greek given name can carry. Ordered longest-first
#: inside the regex so that e.g. "-ουλα" wins over "-α".
_NAME_ENDINGS: tuple[str, ...] = (
    "ιτσα", "ουλα", "ακης", "ακη", "ακι",       # diminutives
    "ιδης", "ιδη", "οπουλος", "οπουλου",         # patronymic-ish
    "ος", "ας", "ης", "ες", "ους", "ους",
    "ου", "ων", "οι", "ες",
    "α", "η", "ο", "ε", "ς", "ι", "ω",
    "",                                            # bare stem
)

#: Ordinary vocabulary that collides with a name stem. Never redacted.
#: This blocklist is corpus-specific. If the data source changes, re-audit it:
#: a word that is ordinary vocabulary in one corpus may be a person in another.
NAME_LIKE_COMMON_WORDS: frozenset[str] = _normalised_set({
    "νικη", "νικης", "νικο", "νικα",     # νίκη = victory
    "χαρα", "χαρας",                       # χαρά = joy
    "ελπιδα", "ελπιδας",                   # ελπίδα = hope
    "φως", "φωτα", "φωτο",                 # φως = light
    "μαρκα", "μαρκας",                     # μάρκα = brand
    "ωρα", "ωρες",                         # ώρα = hour
})


#: Short names whose stems collide badly with ordinary vocabulary — notably
#: ``νικ-``, which is also the verb *νικώ* "to win". These are matched as
#: exact whole words only, with no ending expansion.
EXACT_GIVEN_NAMES: frozenset[str] = _normalised_set({
    "νικος", "νικο", "νικου",             # Νίκος — stem νικ- is also νικώ "to win"
    "ευα", "ευας",                          # Εύα
    "ζωη", "ζωης",                          # Ζωή — also "life"; recall wins
    "τασος", "τασο", "τασου",             # Τάσος — stem τασ- is also τάση "trend"
    "φωτης", "φωτη",                        # Φώτης — stem φωτ- is also φως/φωτό
})


def build_name_pattern(
    stems: Iterable[str] = GIVEN_NAME_STEMS,
    exact: Iterable[str] = EXACT_GIVEN_NAMES,
) -> re.Pattern[str]:
    """Compile the stem gazetteer into one alternation regex.

    Both stems and endings are pushed through :func:`strip_accents` before
    compilation. This is not cosmetic: ``str.casefold()`` maps final sigma
    ``ς`` (U+03C2) to ``σ`` (U+03C3), so a literal ending written as ``"ας"``
    would never match normalised text. The same class of bug — final sigma
    handling — is documented in ``patterns.py`` for the v2 sanitiser.

    Stems are sorted longest-first so ``παναγιωτα`` is preferred over
    ``παναγιωτ``; ``exact`` names are matched as whole words with no endings.
    """
    ordered = sorted({strip_accents(s) for s in stems}, key=len, reverse=True)
    stem_alt = "|".join(re.escape(s) for s in ordered)

    endings = {strip_accents(e) for e in _NAME_ENDINGS}
    end_alt = "|".join(
        re.escape(e) for e in sorted(endings, key=len, reverse=True)
    )

    exact_norm = sorted({strip_accents(e) for e in exact}, key=len, reverse=True)
    exact_alt = "|".join(re.escape(e) for e in exact_norm)

    return re.compile(
        rf"\b(?:(?:{stem_alt})(?:{end_alt})|(?:{exact_alt}))\b"
    )


_NAME_RE = build_name_pattern()


def find_given_names(
    text: str, self_names: Iterable[str] = ("γιωργ", "george", "giorgos")
) -> list[tuple[int, int, str]]:
    """Locate third-party given names as ``(start, end, matched_text)`` spans.

    Offsets index into the *accent-stripped* form of ``text``. Because
    :func:`strip_accents` is length-preserving for Greek (it only drops
    combining marks after NFD, and Greek precomposed letters decompose to
    exactly one base + one mark), the spans map back to the original string
    one-to-one.
    """
    normalised = strip_accents(text)
    self_norm = tuple(strip_accents(s) for s in self_names)

    spans: list[tuple[int, int, str]] = []
    for match in _NAME_RE.finditer(normalised):
        word = match.group(0)
        if word in NAME_LIKE_COMMON_WORDS:
            continue
        if any(word.startswith(s) for s in self_norm):
            continue
        spans.append((match.start(), match.end(), word))
    return spans


def redact_given_names(
    text: str,
    placeholder: str = "[NAME]",
    self_names: Iterable[str] = ("γιωργ", "george", "giorgos"),
) -> tuple[str, int]:
    """Replace third-party given names with ``placeholder``.

    Returns ``(redacted_text, n_replacements)``. Replacement walks the spans
    in reverse so earlier offsets stay valid as the string is rewritten.
    """
    spans = find_given_names(text, self_names=self_names)
    if not spans:
        return text, 0

    out = text
    for start, end, _word in reversed(spans):
        out = out[:start] + placeholder + out[end:]
    return out, len(spans)
