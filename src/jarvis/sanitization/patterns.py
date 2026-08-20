"""Greek-specific PII detection patterns.

Detection ORDER matters and is preserved from the validated v2 design:
financial identifiers first (IBAN, ΑΦΜ, ΑΜΚΑ), then state IDs, then
phone numbers, then emails. Longer/more-specific patterns run before
shorter ones so that e.g. an 11-digit ΑΜΚΑ is never partially consumed
by a 9-digit ΑΦΜ match (word boundaries guard this as well).

FIX (v3): the Greek lowercase character class now explicitly includes
the FINAL SIGMA "ς" (U+03C2). In v2 the name-detection class listed
letters individually and omitted ς, so names ending in ς (Γιώργος,
Νίκος, Βασίλης — i.e. most Greek male names) escaped detection.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Greek character classes
# ---------------------------------------------------------------------------
# Uppercase: plain Α-Ω plus accented capitals and diaeresis forms.
GREEK_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΆΈΉΊΌΎΏΪΫ"

# Lowercase: NOTE the explicit "ς" (final sigma, U+03C2) — the v2 bug.
GREEK_LOWER = "αβγδεζηθικλμνξοπρσςτυφχψωάέήίόύώϊϋΐΰ"

GREEK_LETTER = GREEK_UPPER + GREEK_LOWER

# A capitalized Greek word of length >= 3 (e.g. Γιώργος, Μαρία).
CAPITALIZED_GREEK_WORD = rf"[{GREEK_UPPER}][{GREEK_LOWER}]{{2,}}"

# Two consecutive capitalized Greek words → candidate "Firstname Surname".
GREEK_FULL_NAME = re.compile(rf"\b{CAPITALIZED_GREEK_WORD}\s+{CAPITALIZED_GREEK_WORD}\b")

# ---------------------------------------------------------------------------
# Validators (reduce false positives on plain digit runs)
# ---------------------------------------------------------------------------


def is_valid_afm(digits: str) -> bool:
    """ΑΦΜ (tax number) mod-11 checksum: 9 digits, last is check digit."""
    if len(digits) != 9 or not digits.isdigit():
        return False
    total = sum(int(d) * 2 ** (8 - i) for i, d in enumerate(digits[:8]))
    return (total % 11) % 10 == int(digits[8])


def is_valid_amka(digits: str) -> bool:
    """ΑΜΚΑ (social security number): 11 digits, first 6 are DDMMYY birth date."""
    if len(digits) != 11 or not digits.isdigit():
        return False
    day, month = int(digits[0:2]), int(digits[2:4])
    try:
        _dt.date(2000, month if month else 1, day if day else 1)
    except ValueError:
        return False
    return 1 <= day <= 31 and 1 <= month <= 12


# ---------------------------------------------------------------------------
# Ordered PII patterns: (category, compiled regex, validator | None)
# The sanitizer applies these strictly in list order.
# ---------------------------------------------------------------------------
PII_PATTERNS: list[tuple[str, re.Pattern[str], Callable[[str], bool] | None]] = [
    # 1. Financial — Greek IBAN: "GR" + 2 check digits + 23 digits (spaces tolerated)
    ("iban", re.compile(r"\bGR\d{2}(?:\s?\d){23}\b", re.IGNORECASE), None),
    # 2. Financial — ΑΜΚΑ before ΑΦΜ (11 digits vs 9, both validated)
    ("amka", re.compile(r"\b\d{11}\b"), is_valid_amka),
    ("afm", re.compile(r"\b\d{9}\b"), is_valid_afm),
    # 3. State ID card (ταυτότητα): 1-2 Greek capitals + 6 digits, e.g. "ΑΒ 123456"
    ("id_card", re.compile(rf"\b[{GREEK_UPPER}]{{1,2}}\s?\d{{6}}\b"), None),
    # 4. Phones — mobile (69XXXXXXXX) and landline (2XXXXXXXXX), optional +30
    ("phone", re.compile(r"(?<!\d)(?:\+30\s?)?69\d{2}\s?\d{3}\s?\d{3}(?!\d)"), None),
    ("phone", re.compile(r"(?<!\d)(?:\+30\s?)?2\d{9}(?!\d)"), None),
    # 5. Email addresses
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
]

# ---------------------------------------------------------------------------
# False-positive guard: common Greek words that look like names when
# capitalized (sentence starts, greetings, days, months, exclamations).
# Extend freely — used by both contact matching and full-name detection.
# ---------------------------------------------------------------------------
GREEK_NON_NAMES: frozenset[str] = frozenset({
    # greetings / politeness
    "Γεια", "Γειά", "Καλημέρα", "Καλησπέρα", "Καληνύχτα", "Ευχαριστώ",
    "Παρακαλώ", "Συγγνώμη", "Συγνώμη", "Χαίρετε",
    # frequent sentence starters
    "Ναι", "Όχι", "Οχι", "Καλά", "Καλό", "Καλή", "Εντάξει", "Οκ",
    "Λοιπόν", "Τέλεια", "Ωραία", "Ωραίο", "Μπράβο", "Έγινε", "Εγινε",
    "Αύριο", "Σήμερα", "Χθες", "Τώρα", "Μετά", "Πριν", "Ίσως",
    "Πάμε", "Έλα", "Ελα", "Άντε", "Αντε", "Δες", "Πες", "Κοίτα",
    # days
    "Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή",
    # months
    "Ιανουάριος", "Φεβρουάριος", "Μάρτιος", "Απρίλιος", "Μάιος", "Ιούνιος",
    "Ιούλιος", "Αύγουστος", "Σεπτέμβριος", "Οκτώβριος", "Νοέμβριος", "Δεκέμβριος",
    # holidays / common nouns often capitalized
    "Πάσχα", "Χριστούγεννα", "Πρωτοχρονιά", "Θεέ", "Παναγία", "Χριστός",
    # places that appear constantly in chat (not third-party PII)
    "Αθήνα", "Θεσσαλονίκη", "Ελλάδα", "Πάτρα", "Τρίπολη",
})
