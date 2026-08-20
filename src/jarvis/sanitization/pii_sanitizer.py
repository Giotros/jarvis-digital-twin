"""PIISanitizer — removes personal data from Greek chat corpora BEFORE training.

Why before training: neural LMs memorise training data verbatim (Carlini et
al., USENIX Security 2021), so any PII present at fine-tuning time must be
assumed recoverable from the model weights. Post-generation filtering is a
defence-in-depth extra, never the primary control.

Pipeline position:
    gold.viber_training_pairs  →  PIISanitizer  →  jarvis_training_data_sanitized.json
The raw export (jarvis_training_data.json) must NEVER be passed to a trainer.

Usage:
    from jarvis.sanitization import PIISanitizer
    sanitizer = PIISanitizer.from_contacts_file("config/contacts.txt")
    result = sanitizer.sanitize("Ο Γιώργος έστειλε στο 6912345678")
    # → "Ο [ΟΝΟΜΑ] έστειλε στο [ΤΗΛΕΦΩΝΟ]"
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.sanitization.patterns import (
    GREEK_FULL_NAME,
    GREEK_NON_NAMES,
    PII_PATTERNS,
)

DEFAULT_PLACEHOLDERS = {
    "iban": "[IBAN]",
    "afm": "[ΑΦΜ]",
    "amka": "[ΑΜΚΑ]",
    "id_card": "[ΤΑΥΤΟΤΗΤΑ]",
    "phone": "[ΤΗΛΕΦΩΝΟ]",
    "email": "[EMAIL]",
    "name": "[ΟΝΟΜΑ]",
}


def strip_accents(text: str) -> str:
    """Return *text* with Greek diacritics removed (Γιώργος → Γιωργος)."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )


def name_variants(name: str) -> set[str]:
    """Generate common Greek declension variants for a personal name.

    Greek names change ending with grammatical case; matching only the
    nominative misses most real occurrences in chat text:
        Γιώργος → Γιώργου (gen.), Γιώργο (acc.), Γιώργε (voc.)
        Βασίλης → Βασίλη        Κώστας → Κώστα        Μαρία → Μαρίας
    Accent-stripped twins are added for sloppy chat spelling (γιωργος).
    """
    variants: set[str] = set()
    for token in name.split():
        token = token.strip()
        if len(token) < 3:
            continue
        forms = {token}
        if token.endswith("ος"):
            stem = token[:-2]
            forms |= {stem + "ου", stem + "ο", stem + "ε"}
        elif token.endswith(("ης", "ας")):
            forms.add(token[:-1])          # Βασίλης→Βασίλη, Κώστας→Κώστα
        elif token.endswith(("α", "η")):
            forms.add(token + "ς")         # Μαρία→Μαρίας, Ελένη→Ελένης
        for form in list(forms):
            forms.add(strip_accents(form))
        variants |= forms
    # Drop anything that is a common word rather than a name.
    return {v for v in variants if v.capitalize() not in GREEK_NON_NAMES}


@dataclass
class SanitizationReport:
    """Aggregate statistics for one sanitization run."""

    counts: dict[str, int] = field(default_factory=dict)
    records_processed: int = 0
    records_changed: int = 0

    @property
    def total_replacements(self) -> int:
        return sum(self.counts.values())

    def add(self, category: str, n: int = 1) -> None:
        if n:
            self.counts[category] = self.counts.get(category, 0) + n

    def summary(self) -> str:
        lines = [
            f"records processed : {self.records_processed}",
            f"records changed   : {self.records_changed}",
            f"total replacements: {self.total_replacements}",
        ]
        for cat, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {cat:<10} {n}")
        return "\n".join(lines)


class PIISanitizer:
    """Greek-aware PII removal with a known-contacts dictionary.

    Detection order (validated in v2, preserved):
      1. financial (IBAN → ΑΜΚΑ → ΑΦΜ)   2. state IDs   3. phones
      4. emails   5. known contact names (all declensions)
      6. generic "Firstname Surname" pattern (with non-name guard)

    Single capitalized words are deliberately NOT auto-masked unless they
    are known contacts: in Greek chat, sentence-initial capitalization
    would create massive false positives. Keep contacts.txt complete.
    """

    def __init__(
        self,
        contacts: Iterable[str] = (),
        placeholders: dict[str, str] | None = None,
        extra_non_names: Iterable[str] = (),
    ) -> None:
        self.placeholders = {**DEFAULT_PLACEHOLDERS, **(placeholders or {})}
        self.non_names = set(GREEK_NON_NAMES) | {n.capitalize() for n in extra_non_names}

        variants: set[str] = set()
        for contact in contacts:
            contact = contact.strip()
            if contact and not contact.startswith("#"):
                variants |= name_variants(contact)
                if " " in contact:                      # full "First Last" too
                    variants.add(contact)
                    variants.add(strip_accents(contact))
        # Longest-first so "Γιώργου" wins over "Γιώργο" at the same position.
        ordered = sorted(variants, key=len, reverse=True)
        self._contact_regex = (
            re.compile(
                r"\b(?:" + "|".join(map(re.escape, ordered)) + r")\b",
                re.IGNORECASE | re.UNICODE,
            )
            if variants
            else None
        )
        self.n_contacts = len(variants)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_contacts_file(cls, path: str | Path, **kwargs) -> PIISanitizer:
        """Build a sanitizer from a one-name-per-line contacts file."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls(contacts=lines, **kwargs)

    # -- core ---------------------------------------------------------------

    def sanitize(self, text: str, report: SanitizationReport | None = None) -> str:
        """Return *text* with all detected PII replaced by placeholders."""
        report = report if report is not None else SanitizationReport()

        # 1-4: ordered structural patterns (financial → IDs → phones → email)
        for category, pattern, validator in PII_PATTERNS:
            def _sub(match: re.Match[str], _cat=category, _val=validator) -> str:
                value = match.group(0)
                if _val is not None and not _val(re.sub(r"\s", "", value)):
                    return value                      # failed checksum → not PII
                report.add(_cat)
                return self.placeholders[_cat]

            text = pattern.sub(_sub, text)

        # 5: known contacts (all declension variants, case/accent tolerant)
        if self._contact_regex is not None:
            text, n = self._contact_regex.subn(self.placeholders["name"], text)
            report.add("name", n)

        # 6: generic "Firstname Surname" — guarded by the non-name list
        def _full_name_sub(match: re.Match[str]) -> str:
            first_word = match.group(0).split()[0]
            if first_word in self.non_names:
                return match.group(0)
            report.add("name")
            return self.placeholders["name"]

        text = GREEK_FULL_NAME.sub(_full_name_sub, text)
        return text

    def sanitize_records(
        self,
        records: Sequence[dict],
        fields: Sequence[str] = ("instruction", "response"),
    ) -> tuple[list[dict], SanitizationReport]:
        """Sanitize selected *fields* of every record; returns new list + report."""
        report = SanitizationReport()
        output: list[dict] = []
        for record in records:
            clean = dict(record)
            changed = False
            for f_name in fields:
                if isinstance(clean.get(f_name), str):
                    before = clean[f_name]
                    clean[f_name] = self.sanitize(before, report)
                    changed |= clean[f_name] != before
            report.records_processed += 1
            report.records_changed += int(changed)
            output.append(clean)
        return output, report

    def sanitize_json_file(
        self,
        src: str | Path,
        dst: str | Path,
        fields: Sequence[str] = ("instruction", "response"),
    ) -> SanitizationReport:
        """Sanitize a JSON list-of-records file. Refuses dst == src."""
        src, dst = Path(src), Path(dst)
        if src.resolve() == dst.resolve():
            raise ValueError("Refusing to overwrite the raw file — write to a new path.")
        records = json.loads(src.read_text(encoding="utf-8"))
        clean, report = self.sanitize_records(records, fields)
        dst.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
        return report
