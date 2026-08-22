"""Greek surname detection — the half the given-name detector never covered.

:mod:`jarvis.sanitization.greek_names` finds first names, and finding first
names is tractable: there are a few thousand of them, they repeat, and a
gazetteer of stems plus inflectional endings reaches every one. Surnames are
open-class. There is no list.

The gap was invisible until the trained model produced it. Asked about a
meeting, the twin answered with two real surnames of people at the
university — lifted from 34 occurrences in the training corpus that the
pipeline had passed as clean. They survived because they are written in
lower case, like everything else in casual Greek, and because a surname in
the accusative ("παρουσιαζω ταμπακα") carries no article, no title and no
capital to mark it.

Two mechanisms, because one is not enough:

**Morphology** catches the distinctive endings. -όπουλος, -ίδης, -άκης and
their relatives are surnames and essentially nothing else, so they can be
matched with high precision and no list.

**A gazetteer** catches the rest. Ταμπακάς ends in -ας, which is also the
ending of hundreds of ordinary words; no rule separates them. Those names
have to be named, which means a person has to look at candidates and decide.
:mod:`scripts/mine_surnames.py` produces the shortlist.

This is stated as a limitation rather than hidden: fully automatic surname
detection in lower-cased Greek is not solved here, and claiming otherwise
would be the more dangerous error given what the corpus contains.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable


def strip_accents(text: str) -> str:
    """Fold case and diacritics.

    ``casefold`` maps final sigma to sigma, which is why every gazetteer in
    this package is normalised through here at import rather than compared
    raw — the same trap that made three earlier filters silently match
    nothing.
    """
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _normalised_set(words: Iterable[str]) -> frozenset[str]:
    return frozenset(strip_accents(w) for w in words if w and w.strip())


#: Endings that are surnames and almost nothing else.
#:
#: Deliberately excludes -ας, -ης and -ος on their own. Those are the endings
#: of ordinary nouns and adjectives in their thousands, and matching them
#: would delete half the corpus — the over-redaction failure that removed
#: 268 instances of "Παρασκευή" the first time round, at a much larger scale.
#: Four endings were removed after measuring on the corpus, because each was
#: overwhelmingly a common noun rather than a name:
#:   -εση    πίεση, σχέση, εξαίρεση, σύνδεση, ανάθεση
#:   -ατου   Σαββάτου, ονόματος, ποσοστού
#:   -εα     βαρέα, ιδέα, παρέα
#: -οπουλο was removed and then restored. It matches κοτόπουλο, but it also
#: matches every surname in the accusative — "είπα στον Παπαδόπουλο" — which
#: is how surnames most often appear in these messages. One known word is
#: cheaper to list than a whole grammatical case is to lose.
#: Normalised at import, like every other table in this package.
#:
#: Written naturally, "οπουλος" ends in final sigma (ς). ``casefold`` maps
#: that to σ, so a folded word ends in σ and never matches a suffix that
#: still holds ς. Every entry here silently matched nothing until the tests
#: caught it — the fourth occurrence of this bug in the project, made while
#: the comment warning about it was ten lines above.
def _normalised_suffixes(suffixes: Iterable[str]) -> tuple[str, ...]:
    return tuple(strip_accents(s) for s in suffixes)


_DISTINCTIVE_SUFFIXES: tuple[str, ...] = _normalised_suffixes((
    "οπουλος", "οπουλου", "οπουλο",
    "ιδης", "ιδη", "ιδου", "ιδων",
    "ιαδης", "ιαδη", "ιαδου",
    "ακης", "ακη", "ακηδες",
    "ογλου",
    "ουδης", "ουδη",
    # -εσης was here and is gone. It is the genitive of every noun in -εση:
    # σύνδεσης, θέσης, κατάθεσης, διάθεσης. It flagged "σύνδεσης" sixteen
    # times in the corpus. The one surname it caught, Συρμακέσης, is in the
    # gazetteer, which is where names that share an ending with ordinary
    # grammar belong.
    "ακος", "ακου",
    "ελλης", "ελλη",
    "εας",
))

#: Endings that look like the ones above but mark ordinary morphology.
#:
#: -ιακή is the feminine adjectival ending: πτυχιακή, ταμειακή, ξενοδοχειακή,
#: επαρχιακή. It ends in -ακή and would otherwise be read as a Cretan
#: surname every time. Checked before the suffix rule and cheaper than
#: listing every adjective in the language.
_ADJECTIVAL_ENDINGS: tuple[str, ...] = _normalised_suffixes(
    ("ιακη", "ιακης", "ιακο", "ιακου")
)

#: Ordinary words that end like surnames. Checked before the suffix rule.
#:
#: Every entry here was a real false positive, found by running the detector
#: over the corpus and reading what it wanted to delete.
_NOT_SURNAMES: frozenset[str] = _normalised_set({
    # -ιδης / -ιδη family
    "ελπιδα", "ελπιδη", "παιδη", "παιδι", "κλειδη", "κλειδι", "σελιδα",
    "σελιδη", "μεριδα", "μεριδη", "πατριδα", "πατριδη", "ασπιδα",
    "σφραγιδα", "χλωριδα", "πυραμιδα", "ολυμπιαδα", "εβδομαδα",
    # -ακης / -ακη family
    "φυλακη", "θηκη", "αποθηκη", "βιβλιοθηκη", "συνθηκη", "υποθηκη",
    "μαλακη",
    # -εας / -εα family
    "ιδεα", "παρεα", "γραμμεα", "συγγραφεα", "γονεα", "ιππεα", "φονεα",
    "κουρεα", "ιερεα", "βασιλεα",
    # -ατος / -ατου family
    "θανατος", "θανατου", "καματος", "πλατος", "κρατος", "κρατους",
    "γεματος", "αρματος", "ονοματος", "χρηματος", "πραγματος",
    "συστηματος", "θεματος", "προβληματος", "αιματος", "σωματος",
    "κτηματος", "τμηματος", "ρευματος", "κληματος", "βληματος",
    "στοματος", "ποσοστου", "κοστους",
    # -οπουλο on food, not people
    "κοτοπουλο", "κοτοπουλα", "κοτοπουλου", "αρχοντοπουλο",
    # -ακος / -ακου
    "μακρακος",
    # -εση / -εσης
    "θεση", "θεσης", "προθεση", "καταθεση", "συνθεση", "διαθεση",
    "επιθεση", "αφαιρεση", "διαιρεση", "παρεση", "αιρεση", "περιπτωση",
    "αποφαση", "σκεψη", "λεση", "πεση", "αντιθεση", "υποθεση",
    "παραθεση", "εκθεση", "μεταθεση", "προσθεση", "ενθεση",
    # The single worst false positive: 113 occurrences, and it ends in the
    # same four letters as a -ίδης surname.
    "επειδη",
    "ειδη", "ειδηση", "ειδησεις", "συνηθεια",
    # -ακη and -ακης on ordinary words
    "κυριακη", "κυριακης", "μαλακη", "μαλακια", "κατοικη",
    # -ουδη
    "τραγουδη", "τραγουδι", "λουλουδη", "λουλουδι", "χνουδη",
    # -εας
    "γραμματεας", "συγγραφεας", "γονεας", "ιππεας", "φονεας",
    "κουρεας", "ιερεας", "βασιλεας", "μεταφορεας", "εργαζομενος",
    "αποστολεας", "παραλαβεας", "εισαγωγεας", "εξαγωγεας", "χειριστεας",
    # Misspellings of excluded words inherit their exclusion. "επεειδη" is
    # "επειδή" with a slipped finger, and casual typing produces these
    # constantly — a detector that only knows correct spellings will keep
    # finding "names" in typos forever.
    "επεειδη", "επιδη", "επειδι",
})

#: Place names with surname morphology.
#:
#: Greek toponyms and family names share endings constantly — Μαρκόπουλο is
#: a town, Μαρκόπουλος a person. Deleting the town from someone's messages
#: is the same class of damage as deleting "Παρασκευή", which cost 268
#: occurrences the first time this pipeline over-reached.
_PLACE_NAMES: frozenset[str] = _normalised_set({
    "μαρκοπουλο", "μαρκοπουλου", "γιαννιτσα", "γιαννιτσων",
    "θεσσαλονικη", "θεσσαλονικης", "τριπολη", "τριπολης",
    "κερκυρα", "κερκυρας", "καλαματα", "καλαματας",
    "χαλκιδικη", "χαλκιδικης", "αττικη", "αττικης",
    "κρητη", "κρητης", "ροδοπη", "ροδοπης", "αρκαδια",
    "πελοποννησο", "πελοποννησου", "μακεδονια", "θρακη",
})

#: Confirmed surnames, one per line, loaded from disk.
#:
#: Never committed — it is a list of identifiable third parties, which is
#: precisely the category this whole pipeline exists to remove. Kept beside
#: ``contacts.txt``, under the same gitignore rule.
#: George's own names, which must never be redacted.
#:
#: The twin speaks as him: removing his own name leaves "Με λένε" with
#: nothing after it. :mod:`greek_names` has always guarded against this and
#: this module did not, which mattered as soon as the candidate miner
#: surfaced his first name 35 times as a likely third party.
#:
#: Kept here rather than read from identity.yaml so the sanitiser has no
#: dependency on a file that is gitignored and may be absent at import.
SELF_NAMES: frozenset[str] = _normalised_set({
    "γιωργος", "γιωργο", "γιωργου", "γιωργη", "γεωργιος", "γεωργιου",
    "τροχιδης", "τροχιδη", "τροχιδου",
})

_GAZETTEER_ENV = "JARVIS_SURNAMES_FILE"
_DEFAULT_GAZETTEER = Path("config/surnames.txt")

_gazetteer: frozenset[str] | None = None


def load_gazetteer(path: Path | None = None, force: bool = False) -> frozenset[str]:
    """Load confirmed surname stems, or an empty set if none exist.

    A missing file is normal and not an error: the morphological rules still
    work without it. It does mean coverage is partial, which
    :func:`gazetteer_status` reports so the gap is visible rather than
    assumed away.
    """
    global _gazetteer
    if _gazetteer is not None and not force:
        return _gazetteer

    candidates = [path] if path else [
        Path(os.environ[_GAZETTEER_ENV]) if os.environ.get(_GAZETTEER_ENV) else None,
        Path("/app/config/surnames.txt"),
        Path(__file__).resolve().parents[3] / "config" / "surnames.txt",
        _DEFAULT_GAZETTEER,
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            stems = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
            _gazetteer = _normalised_set(stems)
            return _gazetteer

    _gazetteer = frozenset()
    return _gazetteer


def gazetteer_status() -> str:
    """Human-readable coverage note, for logs and the smoke test."""
    n = len(load_gazetteer())
    if n:
        return f"{n} confirmed surname stems loaded"
    return (
        "no surname gazetteer — only distinctive endings are detected; "
        "run scripts/mine_surnames.py to build one"
    )


_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def is_surname(word: str) -> bool:
    """Whether a single token looks like a surname.

    Order matters. The gazetteer is consulted before the exclusions so a
    confirmed name is never overridden by a coincidence; everything else has
    to clear the exclusions first, because a false positive here deletes a
    word from a person's own vocabulary.
    """
    folded = strip_accents(word)
    if len(folded) < 5:
        return False

    # Checked before the gazetteer, so that adding his own surname to the
    # file by mistake cannot silence the twin about itself.
    if folded in SELF_NAMES:
        return False

    if any(folded.startswith(stem) for stem in load_gazetteer()):
        return True

    if folded in _NOT_SURNAMES or folded in _PLACE_NAMES:
        return False
    if any(folded.endswith(ending) for ending in _ADJECTIVAL_ENDINGS):
        return False

    return any(folded.endswith(suffix) for suffix in _DISTINCTIVE_SUFFIXES)


def find_surnames(text: str) -> list[str]:
    """Every token in ``text`` that looks like a surname, in order."""
    return [w for w in _WORD_RE.findall(text or "") if is_surname(w)]


def redact_surnames(text: str, placeholder: str = "[SURNAME]") -> tuple[str, int]:
    """Replace surnames with a placeholder.

    Returns the cleaned text and the number of replacements, so a caller can
    fail a build on a non-zero count rather than discovering the leak in a
    model that has already been trained.
    """
    if not text:
        return text, 0

    count = 0
    out: list[str] = []
    last = 0
    for match in _WORD_RE.finditer(text):
        if not is_surname(match.group(0)):
            continue
        out.append(text[last:match.start()])
        out.append(placeholder)
        last = match.end()
        count += 1
    out.append(text[last:])
    return "".join(out), count
