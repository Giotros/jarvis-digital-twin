"""Tests for surname detection.

The gap these cover was invisible until the trained model produced it. Asked
about a meeting, the deployed twin answered with two real surnames of people
at the university — lifted from 34 occurrences in a corpus the pipeline had
passed as clean.

Half these tests are about *not* redacting. Over-redaction is the failure
this package has already made once, at the cost of 268 instances of
"Παρασκευή", and surname endings collide with ordinary Greek far more often
than given names do.
"""

import pytest

from jarvis.sanitization.greek_surnames import (
    find_surnames,
    is_surname,
    load_gazetteer,
    redact_surnames,
    strip_accents,
)


# ── Distinctive morphology ──────────────────────────────────────

@pytest.mark.parametrize("word", [
    "Παπαδόπουλος", "Παπαδοπούλου",
    "Ιωαννίδης", "Ιωαννίδη", "Ιωαννίδου",
    "Αναστασιάδης", "Αναστασιάδου",
    "Κασελάκης", "Κασελάκη",
    "Βεϊνόγλου", "Ιπλικτζόγλου",
    "Μαυρούδης", "Μαυρούδη",
])
def test_distinctive_endings_are_recognised(word):
    """-όπουλος, -ίδης, -άκης, -όγλου are surnames and little else."""
    assert is_surname(word)


@pytest.mark.parametrize("word", [
    "παπαδοπουλου", "ΠΑΠΑΔΟΠΟΥΛΟΥ", "Παπαδοπούλου",
])
def test_case_and_accents_do_not_matter(word):
    """The corpus is casual Greek: no capitals, inconsistent accents.

    That is exactly why the capitalisation rule missed 8.1% of given names,
    and it applies with equal force here.
    """
    assert is_surname(word)


# ── Over-redaction ──────────────────────────────────────────────

@pytest.mark.parametrize("word", [
    "επειδή",       # 113 occurrences — the worst false positive found
    "επειδη",
    "ειδήσεις",
    "Κυριακή",      # a day, and a given name, and never a surname here
    "πτυχιακή",     # -ιακή is adjectival, not Cretan
    "ταμειακή", "ξενοδοχειακή", "επαρχιακή",
    "πίεση", "σχέση", "εξαίρεση", "σύνδεση", "ανάθεση", "θέση",
    "Σαββάτου", "ονόματος", "ποσοστού",
    "κοτόπουλο",    # -όπουλο, but dinner
    "ιδέα", "παρέα", "βαρέα",
    "γραμματέας", "συγγραφέας", "γονέας",
    "τραγούδι", "λουλούδι",
    "βιβλιοθήκη", "αποθήκη", "συνθήκη",
    "ελπίδα", "σελίδα", "πατρίδα", "εβδομάδα",
])
def test_ordinary_words_are_not_redacted(word):
    """Every one of these was a real false positive on the corpus.

    Four endings were dropped from the rule set because of them: -εση,
    -ατου, -εα and the bare -οπουλο. A detector that deletes "επειδή" 113
    times has destroyed more of the person's voice than it protected.
    """
    assert not is_surname(word), f"would redact the ordinary word {word!r}"


@pytest.mark.parametrize("place", [
    "Μαρκόπουλο", "Μαρκοπούλου", "Γιαννιτσά", "Θεσσαλονίκη",
    "Τρίπολη", "Κέρκυρα", "Χαλκιδική", "Κρήτη",
])
def test_place_names_survive(place):
    """Toponyms and family names share endings constantly in Greek.

    Μαρκόπουλο is a town, Μαρκόπουλος a person. Deleting the town from
    someone's messages is the same damage as deleting "Παρασκευή".
    """
    assert not is_surname(place)


def test_short_tokens_are_ignored():
    """A four-letter match is far more likely to be a common word."""
    assert not is_surname("ιδη")
    assert not is_surname("ακη")


# ── The gazetteer ───────────────────────────────────────────────

def test_gazetteer_covers_what_morphology_cannot(tmp_path):
    """Ταμπακάς ends in -ας, like hundreds of ordinary words.

    No rule separates them, which is why the file exists and why building it
    is a human decision rather than an inferred one.
    """
    path = tmp_path / "surnames.txt"
    path.write_text("# σχόλιο\nταμπακ\nσυρμακεσ\n", encoding="utf-8")
    load_gazetteer(path, force=True)
    try:
        assert is_surname("ταμπακα")
        assert is_surname("Ταμπακάς")
        assert is_surname("συρμακεσης")
        assert is_surname("Συρμακέση")
    finally:
        load_gazetteer(force=True)


def test_missing_gazetteer_is_not_an_error(tmp_path):
    """Morphology still works without it; coverage is just partial."""
    load_gazetteer(tmp_path / "absent.txt", force=True)
    try:
        assert load_gazetteer(tmp_path / "absent.txt") == frozenset()
        assert is_surname("Παπαδόπουλος")
    finally:
        load_gazetteer(force=True)


def test_comments_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "surnames.txt"
    path.write_text("# επικεφαλίδα\n\n  ταμπακ  \n\n", encoding="utf-8")
    load_gazetteer(path, force=True)
    try:
        assert load_gazetteer(path) == frozenset({"ταμπακ"})
    finally:
        load_gazetteer(force=True)


# ── Redaction ───────────────────────────────────────────────────

def test_redaction_reports_a_count():
    """A caller should be able to fail a build on a non-zero count.

    Discovering the leak in a model that has already been trained is what
    happened, and it cost a retraining cycle.
    """
    _, count = redact_surnames("ο Παπαδόπουλος και ο Ιωαννίδης")
    assert count == 2


def test_redaction_leaves_the_rest_intact():
    text = "Μίλησα με τον Παπαδόπουλο για τη δουλειά χθες το βράδυ."
    cleaned, _ = redact_surnames(text)
    assert "δουλειά" in cleaned
    assert "χθες το βράδυ" in cleaned
    assert "Παπαδόπουλο" not in cleaned


def test_clean_text_is_returned_unchanged():
    text = "Επειδή είχα πτυχιακή την Κυριακή, δεν πρόλαβα."
    cleaned, count = redact_surnames(text)
    assert count == 0
    assert cleaned == text


def test_find_surnames_preserves_order():
    found = find_surnames("ο Ιωαννίδης είπε στον Παπαδόπουλο")
    assert found == ["Ιωαννίδης", "Παπαδόπουλο"]


def test_empty_input():
    assert redact_surnames("") == ("", 0)
    assert find_surnames("") == []


# ── Normalisation ───────────────────────────────────────────────

def test_final_sigma_folds_like_everywhere_else():
    """casefold maps ς to σ, which broke three earlier filters silently."""
    assert strip_accents("Παπαδόπουλος") == strip_accents("ΠΑΠΑΔΟΠΟΥΛΟΣ")


# ── The twin's own name ─────────────────────────────────────────

@pytest.mark.parametrize("word", [
    "Γιώργος", "Γιώργο", "γιωργο", "Γεώργιος",
    "Τροχίδης", "Τροχίδη", "τροχιδη",
])
def test_the_twin_never_redacts_itself(word):
    """"Με λένε [SURNAME]" is not an improvement on a privacy leak.

    His first name fills name frames more often than anyone else's — 35
    occurrences in the mined candidate list — so this is the single most
    likely wrong entry for the gazetteer to acquire.
    """
    assert not is_surname(word)


def test_self_name_survives_even_if_added_to_the_gazetteer(tmp_path):
    """A mistaken entry must not be able to silence the twin about itself."""
    path = tmp_path / "surnames.txt"
    path.write_text("τροχιδ\n", encoding="utf-8")
    load_gazetteer(path, force=True)
    try:
        assert not is_surname("Τροχίδης")
    finally:
        load_gazetteer(force=True)
