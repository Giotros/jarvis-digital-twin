"""Tests for Greek given-name detection.

The v3 sanitiser leaked third-party first names into 8.1% of the training
corpus because its rule required two consecutive *capitalised* words. Each
test below pins one property that failure depended on, so the regression
cannot silently return.
"""

from __future__ import annotations

import pytest

from jarvis.sanitization.greek_names import (
    EXACT_GIVEN_NAMES,
    GIVEN_NAME_STEMS,
    NAME_LIKE_COMMON_WORDS,
    find_given_names,
    redact_given_names,
    strip_accents,
)


# ── Normalisation ───────────────────────────────────────────────


def test_strip_accents_removes_diacritics():
    assert strip_accents("Γιώτα") == "γιωτα"
    assert strip_accents("ΠΑΝΑΓΙΩΤΗΣ") == strip_accents("παναγιώτης")


def test_strip_accents_preserves_length():
    """Span offsets index the normalised string; length must not shift."""
    for s in ["Γιώτα", "ΔΗΜΉΤΡΗ", "Ελένης", "ς", "άέήίόύώ"]:
        assert len(strip_accents(s)) == len(s), s


def test_casefold_maps_final_sigma():
    """The bug this module was written around: ς → σ under casefold.

    Endings must be normalised the same way or they never match.
    """
    assert strip_accents("Κώστας").endswith("σ")
    assert redact_given_names("Κώστας")[1] == 1


# ── The regression: lowercase, unaccented, single first name ────


@pytest.mark.parametrize("text", [
    "γιωτα μολις ειδα κ αλλο προιον",
    "παναγιωτη ελα εδω",
    "δημητρη τι κανεις",
    "ελενη θα παμε αυριο",
    "κατερινα εστειλε μηνυμα",
])
def test_lowercase_vocative_names_redacted(text):
    """The exact form the v3 rule was blind to."""
    out, n = redact_given_names(text)
    assert n >= 1
    assert "[NAME]" in out


@pytest.mark.parametrize("name", [
    "Κώστας", "Γιάννης", "Μάνθος", "Βασίλης", "Νίκος",
    "Δημήτρης", "Παναγιώτης", "Μαρία", "Ελένη", "Γιώτα",
])
def test_nominative_names_redacted(name):
    assert redact_given_names(name)[1] == 1


@pytest.mark.parametrize("form", [
    "Δημήτρη", "Δημήτρης", "Δημήτρη μου", "δημητρακη",
])
def test_inflected_forms_all_caught(form):
    """Stem matching must survive Greek inflection, not just nominative."""
    assert redact_given_names(form)[1] >= 1


# ── Precision guards ────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "η νίκη ήταν σημαντική",
    "τι ώρα λέμε",
    "με μεγάλη χαρά",
    "άναψε το φως",
    "καλή μάρκα είναι",
])
def test_common_vocabulary_not_redacted(text):
    """False positives cost training data; the blocklist must hold."""
    assert redact_given_names(text)[1] == 0


@pytest.mark.parametrize("word", [
    # Days that are also given names — an early version deleted 129 instances
    # of "Παρασκευή" and 126 of "Κυριακή" from the corpus.
    "Παρασκευή", "Παρασκευής", "Κυριακή", "Κυριακής",
    # Months that are also given names
    "Ιούλιος", "Ιούλιο", "Ιουλίου", "Μάρτιος", "Αύγουστος",
    # Places built on given-name stems
    "Γιαννιτσά", "Μαρκόπουλο",
])
def test_calendar_and_places_survive(word):
    """Scheduling vocabulary must not be destroyed by name redaction."""
    assert redact_given_names(word)[1] == 0, f"{word} was wrongly redacted"


@pytest.mark.parametrize("text", [
    "Θα σε δω Παρασκευή",
    "τον Ιούλιο φεύγω",
    "είμαι από τα Γιαννιτσά Πέλλας",
])
def test_calendar_words_in_context(text):
    assert redact_given_names(text)[1] == 0


def test_male_names_sharing_calendar_stems_still_redacted():
    """Κυριάκος and Παρασκευάς are people, not days — the guard is form-specific."""
    assert redact_given_names("Κυριάκος")[1] == 1
    assert redact_given_names("Παρασκευάς")[1] == 1


@pytest.mark.parametrize("text", [
    "ο Γιώργος είμαι εγώ",
    "George: θα δω",
    "giorgos edw",
])
def test_data_subject_preserved(text):
    """George is not third-party PII and is a structural corpus marker."""
    assert redact_given_names(text)[1] == 0


def test_self_names_configurable():
    """A different data subject must be protectable without code changes."""
    assert redact_given_names("μαρια", self_names=("μαρι",))[1] == 0
    assert redact_given_names("μαρια", self_names=("γιωργ",))[1] == 1


# ── Span mechanics ──────────────────────────────────────────────


def test_multiple_names_in_one_message():
    out, n = redact_given_names("γιωτα και παναγιωτη ελατε")
    assert n == 2
    assert out.count("[NAME]") == 2
    assert "γιωτα" not in out and "παναγιωτη" not in out


def test_surrounding_text_preserved_exactly():
    out, _ = redact_given_names("πες στη γιωτα οτι αργω 10 λεπτα")
    assert out == "πες στη [NAME] οτι αργω 10 λεπτα"


def test_spans_are_ordered_and_non_overlapping():
    spans = find_given_names("γιωτα και δημητρη και ελενη")
    starts = [s for s, _, _ in spans]
    assert starts == sorted(starts)
    for (_, end_a, _), (start_b, _, _) in zip(spans, spans[1:]):
        assert end_a <= start_b


def test_no_names_returns_input_unchanged():
    text = "τι ωρα ξεκιναμε αυριο"
    out, n = redact_given_names(text)
    assert (out, n) == (text, 0)


def test_empty_string_is_safe():
    assert redact_given_names("") == ("", 0)


def test_custom_placeholder():
    out, _ = redact_given_names("γιωτα", placeholder="<REDACTED>")
    assert out == "<REDACTED>"


# ── Gazetteer hygiene ───────────────────────────────────────────

def test_stems_are_normalised_and_long_enough():
    """Short or accented stems cause collisions and silent misses."""
    for stem in GIVEN_NAME_STEMS:
        assert stem == strip_accents(stem), f"{stem} is not normalised"
        assert len(stem) >= 4, f"{stem} is too short to be safe"


def test_exact_names_are_normalised():
    for name in EXACT_GIVEN_NAMES:
        assert name == strip_accents(name)


def test_blocklist_is_normalised():
    for word in NAME_LIKE_COMMON_WORDS:
        assert word == strip_accents(word)


def test_redaction_is_idempotent():
    once, _ = redact_given_names("γιωτα και δημητρη")
    twice, n = redact_given_names(once)
    assert twice == once and n == 0
