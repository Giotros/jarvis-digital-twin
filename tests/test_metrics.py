"""Tests for the evaluation metrics.

Each test encodes a claim the thesis will make about what the metric
measures. If a metric cannot distinguish the cases below, it should not be
reported as evidence.
"""

from __future__ import annotations

import pytest

from jarvis.evaluation.metrics import (
    aggregate_report,
    assistant_drift_rate,
    distinct_n,
    grounding_score,
    mean_pairwise_similarity,
    refusal_rate,
    repetition_rate,
    report_markdown,
    style_breakdown,
    style_distance,
    style_profile,
)

# Short, first-person, Greek — how George actually writes.
GEORGE_LIKE = [
    "Καλά είμαι, εσύ τι κάνεις;",
    "Ναι ρε, θα έρθω. Τι ώρα λέμε;",
    "Δεν θυμάμαι ακριβώς, θα το κοιτάξω.",
    "Πάω για τρέξιμο μετά, έχω κανονίσει.",
]

# Long, formal, assistant-voiced — the failure mode.
ASSISTANT_LIKE = [
    "Γεια σας! Πώς μπορώ να σας βοηθήσω σήμερα με το αίτημά σας;",
    "Ως τεχνητή νοημοσύνη, δεν έχω τη δυνατότητα να παρευρεθώ σε εκδηλώσεις.",
    "Είμαι εδώ για να σας παρέχω πληροφορίες σχετικά με οποιοδήποτε θέμα.",
    "Θα χαρώ πολύ να σας εξυπηρετήσω με κάθε δυνατή λεπτομέρεια και ακρίβεια.",
]


# ── Style profiling ─────────────────────────────────────────────


def test_profile_captures_length_difference():
    george = style_profile(GEORGE_LIKE)
    assistant = style_profile(ASSISTANT_LIKE)
    assert george.mean_words_per_response < assistant.mean_words_per_response


def test_profile_detects_assistant_register():
    assert style_profile(ASSISTANT_LIKE).assistant_tell_rate > 0.5
    assert style_profile(GEORGE_LIKE).assistant_tell_rate == 0.0


def test_profile_detects_first_person():
    assert style_profile(GEORGE_LIKE).first_person_rate == 1.0


@pytest.mark.parametrize("text", [
    "Πάω για ύπνο",          # -ω active
    "Δεν θυμάμαι τίποτα",     # -μαι mediopassive
    "Θα έρθω αύριο",          # -ω subjunctive
    "Εγώ το έκανα",           # explicit pronoun
    "Το βιβλίο μου",          # possessive clitic
])
def test_first_person_detected_morphologically(text):
    """Greek marks person by inflection — a word list cannot cover it."""
    assert style_profile([text]).first_person_rate == 1.0


@pytest.mark.parametrize("text", [
    "Τι κάνεις εσύ;",         # 2nd person
    "Έλα εδώ κάτω",           # -ω adverbs, not verbs
    "Αυτό είναι σωστό",       # 3rd person
])
def test_third_and_second_person_not_counted(text):
    assert style_profile([text]).first_person_rate == 0.0


def test_greek_ratio_separates_scripts():
    assert style_profile(["Καλημέρα σου"]).greek_ratio == 1.0
    assert style_profile(["Good morning"]).greek_ratio == 0.0
    mixed = style_profile(["Καλημέρα, all good"]).greek_ratio
    assert 0.0 < mixed < 1.0


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        style_profile([])
    with pytest.raises(ValueError):
        style_profile(["", "   "])


# ── Style distance ──────────────────────────────────────────────


def test_identical_profiles_have_zero_distance():
    p = style_profile(GEORGE_LIKE)
    assert style_distance(p, p) == 0.0


def test_distance_is_symmetric():
    a, b = style_profile(GEORGE_LIKE), style_profile(ASSISTANT_LIKE)
    assert style_distance(a, b) == pytest.approx(style_distance(b, a))


def test_distance_bounded_in_unit_interval():
    a, b = style_profile(GEORGE_LIKE), style_profile(ASSISTANT_LIKE)
    assert 0.0 <= style_distance(a, b) <= 1.0


def test_assistant_voice_is_far_from_george():
    """The metric must separate the two registers, or it measures nothing."""
    george = style_profile(GEORGE_LIKE)
    similar = style_profile([
        "Μια χαρά, εσύ;",
        "Ναι θα περάσω αργότερα.",
        "Δεν ξέρω ακόμα, θα δω.",
        "Πάω γυμναστήριο τώρα.",
    ])
    assistant = style_profile(ASSISTANT_LIKE)
    assert style_distance(george, similar) < style_distance(george, assistant)


def test_breakdown_names_the_drifting_feature():
    b = style_breakdown(style_profile(GEORGE_LIKE), style_profile(ASSISTANT_LIKE))
    assert b["assistant_tell_rate"] > 0.5
    assert set(b) >= {"greek_ratio", "first_person_rate"}


# ── Grounding ───────────────────────────────────────────────────


def test_fully_grounded_response_scores_high():
    ctx = "Ο Γιώργος σπούδασε πληροφορική στο Ιόνιο Πανεπιστήμιο στην Κέρκυρα."
    assert grounding_score("Σπούδασα πληροφορική στην Κέρκυρα.", ctx) > 0.6


def test_hallucinated_specifics_score_low():
    ctx = "Ο Γιώργος σπούδασε πληροφορική στην Κέρκυρα."
    assert grounding_score("Σπούδασα ιατρική στο Παρίσι το 2015.", ctx) < 0.4


def test_empty_context_flags_everything_as_ungrounded():
    assert grounding_score("Γεννήθηκα στα Γιαννιτσά.", "") < 0.5


def test_contentless_response_cannot_hallucinate():
    assert grounding_score("Ναι.", "") == 1.0
    assert grounding_score("Ok", "") == 1.0


def test_grounding_ignores_diacritics():
    assert grounding_score("θυμαμαι", "δεν θυμάμαι τίποτα") == 1.0


# ── Reliability ─────────────────────────────────────────────────


def test_refusal_detected_with_and_without_accents():
    assert refusal_rate(["Δεν θυμάμαι.", "Δεν ξερω."]) == 1.0
    assert refusal_rate(["Ναι, σίγουρα."]) == 0.0


def test_assistant_drift_rate_matches_profile():
    assert assistant_drift_rate(ASSISTANT_LIKE) > 0.5
    assert assistant_drift_rate(GEORGE_LIKE) == 0.0


def test_rates_handle_empty_input():
    assert refusal_rate([]) == 0.0
    assert assistant_drift_rate([]) == 0.0


# ── Fluency ─────────────────────────────────────────────────────


def test_distinct_n_penalises_canned_responses():
    varied = distinct_n(GEORGE_LIKE, 2)
    canned = distinct_n(["Καλά είμαι εσύ"] * 4, 2)
    assert varied > canned


def test_repetition_rate_catches_degenerate_output():
    assert repetition_rate("το ίδιο το ίδιο το ίδιο το ίδιο") > 0.3
    assert repetition_rate("Καλά είμαι, εσύ τι κάνεις σήμερα;") == 0.0


def test_repetition_rate_on_short_text_is_zero():
    assert repetition_rate("Ναι") == 0.0


def test_pairwise_similarity_detects_persona_collapse():
    collapsed = mean_pairwise_similarity(["Καλά είμαι"] * 4)
    varied = mean_pairwise_similarity(GEORGE_LIKE)
    assert collapsed > varied


def test_pairwise_similarity_needs_two_texts():
    assert mean_pairwise_similarity(["μόνο ένα"]) == 0.0


@pytest.mark.parametrize("n", [0, -1])
def test_distinct_n_rejects_invalid_n(n):
    with pytest.raises(ValueError):
        distinct_n(GEORGE_LIKE, n)


# ── Aggregate report ────────────────────────────────────────────


def test_report_contains_all_three_thesis_axes():
    report = aggregate_report(
        GEORGE_LIKE,
        reference_style=style_profile(GEORGE_LIKE),
        contexts=["πλαίσιο"] * len(GEORGE_LIKE),
        latencies_s=[1.2, 2.4, 0.9, 3.1],
    )
    assert "naturalness" in report
    assert "reliability" in report
    assert "accuracy" in report
    assert "performance" in report


def test_report_style_distance_zero_against_self():
    report = aggregate_report(GEORGE_LIKE, reference_style=style_profile(GEORGE_LIKE))
    assert report["naturalness"]["style_distance_to_reference"] == 0.0


def test_report_rejects_misaligned_contexts():
    with pytest.raises(ValueError, match="mismatch"):
        aggregate_report(GEORGE_LIKE, contexts=["only one"])


def test_report_requires_responses():
    with pytest.raises(ValueError):
        aggregate_report([])


def test_optional_sections_omitted_when_no_data():
    report = aggregate_report(GEORGE_LIKE)
    assert "accuracy" not in report
    assert "performance" not in report


def test_latency_percentile_within_observed_range():
    report = aggregate_report(GEORGE_LIKE, latencies_s=[1.0, 2.0, 3.0, 10.0])
    p95 = report["performance"]["p95_latency_s"]
    assert 1.0 <= p95 <= 10.0


def test_markdown_is_flat_and_paste_ready():
    md = report_markdown(aggregate_report(GEORGE_LIKE))
    lines = md.splitlines()
    assert lines[0] == "| Metric | Value |"
    assert all(line.startswith("|") for line in lines)
    assert any("reliability" in line for line in lines)


# ── The grounding metric was measuring the wrong thing ──────────

from jarvis.evaluation.metrics import (  # noqa: E402
    checkable_claims,
    grounding_score,
    unsupported_specifics_rate,
    verbatim_overlap,
)

CTX = (
    "Ερώτηση: θα ερθεις τελικα το σαββατο;\n"
    "Απάντηση Γιώργου: ναι θα ερθω το σαββατο κατα τις οκτω"
)


def test_lexical_overlap_rewards_copying():
    """Documents the defect the replacement exists for.

    A reply that repeats the retrieved context word for word scores best on
    the old metric — and verbatim repetition is exactly the context-bleeding
    failure the pipeline was fixed to remove. Fixing the bleeding made the
    number worse, which is how the metric was found out.
    """
    copied = "ναι θα ερθω το σαββατο κατα τις οκτω"
    paraphrase = "Ναι εννοείται θα έρθω, θα σε πάρω τηλέφωνο όταν ξεκινήσω"
    assert grounding_score(copied, CTX) > grounding_score(paraphrase, CTX)


@pytest.mark.parametrize("reply", [
    "Ναι εννοείται θα έρθω",
    "Ναι εννοείται θα έρθω το Σάββατο, θα σε πάρω τηλέφωνο όταν ξεκινήσω",
    "Καλά, εσύ;",
])
def test_correct_replies_assert_nothing_unsupported(reply):
    """Paraphrase must cost nothing. Only stated specifics are checked."""
    assert unsupported_specifics_rate(reply, CTX) == 0.0


@pytest.mark.parametrize("reply", [
    "Θα πάω Παρίσι με αεροπλάνο",
    "Ναι, 06/10 θα είμαι εκεί για την εγκατάσταση του server",
    "Μίλησα με τον Παπαδόπουλο χθες",
])
def test_invented_specifics_are_caught(reply):
    """A date, a number or a name has to come from somewhere."""
    assert unsupported_specifics_rate(reply, CTX) > 0


def test_small_talk_is_not_penalised_for_lacking_evidence():
    """"Καλά, εσύ;" asserts nothing, so it cannot be unsupported.

    A metric that demanded evidence from small talk would push the twin
    towards sounding like a report — the opposite of its purpose.
    """
    assert unsupported_specifics_rate("Καλά, εσύ;", CTX) == 0.0
    assert checkable_claims("Καλά, εσύ;") == set()


def test_sentence_openers_are_not_mistaken_for_names():
    """Greek replies start with capitalised "Ναι", "Καλά", "Αύριο"."""
    assert checkable_claims("Ναι, καλά είμαι") == set()
    assert "παρισι" in checkable_claims("Θα πάω Παρίσι")


def test_verbatim_copying_is_measured_separately():
    """Grounding is trivially perfect for a system that quotes its source.

    The two numbers are only meaningful read together: low unsupported with
    low verbatim is the target; low with high is parroting.
    """
    copied = "ναι θα ερθω το σαββατο κατα τις οκτω"
    assert verbatim_overlap(copied, CTX) == 1.0
    assert verbatim_overlap("Ναι θα περάσω, γύρω στις οκτώ", CTX) == 0.0


def test_short_replies_are_not_scored_for_copying():
    """Three words in common is coincidence, not quotation."""
    assert verbatim_overlap("ναι θα ερθω", CTX) == 0.0
