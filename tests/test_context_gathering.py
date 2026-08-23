"""Tests για την πολυπηγαία άντληση.

Η αρχική σχεδίαση διάλεγε ΜΙΑ πηγή ανά μήνυμα: το intent έδινε κλάδο και ο
Switch του n8n έκοβε τις υπόλοιπες. Φάνηκε στη διεπαφή, στις 23/08/2026, με
την ερώτηση «τι θα κάνεις αύριο το απόγευμα;» — που χρειάζεται ταυτόχρονα
το ημερολόγιο (τι υπάρχει) και το αρχείο συνομιλιών (τι έχει ειπωθεί) και
πήρε κανένα από τα δύο. Το μοντέλο απάντησε με εκδρομή στο Ναύπλιο, ώρα
αναχώρησης και ώρα επιστροφής.
"""

import asyncio

import pytest

from jarvis.orchestration.api_routes import (
    ContextRequest,
    _PRIORITY,
    _trim,
    gather_context,
)


def _gather(message: str, **kw):
    return asyncio.run(gather_context(ContextRequest(message=message, **kw)))


def test_every_source_is_asked_not_just_one():
    """Το intent καθορίζει τη ΣΕΙΡΑ, όχι το ποιες πηγές ρωτιούνται."""
    result = _gather("τι θα κανεις αυριο το απογευμα;")
    assert result.intent == "schedule"
    names = {s.name for s in result.sources}
    assert {"calendar", "rag", "email"} <= names, names


def test_every_source_reports_a_status():
    """Ποτέ σιωπηλά κενό.

    Το κεφάλαιο 6 καταγράφει τι κόστισε η σιωπηλή αποτυχία: η ανάκτηση
    κατάπινε εξαιρέσεις, επέστρεφε κενό, και το σύστημα έμοιαζε λειτουργικό
    ενώ το μοντέλο επινοούσε ελεύθερα.
    """
    result = _gather("τι μου ελεγες για το σπιτι;")
    assert result.sources
    for s in result.sources:
        assert s.status in {"ok", "empty", "unavailable", "failed", "dropped"}
        if s.status in {"unavailable", "failed"}:
            assert s.detail, f"το {s.name} απέτυχε χωρίς εξήγηση"


def test_sensitive_reaches_no_source_at_all():
    """Μήνυμα για χρήματα δεν φεύγει προς καμία υπηρεσία."""
    result = _gather("μπορεις να μου στειλεις λεφτα στο iban μου;")
    assert result.intent == "sensitive"
    assert result.sources == []
    assert result.context == ""


def test_the_context_budget_is_enforced():
    """Έξι πηγές ασυμπίεστες προσθέτουν ~450 λέξεις σε prompt 963 λέξεων.

    Το κεφάλαιο 7 μετράει τι συμβαίνει εκεί: 715 λέξεις τεκμηρίου έδωσαν
    37λεξη απάντηση σε register με στόχο 6. Το «όλες οι πηγές» δεν σημαίνει
    «όλο το κείμενό τους».
    """
    result = _gather("τι θα κανεις αυριο;", budget=60)
    assert result.total_words <= 60


def test_archived_and_live_are_separated():
    """Χρειάζονται ΑΝΤΙΘΕΤΕΣ οδηγίες και δεν επιτρέπεται να ενωθούν.

    Το αρχείο συνομιλιών πλαισιώνεται με «ΜΗΝ αντιγράφεις ημερομηνίες, ώρες
    ή ραντεβού». Το ημερολόγιο κάτω από την ίδια οδηγία θα ερχόταν στο
    prompt μόνο και μόνο για να αγνοηθεί — άντληση επιτυχής, χρήση
    αποτυχημένη, κανένα σήμα.
    """
    result = _gather("τι θα κανεις αυριο το απογευμα;")
    assert "rag" not in result.live.lower() or not result.live
    if result.live:
        assert "[CALENDAR]" in result.live or "[EMAIL]" in result.live


def test_the_two_frames_say_opposite_things():
    from jarvis.rag.context_builder import frame_context, frame_live_context

    archived = frame_context("Ερώτηση: θα ερθεις; Απάντηση: ναι")
    live = frame_live_context("[CALENDAR] Αύριο 16:00 συνάντηση")

    assert "ΜΗΝ αντιγράφεις" in archived
    assert "ανάφερέ τα όπως είναι" in live
    assert "ΜΗΝ αντιγράφεις" not in live


def test_absent_sources_say_so_rather_than_vanishing():
    """Weather, News και GitHub ζουν στο n8n, όχι στο API.

    Το διάγραμμα της διεπαφής τα δείχνει· ένας κόμβος που φαίνεται αλλά δεν
    υπάρχει είναι χειρότερος από έναν που λέει «δεν είμαι συνδεδεμένος».
    """
    result = _gather("τι καιρο κανει;")
    by_name = {s.name: s for s in result.sources}
    for name in ("weather", "news", "github"):
        assert name in by_name, f"το {name} έλειπε από την αναφορά"
        assert by_name[name].status == "unavailable"
        assert "n8n" in by_name[name].detail


@pytest.mark.parametrize("intent", list(_PRIORITY))
def test_every_intent_has_a_priority_order(intent):
    order = _PRIORITY[intent]
    assert isinstance(order, tuple)
    if intent != "sensitive":
        assert order, f"το {intent} δεν έχει σειρά πηγών"
        assert len(set(order)) == len(order), "διπλή πηγή στη σειρά"


def test_trim_prefers_a_sentence_boundary():
    text = "Πρώτη πρόταση εδώ. Δεύτερη πρόταση εδώ. Τρίτη πρόταση εδώ."
    out = _trim(text, 6)
    assert out.endswith(".")
    assert len(out.split()) <= 6


def test_trim_leaves_short_text_alone():
    assert _trim("μια δυο τρεις", 10) == "μια δυο τρεις"


# ── Το διαγνωστικό μήνυμα ως τεκμήριο ────────────────────────


def test_an_unconfigured_source_contributes_nothing_to_the_prompt():
    """Το «[Calendar not configured]» γραφόταν ΜΕΣΑ στο πεδίο context.

    Το πεδίο ``context`` είναι το κείμενο που μπαίνει στο prompt. Ένα
    διαγνωστικό μήνυμα εκεί μέσα παύει να είναι διαγνωστικό και γίνεται
    πρόταση που το μοντέλο καλείται να λάβει υπόψη — και επειδή το
    ημερολόγιο πλαισιώνεται ως ΤΡΕΧΟΝ ΣΤΟΙΧΕΙΟ, το μήνυμα σφάλματος
    παρουσιαζόταν ως γεγονός που «ισχύει ΤΩΡΑ και είναι αληθινό».

    Δεκάτη εμφάνιση του ίδιου μοτίβου: μια κατάσταση που αναφέρει ``ok``
    ενώ δεν είναι, σε κώδικα γραμμένο την ίδια μέρα ώστε να μην υπάρχουν
    τέτοιες καταστάσεις.
    """
    result = _gather("τι θα κανεις αυριο το απογευμα;")

    assert "not configured" not in result.context
    assert "not configured" not in result.live
    assert "δεν είναι συνδεδεμένο" not in result.live

    by_name = {s.name: s for s in result.sources}
    for name in ("calendar", "email"):
        source = by_name[name]
        if source.status == "unavailable":
            assert source.detail, f"το {name} σιώπησε χωρίς εξήγηση"
            assert source.words == 0


def test_the_explanation_lives_in_detail_not_in_the_prompt():
    """Η εξήγηση πάει στον σχεδιαστή, όχι στο μοντέλο."""
    from jarvis.orchestration.api_routes import (
        CalendarRequest, EmailSearchRequest, calendar_lookup, email_search,
    )

    cal = asyncio.run(calendar_lookup(CalendarRequest(query="αύριο")))
    mail = asyncio.run(email_search(EmailSearchRequest(query="τιμολόγιο")))

    for response in (cal, mail):
        if response.status == "unavailable":
            assert response.context == ""
            assert "διαπιστευτήρια" in response.detail
