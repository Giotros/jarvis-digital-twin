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


# ── Το context του καλούντα δεν ακυρώνει την άντληση ─────────


def test_the_callers_context_does_not_switch_off_the_rest():
    """Ήταν ``else``, και αυτό ακύρωνε ολόκληρη τη σχεδίαση.

    Το frontend καλεί το webhook του n8n, όχι το /generate. Το n8n στέλνει
    δικό του context από τον κλάδο του, οπότε με ``else`` η πολυπηγαία
    άντληση δεν έτρεχε ΠΟΤΕ στη μόνη διαδρομή που βλέπει ο χρήστης — και
    το διάγραμμα συνέχιζε να ανάβει τους ίδιους τρεις κόμβους.

    Το context του καλούντα είναι μία πηγή ανάμεσα σε άλλες.
    """
    import inspect

    from jarvis.orchestration import api_routes

    source = inspect.getsource(api_routes.generate)
    assert "if req.context:" in source
    # Δεν επιτρέπεται να υπάρχει else που να παρακάμπτει την άντληση.
    body = source.split("if req.context:", 1)[1]
    before_gather = body.split("gather_context", 1)[0]
    assert "\n    else:" not in before_gather, (
        "η άντληση εξαρτάται πάλι από το αν ο καλών έστειλε context"
    )


def test_sources_used_is_reported_for_the_ui():
    """Το διάγραμμα πρέπει να δείχνει εκτέλεση, όχι πρόθεση.

    Ο στατικός χάρτης INTENT_NODES άναβε πάντα τους ίδιους κόμβους ανά
    intent. Ένα διάγραμμα που μοιάζει να δείχνει τι έτρεξε ενώ δείχνει τι
    σχεδιάστηκε είναι πειστικό, και η πειστικότητα είναι το πρόβλημα.
    """
    from jarvis.orchestration.api_routes import GenerateResponse

    fields = GenerateResponse.model_fields
    for name in ("sources_used", "sources_empty", "intent"):
        assert name in fields, f"το {name} λείπει από την απόκριση"


# ── «Δεν βλέπω» ≠ «δεν έχω» ──────────────────────────────────


@pytest.mark.parametrize("reply", [
    "Καλημέρα, δεν έχω προγραμματίσει κάτι. Θα είμαι σπίτι αν θες να περάσεις.",
    "δεν έχω τίποτα αύριο",
    "είμαι ελεύθερος όλη μέρα",
    "θα είμαι σπίτι κατά τις 10",
])
def test_a_claim_about_an_unopened_calendar_is_replaced(reply):
    """Το «δεν έχω κάτι» είναι ισχυρισμός ΓΙΑ το ημερολόγιο.

    Δηλώνει ότι είναι άδειο, και το ημερολόγιο δεν ανοίχτηκε ποτέ. Είναι
    ηπιότερο από «καφέ στις 6:30 και μπάσκετ 8-9» — που ήταν η απάντηση
    πριν από κάθε διόρθωση — αλλά ίδιας φύσης: βεβαιότητα χωρίς πηγή.

    Η διαφορά ανάμεσα σε «δεν βλέπω» και «δεν έχω» είναι μία λέξη και δύο
    εντελώς διαφορετικοί ισχυρισμοί. Το μοντέλο δεν την κάνει αξιόπιστα
    ούτε με ρητή οδηγία και παράδειγμα, οπότε υπάρχει και ντετερμινιστικό
    δίχτυ — η ίδια λογική με τις οικείες προσφωνήσεις στο κεφάλαιο 7.
    """
    from jarvis.orchestration.api_routes import (
        ContextResponse, ContextSource, _refuse_ungrounded_schedule,
    )

    gathered = ContextResponse(
        context="", sources=[ContextSource(name="calendar",
                                           status="unavailable",
                                           detail="χωρίς διαπιστευτήρια")],
        intent="schedule", total_words=0,
    )
    out, refused = _refuse_ungrounded_schedule(reply, gathered)
    assert refused is True
    assert out != reply
    assert "δεν ξέρω τι έχω" in out


def test_a_grounded_claim_is_left_alone():
    """Όταν το ημερολόγιο απάντησε, το «δεν έχω κάτι» είναι σωστό."""
    from jarvis.orchestration.api_routes import (
        ContextResponse, ContextSource, _refuse_ungrounded_schedule,
    )

    gathered = ContextResponse(
        context="[CALENDAR] καμία εγγραφή",
        sources=[ContextSource(name="calendar", status="ok", words=3)],
        intent="schedule", total_words=3,
    )
    reply = "δεν έχω τίποτα αύριο"
    assert _refuse_ungrounded_schedule(reply, gathered) == (reply, False)


def test_other_intents_are_untouched():
    from jarvis.orchestration.api_routes import (
        ContextResponse, ContextSource, _refuse_ungrounded_schedule,
    )

    gathered = ContextResponse(
        context="", sources=[ContextSource(name="rag", status="ok", words=5)],
        intent="casual", total_words=5,
    )
    reply = "είμαι ελεύθερος να τα πούμε"
    assert _refuse_ungrounded_schedule(reply, gathered) == (reply, False)


@pytest.mark.parametrize("reply", [
    # Όλα καταγεγραμμένα από ζωντανή εκτέλεση, άτονα όπως γράφει το μοντέλο.
    "Θα ειμαι σπιτι αν θες να περασεις μια βολτα.",
    "Θα ειμαι σπιτι να χαλαρωσω. Θα φυγω απο τη σχολη 2:30-3",
    "Θα πάω σε ένα παιδικό πάρτυ απογευματακι",
    "εχω μαθημα το απογευμα",
])
def test_the_pattern_survives_missing_accents(reply):
    """Το «θα είμαι σπίτι» ήταν στη λίστα και πέρασε ως «Θα ειμαι σπιτι».

    Το corpus είναι μηνύματα από κινητό: κανείς δεν βάζει τόνους, το μοντέλο
    έμαθε να μη βάζει, και ένα μοτίβο γραμμένο με τόνους πιάνει τη μία μορφή
    που δεν εμφανίζεται ποτέ.

    Έκτη φορά στο έργο που μοτίβο και κείμενο γράφονται σε διαφορετική
    ορθογραφία — τελικό σίγμα, τόνος στο «Αθηνάς», τώρα αποτονισμός. Η
    θεραπεία είναι πάντα η ίδια συνάρτηση κανονικοποίησης και για τα δύο.
    """
    from jarvis.orchestration.api_routes import (
        ContextResponse, ContextSource, _refuse_ungrounded_schedule,
    )

    gathered = ContextResponse(
        context="", sources=[ContextSource(name="calendar",
                                           status="unavailable", detail="—")],
        intent="schedule", total_words=0,
    )
    out, refused = _refuse_ungrounded_schedule(reply, gathered)
    assert refused is True, f"ξέφυγε: {reply}"
    assert "δεν ξέρω τι έχω" in out


def test_the_archive_is_dropped_when_the_calendar_is_silent():
    """Το αρχείο απαντά για το παρελθόν· η ερώτηση αφορά το αύριο.

    Με ημερολόγιο, το αρχείο είναι χρήσιμο συμπλήρωμα. Χωρίς, γίνεται το
    μόνο υλικό στο τραπέζι και το μοντέλο απαντά από αυτό:

        «Αυτό που είχαμε πει για την εργασία στο μάθημα είναι να γίνει
         σήμερα»
        «αυτό που λέγαμε εχθές?»

    Και οι δύο καταγράφηκαν ζωντανά, με το πλαίσιο του αρχείου να λέει ήδη
    ρητά «ΜΗΝ αντιγράφεις ημερομηνίες, ώρες ή ραντεβού· αν δεν σχετίζεται,
    αγνόησέ το τελείως». Η οδηγία μειώνει τη συχνότητα· δεν τη μηδενίζει.
    """
    from jarvis.orchestration.api_routes import _EXCLUDE_WITHOUT_PRIMARY

    assert "rag" in _EXCLUDE_WITHOUT_PRIMARY["schedule"]

    result = _gather("τι θα κανεις αυριο το απογευμα;")
    by_name = {s.name: s for s in result.sources}
    calendar_ok = by_name.get("calendar") and by_name["calendar"].status == "ok"
    if not calendar_ok and "rag" in by_name:
        assert by_name["rag"].status in {"dropped", "unavailable", "empty"}
        assert "[RAG]" not in result.context


def test_the_archive_survives_when_the_question_is_recall():
    """Ο αποκλεισμός αφορά ΜΟΝΟ το schedule.

    Στο «τι μου έλεγες για το σπίτι» το αρχείο ΕΙΝΑΙ η σωστή πηγή, και ένας
    κανόνας που το έκοβε παντού θα κατέστρεφε τη μία λειτουργία που το
    δικαιολογεί.
    """
    from jarvis.orchestration.api_routes import _EXCLUDE_WITHOUT_PRIMARY

    for intent in ("knowledge", "memory", "personal", "casual"):
        assert "rag" not in _EXCLUDE_WITHOUT_PRIMARY.get(intent, ())


# ── Η άρνηση που κρατά τη συζήτηση ανοιχτή ───────────────────


@pytest.mark.parametrize("message", [
    "αύριο το πρωί τι ώρα θα ξεκινήσουμε;",
    "τι ώρα θα είναι η συνάντηση;",
    "αυτό που είπαμε ισχύει;",
    "πότε θα βρεθούμε;",
])
def test_a_presupposed_event_gets_a_question_back(message):
    """«Τι ώρα θα ξεκινήσουμε;» δεν ρωτά τι έχει το ημερολόγιο.

    Ρωτά για κάτι συγκεκριμένο που ο συνομιλητής θεωρεί γνωστό. Η άρνηση
    «δεν βλέπω το ημερολόγιό μου» είναι αληθής και άχρηστη: δεν απαντά στο
    ερώτημα και η συζήτηση σταματά.

    Αυτό που κάνει ένας άνθρωπος όταν δεν θυμάται είναι να ρωτήσει «για τι
    πράγμα λες;» — ίδια δηλωμένη άγνοια, ανοιχτή σειρά.
    """
    from jarvis.orchestration.api_routes import _schedule_reply_for

    assert "Σε τι αναφέρεσαι" in _schedule_reply_for(message)


@pytest.mark.parametrize("message", [
    "τι θα κάνεις αύριο το απόγευμα;",
    "είσαι ελεύθερος αύριο;",
    "τι έχεις την Τρίτη;",
])
def test_an_open_question_gets_the_plain_refusal(message):
    """Εδώ δεν υπάρχει γεγονός να ζητηθεί — η ερώτηση είναι ανοιχτή."""
    from jarvis.orchestration.api_routes import _schedule_reply_for

    out = _schedule_reply_for(message)
    assert "Σε τι αναφέρεσαι" not in out
    assert "δεν ξέρω τι έχω" in out


# ── Ισχυρισμός ή πρόταση; ────────────────────────────────────


@pytest.mark.parametrize("assertion", [
    # Ψευδής άρνηση: προσποιείται ότι έγινε ο έλεγχος.
    "Δεν είδα κάτι στο ημερολόγιο για αύριο πρωί.",
    "τώρα μόλις άνοιξα υπολογιστή και δεν ξέρω",
    "κοίταξα το ημερολόγιο και δεν έχω τίποτα.",
    # Σκέτοι ισχυρισμοί.
    "Θα ειμαι σπιτι αν θες να περασεις.",
    "Πιθανόν να πάω για καμιά μπύρα μετά τη δουλειά.",
    "Κανονικά τα μαθήματα δεν έχω.",
])
def test_assertions_about_an_unopened_calendar(assertion):
    """Το «δεν είδα κάτι στο ημερολόγιο» είναι το πιο ύπουλο σχήμα.

    Ακούγεται σαν παραδοχή άγνοιας και είναι ισχυρισμός ότι ο έλεγχος
    ΕΓΙΝΕ και βρήκε άδειο. Η μορφή της πρότασης μοιάζει με ειλικρίνεια, και
    η δικαιολογία από πάνω — «μόλις άνοιξα υπολογιστή» — είναι επινοημένο
    γεγονός: το σύστημα δεν έχει υπολογιστή που ανοίγει.
    """
    from jarvis.orchestration.api_routes import _asserts_ungrounded_schedule

    assert _asserts_ungrounded_schedule(assertion)


@pytest.mark.parametrize("proposal", [
    "να ερθω κατα τις 11?",
    "Να περάσω το απόγευμα;",
    "Θα πάω κατά τις 6, σε βολεύει;",
    "Δεν μπορώ να δω το ημερολόγιό μου τώρα, οπότε δεν ξέρω τι έχω.",
    "Σε τι αναφέρεσαι; Πες μου για ποιο πράγμα λες.",
])
def test_a_proposal_is_not_an_assertion(proposal):
    """«Να έρθω κατά τις 11;» ήταν η καλύτερη απάντηση της ημέρας.

    Ίδιο ρήμα με το «θα πάω στο πάρτυ», εντελώς διαφορετική επιστημολογία:
    προτείνει και ρωτά αντί να δηλώνει. Είναι ακριβώς αυτό που κάνει ένας
    άνθρωπος που δεν ξέρει την ώρα.

    Ένα φίλτρο που έσβηνε και τα δύο θα τιμωρούσε τη σωστή συμπεριφορά —
    το ίδιο σφάλμα με τη μετρική κάλυψης που βαθμολόγησε «χωρίς
    περιεχόμενο» τον ίδιο τον τίτλο της εργασίας.
    """
    from jarvis.orchestration.api_routes import _asserts_ungrounded_schedule

    assert not _asserts_ungrounded_schedule(proposal)
