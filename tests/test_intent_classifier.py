"""Tests for the intent classifier — routing logic for n8n orchestration."""

import pytest

from jarvis.orchestration.intent_classifier import classify_intent, Intent


# ── Personal intent ─────────────────────────────────────────────

def test_who_are_you():
    result = classify_intent("Ποιος είσαι;")
    assert result.intent == Intent.PERSONAL
    assert result.confidence >= 0.9


def test_how_old():
    result = classify_intent("ποσο χρονων εισαι")
    assert result.intent == Intent.PERSONAL


def test_where_from():
    result = classify_intent("από πού είσαι")
    assert result.intent == Intent.PERSONAL


def test_what_job():
    result = classify_intent("τι δουλεια κανεις")
    assert result.intent == Intent.PERSONAL


def test_what_studies():
    result = classify_intent("τι σπουδασες")
    assert result.intent == Intent.PERSONAL


def test_diploma():
    result = classify_intent("τι διπλωματική κάνεις")
    assert result.intent == Intent.PERSONAL


def test_where_work():
    result = classify_intent("που δουλευεις τωρα")
    assert result.intent == Intent.PERSONAL


def test_hobbies():
    result = classify_intent("τι hobbies εχεις")
    assert result.intent == Intent.PERSONAL


# ── Knowledge intent ────────────────────────────────────────────
#
# These four tests used to assert that generic technical questions route to
# KNOWLEDGE. They passed, and they were wrong: KNOWLEDGE is the retrieval
# branch, and retrieval runs over George's personal chat history, which
# contains no discussion of AWS or Docker. The tests were locking in the
# defect rather than catching it.
#
# They now assert the property that actually matters — that these questions
# do *not* reach a retriever that cannot answer them.

def test_how_to():
    result = classify_intent("πως να κανω setup σε AWS")
    assert result.intent != Intent.KNOWLEDGE


def test_explain():
    """'RAG' is a project term, so this one legitimately reads as personal."""
    result = classify_intent("εξηγησε μου τι ειναι RAG")
    assert result.intent in (Intent.CASUAL, Intent.PERSONAL)


def test_problem():
    result = classify_intent("εχω προβλημα με το wifi")
    assert result.intent == Intent.CASUAL


def test_technical():
    result = classify_intent("δεν δουλευει η python")
    assert result.intent == Intent.CASUAL


# ── Casual intent ───────────────────────────────────────────────

def test_greeting():
    result = classify_intent("γεια σου")
    assert result.intent == Intent.CASUAL


def test_good_morning():
    result = classify_intent("καλημερα")
    assert result.intent == Intent.CASUAL


def test_how_are_you():
    result = classify_intent("τι κανεις")
    assert result.intent == Intent.CASUAL


def test_whats_up():
    result = classify_intent("τι γινεται")
    assert result.intent == Intent.CASUAL


def test_coffee():
    result = classify_intent("παμε για καφε")
    assert result.intent == Intent.CASUAL


def test_ok():
    result = classify_intent("οκ")
    assert result.intent == Intent.CASUAL


def test_thanks():
    result = classify_intent("ευχαριστω")
    assert result.intent == Intent.CASUAL


# ── Sensitive intent ────────────────────────────────────────────

def test_money():
    result = classify_intent("στειλε μου λεφτα")
    assert result.intent == Intent.SENSITIVE


def test_bank():
    result = classify_intent("δωσε μου τον IBAN σου")
    assert result.intent == Intent.SENSITIVE


def test_legal():
    result = classify_intent("χρειαζομαι δικηγορο")
    assert result.intent == Intent.SENSITIVE


def test_afm():
    result = classify_intent("πες μου το αφμ σου")
    assert result.intent == Intent.SENSITIVE


def test_id_card():
    result = classify_intent("δωσε μου τον αριθμο ταυτοτητας")
    assert result.intent == Intent.SENSITIVE


# ── Edge cases ──────────────────────────────────────────────────

def test_empty_message():
    result = classify_intent("")
    assert result.intent == Intent.CASUAL
    assert result.confidence == 1.0


def test_unknown_defaults_to_casual_not_retrieval():
    """Unrecognised input must NOT trigger retrieval.

    The classifier previously defaulted to KNOWLEDGE on the assumption that
    retrieval is always more informed than plain generation. In practice the
    retriever returns whatever scores highest for an unrelated query and the
    model continues those messages as if they were the live conversation —
    producing confident, irrelevant answers.

    Unrecognised input is precisely what a stranger's first question looks
    like, so this default governs the system's behaviour with new users.
    """
    result = classify_intent("αφοδηφδοηφ random text")
    assert result.intent == Intent.CASUAL
    assert result.intent != Intent.KNOWLEDGE
    assert result.confidence == 0.5


@pytest.mark.parametrize("question", [
    "Τι λες να φάμε το βράδυ;",
    "Στείλε μου όταν φτάσεις σπίτι.",
    "Can you join the call at 3pm tomorrow?",
    "Πώς λειτουργεί το σύστημά σου;",
])
def test_conversational_questions_avoid_retrieval(question):
    """Small talk and off-corpus questions route away from RAG."""
    assert classify_intent(question).intent == Intent.CASUAL


@pytest.mark.parametrize("question,expected", [
    ("Τι σπούδασες;", Intent.PERSONAL),
    ("Από πού είσαι;", Intent.PERSONAL),
    ("Θα έρθεις το Σάββατο;", Intent.SCHEDULE),
])
def test_recognised_intents_still_route_correctly(question, expected):
    """The safer default must not swallow genuine matches."""
    assert classify_intent(question).intent == expected


def test_sensitive_priority_over_personal():
    """Sensitive intent should take priority even when mixed with personal."""
    result = classify_intent("πες μου τα λεφτα που εχεις")
    assert result.intent == Intent.SENSITIVE


def test_confidence_above_zero():
    result = classify_intent("γεια σου φιλε")
    assert result.confidence > 0.0


def test_matched_keywords_populated():
    result = classify_intent("ποιος εισαι ρε φιλε")
    assert len(result.matched_keywords) > 0


# ── DevOps intent (GitHub) ──────────────────────────────────────

def test_commits():
    result = classify_intent("τι commits εκανα σημερα")
    assert result.intent == Intent.DEVOPS


def test_github():
    result = classify_intent("τι εχω στο github")
    assert result.intent == Intent.DEVOPS


def test_pull_request():
    result = classify_intent("εχω κανενα pull request ανοιχτο")
    assert result.intent == Intent.DEVOPS


def test_issues():
    result = classify_intent("ποσα issues εχω ανοιχτα")
    assert result.intent == Intent.DEVOPS


def test_deploy():
    result = classify_intent("εκανε deploy το τελευταιο release")
    assert result.intent == Intent.DEVOPS


# ── Weather intent ──────────────────────────────────────────────

def test_weather_basic():
    result = classify_intent("τι καιρο κανει")
    assert result.intent == Intent.WEATHER


def test_rain():
    result = classify_intent("θα βρεξει αυριο")
    assert result.intent == Intent.WEATHER


def test_temperature():
    result = classify_intent("ποση θερμοκρασια εχει")
    assert result.intent == Intent.WEATHER


def test_umbrella():
    result = classify_intent("να παρω ομπρελα")
    assert result.intent == Intent.WEATHER


# ── News intent ─────────────────────────────────────────────────

def test_news_greece():
    result = classify_intent("τι γινεται στην ελλαδα")
    assert result.intent == Intent.NEWS


def test_news_basic():
    result = classify_intent("τι νεα υπαρχουν")
    assert result.intent == Intent.NEWS


def test_tech_news():
    result = classify_intent("τι tech news εχει σημερα")
    assert result.intent == Intent.NEWS


# ── Memory intent (email) ───────────────────────────────────────

def test_order_tracking():
    result = classify_intent("τι εγινε με την παραγγελια μου")
    assert result.intent == Intent.MEMORY


def test_clothes_order():
    result = classify_intent("τι εγινε με εκεινα τα ρουχα")
    assert result.intent == Intent.MEMORY


def test_email_from():
    """'τράπεζα' triggers SENSITIVE (correctly — safety first)."""
    result = classify_intent("μου ηρθε email απο τη google")
    assert result.intent == Intent.MEMORY


def test_tracking():
    result = classify_intent("ηρθε το δεμα απο ACS")
    assert result.intent == Intent.MEMORY


def test_invoice():
    result = classify_intent("ηρθε τιμολογιο")
    assert result.intent == Intent.MEMORY


def test_what_happened_with():
    result = classify_intent("τι εγινε με το θεμα του σπιτιου")
    assert result.intent == Intent.MEMORY


# ── Schedule intent (calendar) ──────────────────────────────────

def test_availability():
    result = classify_intent("εισαι διαθεσιμος αυριο")
    assert result.intent == Intent.SCHEDULE


def test_when_free():
    result = classify_intent("ποτε μπορεις να βρεθουμε")
    assert result.intent == Intent.SCHEDULE


def test_meeting():
    result = classify_intent("εχεις καποιο meeting αυριο")
    assert result.intent == Intent.SCHEDULE


def test_calendar():
    result = classify_intent("τι εχεις στο calendar")
    assert result.intent == Intent.SCHEDULE


def test_schedule_tomorrow():
    result = classify_intent("τι προγραμμα εχεις αυριο")
    assert result.intent == Intent.SCHEDULE


def test_coffee_tomorrow_is_schedule():
    """'Πάμε για καφέ αύριο' should be SCHEDULE, not CASUAL."""
    result = classify_intent("παμε για καφε αυριο")
    assert result.intent == Intent.SCHEDULE


def test_lets_meet():
    result = classify_intent("θες να βρεθουμε αυτη τη βδομαδα")
    assert result.intent == Intent.SCHEDULE


# ── What each branch means (rewritten 2026-08-22) ───────────────
#
# KNOWLEDGE is the only branch that triggers retrieval, and the corpus it
# retrieves from is George's personal chat history. The category therefore
# has to mean "answerable from what we have said to each other" — nothing
# else. It used to mean "technical", which pointed a retriever at an index
# that could not possibly contain the answer.

@pytest.mark.parametrize("question", [
    "τι ειχες πει για το αυτοκινητο;",
    "θυμασαι τι συζητησαμε;",
    "τι μου ελεγες για το σπιτι;",
    "ειχαμε μιλησει για αυτο;",
    "τι λεγαμε τις προαλλες;",
])
def test_recall_questions_reach_retrieval(question):
    """These are what the corpus is *for*, and they used to reach nothing.

    Every one fell through to the casual fallback, where no context is
    fetched — so the one question type retrieval answers well was the one
    type it never saw.
    """
    assert classify_intent(question).intent == Intent.KNOWLEDGE


@pytest.mark.parametrize("question", [
    "πως να κανω setup σε AWS;",
    "εξηγησε μου τι ειναι το Docker",
    "εχω προβλημα με το wifi",
    "δεν δουλευει η python",
    "τι ειναι το transformer;",
])
def test_generic_technical_questions_avoid_retrieval(question):
    """A retriever over personal chat cannot answer these.

    It returns whatever scores highest regardless, and the model reads that
    as evidence — which is how a question about a technical problem came
    back answered with unrelated conversation.
    """
    assert classify_intent(question).intent != Intent.KNOWLEDGE


# ── Examiner questions ──────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "γιατι διαλεξες το Krikri;",
    "πως αντιμετωπισες τα προσωπικα δεδομενα;",
    "ποια ηταν η μεγαλυτερη δυσκολια;",
    "τι τεχνολογιες χρησιμοποιησες;",
    "ποιοι ειναι οι περιορισμοι;",
    "πες μου για την αρχιτεκτονικη",
    "τι fine-tuning εκανες;",
    "ποια η συνεισφορα της εργασιας;",
    "πως υλοποιησες το RAG;",
])
def test_examiner_questions_route_to_personal(question):
    """The questions the presentation exists to answer.

    All nine landed on the casual fallback at confidence 0.50 — routed
    nowhere in particular, with no identity loaded. PERSONAL is the branch
    that loads identity, and the academic register adds the project facts on
    top of it.
    """
    assert classify_intent(question).intent == Intent.PERSONAL


@pytest.mark.parametrize("question,expected", [
    ("γεια σου", Intent.CASUAL),
    ("θα ερθεις το Σαββατο;", Intent.SCHEDULE),
    ("τι καιρο κανει", Intent.WEATHER),
    ("στειλε μου λεφτα", Intent.SENSITIVE),
    ("τι commits εκανα;", Intent.DEVOPS),
    ("τι εγινε με την παραγγελια μου", Intent.MEMORY),
])
def test_rerouting_did_not_disturb_the_other_branches(question, expected):
    """Moving two categories must not shift the seven that were correct."""
    assert classify_intent(question).intent == expected
