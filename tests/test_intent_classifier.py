"""Tests for the intent classifier — routing logic for n8n orchestration."""

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

def test_how_to():
    result = classify_intent("πως να κανω setup σε AWS")
    assert result.intent == Intent.KNOWLEDGE


def test_explain():
    result = classify_intent("εξηγησε μου τι ειναι RAG")
    assert result.intent == Intent.KNOWLEDGE


def test_problem():
    result = classify_intent("εχω προβλημα με το wifi")
    assert result.intent == Intent.KNOWLEDGE


def test_technical():
    result = classify_intent("δεν δουλευει η python")
    assert result.intent == Intent.KNOWLEDGE


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


def test_unknown_defaults_to_knowledge():
    result = classify_intent("αφοδηφδοηφ random text")
    assert result.intent == Intent.KNOWLEDGE
    assert result.confidence == 0.5


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
