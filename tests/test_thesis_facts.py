"""Tests for grounding technical answers in the real project.

The failures below are transcribed from a live run on 2026-08-22, when the
twin was asked "με τι τεχνολογίες δούλεψες φέτος;" in the academic register.
Each string is something it actually said.
"""

import pytest

from jarvis.inference.thesis_facts import check_technical_claims, load_thesis_facts


# ── Detecting confabulation ─────────────────────────────────────

@pytest.mark.parametrize("text", [
    "έτρεχα το μοντέλο μου σε Krikri-12B",
    "QLoRA fine-tuning πάνω στο BERTweet ως βάση",
    "ανάλυση συναισθημάτων στα ελληνικά",
    "έκανα projects πάνω σε Computer Vision χρησιμοποιώντας PyTorch και OpenCV",
    "Για τη βάση δεδομένων χρησιμοποιήσαμε MongoDB",
    "Python με Django στο backend",
    "έμαθα το TensorFlow 2.x",
    "React μαζί με Redux για διαχείριση καταστάσεων",
])
def test_observed_confabulations_are_caught(text):
    """Every one of these was said by the deployed model.

    They are fluent, specific and false — the combination that makes them
    dangerous. A vague wrong answer invites a follow-up question; a detailed
    one gets believed.
    """
    assert check_technical_claims(text), f"missed: {text}"


def test_correct_answer_is_not_flagged():
    """The check must not punish the answer we actually want."""
    good = (
        "Χρησιμοποίησα το Krikri-8B ως βασικό μοντέλο, με QLoRA fine-tuning "
        "σε 4-bit, και Ray για την κατανεμημένη εκπαίδευση. Η ανάκτηση γίνεται "
        "με υβριδική αναζήτηση BM25 και dense embeddings σε ChromaDB."
    )
    assert check_technical_claims(good) == []


@pytest.mark.parametrize("text", [
    "Το Mistral-7B το εξέτασα αλλά το tokenization των ελληνικών ήταν χειρότερο.",
    "Δεν χρησιμοποίησα GPT γιατί τα δεδομένα είναι προσωπικές συνομιλίες.",
    "Σε σύγκριση με το Mistral, το Krikri έχει καλύτερο tokenizer.",
])
def test_mentioning_a_rejected_alternative_is_not_a_hallucination(text):
    """The thesis compares against Mistral and GPT and explains the rejection.

    Flagging the bare name would mark the correct, well-argued answer as
    false — the check would be actively harmful at exactly the moment the
    answer is at its best.
    """
    assert check_technical_claims(text) == []


def test_empty_text_yields_no_issues():
    assert check_technical_claims("") == []


def test_issue_messages_carry_the_correction():
    """Reporting "something is wrong" is not enough to act on."""
    issues = check_technical_claims("έτρεξα το Krikri-12B")
    assert issues and any("8B" in i for i in issues)


# ── Loading the facts ───────────────────────────────────────────

def test_facts_load_and_contain_the_key_numbers():
    facts = load_thesis_facts(force_reload=True)
    if not facts:
        pytest.skip("config/thesis_facts.yaml not present in this checkout")
    for expected in ("Krikri-8B", "QLoRA", "Ray", "13.289", "8,1%"):
        assert expected in facts


def test_facts_instruct_refusal_over_invention():
    """Grounding only helps if the model is told what to do with a gap."""
    facts = load_thesis_facts(force_reload=True)
    if not facts:
        pytest.skip("config/thesis_facts.yaml not present in this checkout")
    assert "μην το συμπληρώσεις" in facts.lower()


def test_academic_register_receives_the_facts():
    """The grounding is wired in, not merely available."""
    from jarvis.orchestration.persona import build_system_prompt

    academic, _ = build_system_prompt("Άννα", "καθηγήτρια")
    casual, _ = build_system_prompt("Άννα", "φίλη")

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    assert "Krikri-8B" in academic
    # A friend asking gets the twin's voice, not a recitation of the method.
    assert "Krikri-8B" not in casual


@pytest.mark.parametrize("text", [
    "Δεν χρησιμοποιήσαμε MongoDB, η αποθήκευση είναι σε Delta Lake.",
    "Δεν έκανα computer vision σε αυτή την εργασία.",
    "Χωρίς TensorFlow — όλα με PyTorch.",
])
def test_denying_a_technology_is_not_flagged(text):
    """Negation applies to every pattern, not just the model names."""
    assert check_technical_claims(text) == []


def test_negation_does_not_reach_across_a_sentence_boundary():
    """A denial in one sentence must not excuse a claim in the next."""
    text = "Δεν χρησιμοποίησα GPT. Η βάση δεδομένων μου είναι MongoDB."
    assert check_technical_claims(text), "MongoDB claim should still be caught"


# ── Word-boundary regressions (observed 2026-08-22, second run) ─

@pytest.mark.parametrize("text", [
    "PyTorch/Flax/TensorFlow2/Keras ως βιβλιοθήκες ML/AI",
    "έμαθα TensorFlow 2.x",
    "Docker/RayHub για containers",
    "n8n/n8x για αυτοματοποίηση",
    "machine learning με TensorFlow Keras",
])
def test_version_suffixes_and_glued_names_are_caught(text):
    """`\\btensorflow\\b` does not match "TensorFlow2".

    A digit is a word character, so there is no boundary after the name. The
    claim passed a check written specifically to catch it. Version suffixes
    and slash-joined lists are the normal shape of this text.
    """
    assert check_technical_claims(text), f"missed: {text}"


@pytest.mark.parametrize("text,fragment", [
    ("Docker/RayHub για deployment", "RayHub"),
    ("n8n/n8x για ροές", "n8x"),
])
def test_invented_tool_names_are_caught(text, fragment):
    """Neither product exists.

    The model built them by analogy from names that do — the same mechanism
    behind "Krikri-12B", and invisible to anyone who does not already know
    the ecosystem.
    """
    issues = check_technical_claims(text)
    assert issues and any(fragment in i for i in issues)


def test_professional_register_is_grounded_too():
    """Grounding only the academic register was not enough.

    Asked the same question as a colleague, the twin described a different
    thesis: "πρόβλεψη τιμών ενέργειας μέσω deep learning", with Django and
    PostgreSQL. Whoever asks a technical question gets the same technical
    truth; only the tone differs.
    """
    from jarvis.orchestration.persona import build_system_prompt

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    professional, _ = build_system_prompt("Παναγιώτης", "συνάδελφος")
    assert "Krikri-8B" in professional


# ── Third round: produced *with* the facts already in the prompt ─

@pytest.mark.parametrize("text", [
    "Ollama για deployment, Google Cloud Platform (GCP) και n8n για orchestration",
    "QLoRA για fine-tuning σε edge συσκευές",
    "έκανα μια εφαρμογή blockchain στο solidity",
    "PostgreSQL ως βάση δεδομένων",
])
def test_confabulation_survives_grounding(text):
    """These appeared *after* the facts were injected into the prompt.

    That is what makes them worth recording. Grounding reduces invention; it
    does not end it. The model paraphrases what it was given and fills the
    gaps between the facts with plausible neighbours — GCP next to Ollama,
    "edge" next to 4-bit quantisation. A blacklist can only catch what has
    already been observed, which is why the check is a safety net and not a
    guarantee.
    """
    assert check_technical_claims(text), f"missed: {text}"


def test_grounded_answer_from_the_live_run_passes():
    """The best observed academic answer, kept as the target to hold."""
    good = (
        "Στην διπλωματική χρησιμοποίησα Llama-Krikri-8B-Instruct ως βασικό "
        "μοντέλο, QLoRA fine-tuning στο 4-bit, Ray για κατανεμημένη εκπαίδευση "
        "κι ένα σύστημα διαχείρισης δεδομένων με Delta Lake στη Databricks."
    )
    assert check_technical_claims(good) == []


# ── Missed by the checker during the ablation run (2026-08-22) ──

@pytest.mark.parametrize("text", [
    # The verb was 75 characters from the claim; the window was 40.
    "Το επέλεξα επειδή είναι ελληνόγλωσσο και έχει tokenizer "
    "προσαρμοσμένο στα ελληνικά (Mistral-7B) που χειρίζεται καλύτερα",
    # The acronym was expanded wrongly, twice, into two different fields.
    "Το RAG (Retriever-Adapter-Generator) κομμάτι συνδυάζει τρία στοιχεία",
    "Το RAG κομμάτι (Retrovirus Activation Gene) είναι τμήμα του DNA",
])
def test_claims_missed_during_the_ablation_are_now_caught(text):
    """Both slipped past while the comparison reported "0 αντιφάσεις".

    The first because "επέλεξα" was not among the usage verbs and the claim
    sat far from it; the second because the pattern expected the parenthesis
    to follow "RAG" immediately, and the model wrote "RAG κομμάτι (...)".
    A wrong expansion of the acronym is the kind of error an examiner
    notices immediately and a regex notices never.
    """
    assert check_technical_claims(text), f"missed: {text}"


@pytest.mark.parametrize("text", [
    "Το Mistral-7B το εξέτασα αλλά το tokenization των ελληνικών ήταν χειρότερο.",
    "Σε σύγκριση με το Mistral, το Krikri έχει καλύτερο tokenizer.",
    "Το Retrieval Augmented Generation (RAG) είναι υβριδική προσέγγιση.",
    "Το RAG (Retrieval-Augmented Generation) συνδυάζει ανάκτηση και παραγωγή.",
    "Το επέλεξα επειδή είναι ελληνόγλωσσο, με tokenizer προσαρμοσμένο στα ελληνικά.",
])
def test_widening_the_window_did_not_create_false_positives(text):
    """The verb window went from 40 to 90 characters to catch the miss above.

    Widening a pattern is how a checker starts flagging correct answers, so
    the correct forms are pinned here explicitly.
    """
    assert check_technical_claims(text) == []
