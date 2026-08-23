"""Tests for grounding technical answers in the real project.

The failures below are transcribed from a live run on 2026-08-22, when the
twin was asked "με τι τεχνολογίες δούλεψες φέτος;" in the academic register.
Each string is something it actually said.
"""

import asyncio

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

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    assert "Krikri-8B" in academic


def test_a_technical_question_is_grounded_in_every_register():
    """Truth is a property of the question, not of the listener.

    This test replaces one that asserted the opposite — that a friend never
    receives the facts. That assertion passed, and it locked in the bug:
    asked "με τι τεχνολογίες δούλεψες φέτος" in the close register, the twin
    answered with OpenAI GPT-4 and Django. Both are false statements about
    this repository, and casual phrasing does not make them less so.
    """
    from jarvis.orchestration.persona import build_system_prompt

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    for role in ("φίλη", "συνάδελφος", "καθηγήτρια", ""):
        prompt, _ = build_system_prompt(
            "Άννα", role, message="με τι τεχνολογίες δούλεψες φέτος;"
        )
        assert "Krikri-8B" in prompt, f"{role or 'άγνωστος'} δεν πήρε τα στοιχεία"


def test_small_talk_is_not_weighed_down_by_the_facts():
    """The counterweight: 580 words of YAML do not belong in "τι κάνεις".

    Grounding everything unconditionally would be the easy fix and the wrong
    one — it makes every casual reply read like a specification sheet, which
    is the failure mode the register mechanism exists to prevent.
    """
    from jarvis.orchestration.persona import build_system_prompt

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    prompt, _ = build_system_prompt("Άννα", "φίλη", message="τι κάνεις ρε συ;")
    assert "Krikri-8B" not in prompt


def test_technical_question_patterns_survive_folding():
    """Final sigma has silently disabled a pattern table five times here.

    ``casefold`` maps ς to σ, so a pattern written with the natural spelling
    matches nothing in folded text — and a regex that never fires is
    indistinguishable from a question nobody asks. The patterns are folded
    through the same function as the haystack; this checks that they are.
    """
    from jarvis.orchestration.persona import is_technical_question

    # Each of these ends a word in final sigma somewhere.
    assert is_technical_question("τι έκανες φέτος")
    assert is_technical_question("γιατί διάλεξες το Krikri")
    assert is_technical_question("με τι τεχνολογίες δούλεψες")


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


# ── Η επιβολή, όχι μόνο η ανίχνευση ──────────────────────────


def test_generate_does_not_serve_a_confabulated_answer():
    """The detector must run in the path that answers users.

    ``check_technical_claims`` was written, tested, and wired only into
    ``/fact-check`` — an endpoint no part of the workflow calls. It was
    correct and unreachable for as long as it existed, so every claim it
    could recognise was served anyway. The diagnostic script produced six
    invented answers out of six attempts; the detector flagged all six the
    moment it was pointed at them by hand.

    This test pins the wiring, which is the part that was missing.
    """
    from jarvis.orchestration.api_routes import _reject_confabulation
    from jarvis.orchestration import persona

    bad = "Χρησιμοποίησα Kubernetes για orchestration και Google Cloud."
    good = "Χρησιμοποίησα Ray για κατανεμημένη εκπαίδευση και Ollama τοπικά."

    class _Client:
        def __init__(self, reply):
            self._reply = reply
            self.calls = 0

        async def post(self, url, json):  # noqa: A002 — mirrors httpx
            self.calls += 1
            return _Response(self._reply)

    class _Response:
        def __init__(self, reply):
            self._reply = reply

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": self._reply}}

    client = _Client(good)
    reply, retried = asyncio.run(_reject_confabulation(
        client=client, url="http://x", model="m", messages=[],
        reply=bad, register=persona.ACADEMIC, options={},
    ))
    assert retried is True
    assert "Kubernetes" not in reply
    assert client.calls == 1, "ακριβώς μία επανάληψη, όχι βρόχος"


def test_a_clean_answer_is_not_regenerated():
    """The retry costs a second inference; it must not fire on good output."""
    from jarvis.orchestration.api_routes import _reject_confabulation
    from jarvis.orchestration import persona

    clean = "Δούλεψα με Ray, PyTorch και Ollama."

    class _Client:
        calls = 0

        async def post(self, url, json):  # noqa: A002
            raise AssertionError("δεν έπρεπε να ξαναρωτήσει")

    reply, retried = asyncio.run(_reject_confabulation(
        client=_Client(), url="http://x", model="m", messages=[],
        reply=clean, register=persona.ACADEMIC, options={},
    ))
    assert retried is False
    assert reply == clean


def test_two_bad_drafts_produce_a_refusal_not_a_third_roll():
    """When grounding and correction both fail, say nothing specific.

    An examiner hearing "δεν το θυμάμαι ακριβώς" loses a point. An examiner
    hearing about a Kubernetes cluster that does not exist loses the thesis.
    """
    from jarvis.orchestration.api_routes import _reject_confabulation
    from jarvis.orchestration import persona

    bad = "Έτρεξα σε Kubernetes πάνω σε Google Cloud."

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": bad}}

    class _Client:
        async def post(self, url, json):  # noqa: A002
            return _Response()

    reply, retried = asyncio.run(_reject_confabulation(
        client=_Client(), url="http://x", model="m", messages=[],
        reply=bad, register=persona.ACADEMIC, options={},
    ))
    assert retried is True
    assert check_technical_claims(reply) == []
    assert "Kubernetes" not in reply


# ── Allowlist: ό,τι δεν είναι στα facts ──────────────────────
#
# Καταγράφηκαν στις 22/08/2026, ΜΕΤΑ τη σύνδεση του denylist στη ροή. Το
# διαγνωστικό ανέφερε «Τα registers λειτουργούν end-to-end» και οι απαντήσεις
# περιείχαν τα παρακάτω.


@pytest.mark.parametrize("text,expected", [
    ("εκεί χρησιμοποιώ Rust σε συνδυασμό με WebAssembly μέσω του actix-web",
     {"Rust", "WebAssembly", "actix-web"}),
    ("Python και Django για το web development", {"Django"}),
    ("chatbot σε Node.js χρησιμοποιώντας OpenAI GPT-4", {"Node.js", "GPT-4"}),
    ("είχα δουλέψει React, Next.js και AWS Lambda", {"Next.js", "AWS", "Lambda"}),
    ("δούλεψα λίγο με blockchain", {"blockchain"}),
    ("Kubernetes για orchestration", {"Kubernetes"}),
])
def test_inventions_the_denylist_never_saw(text, expected):
    """Every one of these passed ``check_technical_claims`` as clean.

    Not because that function is wrong — because it is a denylist, and a
    denylist cannot recognise what has not happened yet. This is the surname
    problem from chapter 4 in a different costume: an open class, where you
    know what you caught and never what you missed.
    """
    from jarvis.inference.thesis_facts import unsupported_technologies

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    assert set(unsupported_technologies(text)) >= expected


@pytest.mark.parametrize("text", [
    "PyTorch για τα μοντέλα, Ray κατανεμημένα, FastAPI, React, Docker.",
    "Το βασικό μοντέλο ήταν Llama-Krikri-8B-Instruct με QLoRA 4-bit.",
    "Ollama ως runtime, n8n για ενορχήστρωση, ChromaDB για την ανάκτηση.",
    "Η εκπαίδευση έγινε σε Google Colab με Databricks για το pipeline.",
    "Το Mistral-7B το εξέτασα αλλά το tokenization ήταν χειρότερο.",
    "Python 3.11 με uvicorn.",
])
def test_the_real_stack_is_not_flagged(text):
    """An allowlist that flags the correct answer is worse than none."""
    from jarvis.inference.thesis_facts import unsupported_technologies

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    assert unsupported_technologies(text) == []


def test_the_not_used_field_does_not_excuse_the_tools_it_forbids():
    """The field written to forbid them was what made them supported.

    ``stack.not_used`` lists Kubernetes, Rust, Django and the rest by name.
    The allowlist matches names against the facts text, so adding that field
    made every forbidden tool "mentioned in the facts" — and the check
    reported clean on the exact reply that had prompted the field. Same shape
    as everything else here: a tightening that loosened, reporting success.
    """
    from jarvis.inference.thesis_facts import unsupported_technologies

    facts = load_thesis_facts(force_reload=True)
    if not facts:
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    # The names really are in the prompt block — that part is intentional,
    # the model benefits from an explicit list of things not to say.
    assert "Kubernetes" in facts
    # …and they must still be flagged when a reply claims them.
    assert unsupported_technologies("Έτρεξα σε Kubernetes") == ["Kubernetes"]


def test_version_suffixes_do_not_hide_a_name():
    """"GPT-4" must resolve to "gpt"; "llama.cpp" must not resolve to "llama"."""
    from jarvis.inference.thesis_facts import _canonical_tech

    assert _canonical_tech("GPT-4") == "gpt"
    assert _canonical_tech("TensorFlow2") == "tensorflow"
    assert _canonical_tech("Mistral-7B") == "mistral"
    # These carry digits or dots as part of the name itself.
    assert _canonical_tech("llama.cpp") == "llama.cpp"
    assert _canonical_tech("n8n") == "n8n"
    assert _canonical_tech("next.js") == "next.js"
    # Ordinary words stay out.
    assert _canonical_tech("orchestration") == ""
    assert _canonical_tech("distributed") == ""


def test_no_facts_means_no_verdict():
    """Without a source of truth every name is unsupported — say nothing."""
    from jarvis.inference.thesis_facts import unsupported_technologies

    assert unsupported_technologies("Rust και Kubernetes", facts="") == []


def test_the_facts_file_lists_the_tools_actually_used():
    """The allowlist found the source of truth itself to be incomplete.

    PyTorch, Docker, FastAPI and Colab are used and were absent, while the
    closing instruction says "μην αναφέρεις εργαλείο που δεν γράφεται εδώ".
    The gap asked the model not to tell the truth, and vagueness is the space
    confabulation fills.
    """
    facts = load_thesis_facts(force_reload=True)
    if not facts:
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    lowered = facts.lower()
    for tool in ("pytorch", "docker", "fastapi", "colab", "react", "n8n",
                 "ollama", "ray", "chromadb", "databricks"):
        assert tool in lowered, f"το {tool} χρησιμοποιείται και δεν αναφέρεται"


# ── Τέταρτη κατηγορία: ισχυρισμοί ικανοτήτων ────────────────
#
# Καταγράφηκαν 22/08/2026, στην εκτέλεση όπου η *στοίβα* ήταν επιτέλους
# σωστή. Και τα δύο προηγούμενα φίλτρα είναι λεξικά ως προς ονόματα
# εργαλείων, οπότε μια απάντηση με τέλεια απαρίθμηση τεχνολογιών και ψευδή
# ικανότητα δίπλα τους περνά και από τα δύο καθαρή.


@pytest.mark.parametrize("text", [
    "εκπαιδεύεται μέσω machine learning όταν μαθαίνει νέα πράγματα από τις "
    "αλληλεπιδράσεις του",
    "παίρνει αποφάσεις σαν άνθρωπος αλλά μαθαίνει συνεχώς",
    "κάνει continual learning από κάθε συνομιλία",
    "βελτιώνεται κάθε φορά που του μιλάς",
    "Llama-Krikri-8B-Instruct και Mistral-7B που είναι τα πιο ισχυρά "
    "ελληνόγλωσσα μοντέλα",
    "τα μοντέλα μας: Krikri και Mistral-7B",
])
def test_capability_claims_are_caught(text):
    """"Μαθαίνει από τις αλληλεπιδράσεις" is the first thing an examiner asks.

    The adapter is a frozen GGUF; retraining is a manual step that has not
    happened since the corpus was cleaned. Claiming otherwise describes a
    system that would need a completely different privacy analysis.
    """
    assert check_technical_claims(text), f"missed: {text}"


@pytest.mark.parametrize("text", [
    "Ο adapter είναι στατικός — δεν μαθαίνει από τις αλληλεπιδράσεις.",
    "Το Mistral-7B το εξέτασα αλλά το tokenization των ελληνικών ήταν χειρότερο.",
    "Το μοντέλο εκπαιδεύτηκε σε 13.289 ζεύγη από προσωπικές συνομιλίες.",
    "Δεν υπάρχει online learning — η επανεκπαίδευση είναι χειροκίνητη.",
    "Python 3.11, PyTorch, Ray, FastAPI, React/Vite, Docker.",
])
def test_correct_statements_about_capability_pass(text):
    """The denial is the right answer and must not be punished for it."""
    assert check_technical_claims(text) == []


# ── Το μέγεθος της θεμελίωσης ταιριάζει στο μήκος της απάντησης ──


def test_short_registers_get_short_grounding():
    """715 words of evidence override any instruction about length.

    Given the full block, the close register — measured target six words —
    produced a correct 37-word specification sheet. Correct, and in nobody's
    voice. The model reproduces the shape of what it is given.
    """
    from jarvis.orchestration.persona import build_system_prompt

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    q = "με τι τεχνολογίες δούλεψες φέτος;"
    close, _ = build_system_prompt("Άννα", "φίλη", message=q)
    academic, _ = build_system_prompt("Άννα", "καθηγήτρια", message=q)

    # Both grounded…
    assert "Krikri" in close
    assert "Krikri" in academic
    # …but the close prompt must stay small enough not to set the register.
    assert len(close.split()) < len(academic.split()) / 3


def test_the_brief_is_true_where_it_is_short():
    """Shortening may drop facts; it may not introduce wrong ones."""
    from jarvis.inference.thesis_facts import load_thesis_facts_brief

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    brief = load_thesis_facts_brief()
    assert check_technical_claims(brief) == []
    for essential in ("Krikri-8B", "QLoRA", "Ray", "Ollama"):
        assert essential in brief, f"το {essential} λείπει από τη σύνοψη"
    # The two claims an examiner reaches for first.
    assert "τοπικά" in brief
    assert "μαθαίνει" in brief


def test_a_missing_brief_falls_back_to_the_full_block():
    """Being long beats being ungrounded."""
    from jarvis.inference import thesis_facts as tf

    load_thesis_facts(force_reload=True)
    saved = tf._brief_cache
    try:
        tf._brief_cache = ""
        assert "Krikri" in tf.load_thesis_facts_brief()
    finally:
        tf._brief_cache = saved


def test_a_denial_in_the_facts_does_not_support_the_claim():
    """Third occurrence of the same shape, third mechanism.

    The ``brief`` field says «Δεν χρησιμοποιώ ChatGPT ούτε cloud» — written
    to deny GPT. Under substring matching "gpt" is inside "chatgpt", so the
    denial made every GPT claim supported, exactly as ``not_used`` had
    excused Kubernetes and Rust. Matching is now on whole names.
    """
    from jarvis.inference.thesis_facts import unsupported_technologies

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    # ⊇ rather than ==: the vocabulary grows as new inventions are observed,
    # and a test pinned to an exact list would fail every time it does — the
    # wrong signal, since growth here is the point.
    assert "GPT-4" in unsupported_technologies("chatbot με OpenAI GPT-4")
    # …while the name that really is part of the stack still resolves inside
    # a longer product string.
    assert unsupported_technologies("το Krikri ως βάση") == []


def test_pseudo_precision_is_caught():
    """"75% του κώδικα" was never measured. Precision reads as evidence.

    Observed beside otherwise correct facts. It is the most dangerous shape
    of invention: an examiner hearing a figure in parentheses assumes
    somebody counted. The percentages that *were* measured live in the facts
    file and must keep passing.
    """
    assert check_technical_claims("Python 3.11 κυρίως (75% του κώδικα)")
    assert check_technical_claims("το 90% του project είναι σε Python")
    # Μετρημένα — δεν σημαίνονται.
    for measured in (
        "ο κανόνας κεφαλαίου έχανε το 8,1% των εγγραφών",
        "η κάλυψη είναι 43% έναντι 66% του βασικού",
        "589 εμφανίσεις επωνύμων σε 551 εγγραφές (4,1%)",
    ):
        assert check_technical_claims(measured) == [], measured


def test_integrations_are_distinguished_from_inventions():
    """"συνδέθηκε με slack, github και ollama" — one of these is not real."""
    from jarvis.inference.thesis_facts import unsupported_technologies

    if not load_thesis_facts(force_reload=True):
        pytest.skip("config/thesis_facts.yaml not present in this checkout")

    assert unsupported_technologies("συνδέθηκε με slack και github") == ["slack"]
    assert unsupported_technologies("για devops ερωτήσεις μιλάει με GitHub") == []


# ── Πέμπτη και έκτη κατηγορία ────────────────────────────────
#
# Καταγράφηκαν 22/08/2026, στην εκτέλεση όπου η στοίβα, οι ικανότητες και τα
# ποσοστά ήταν όλα σωστά. Κάθε γύρος διορθώσεων αποκάλυψε την επόμενη
# κατηγορία — ο ίδιος μηχανισμός με το «δεν γνωρίζεις την ανάκλησή σου» του
# κεφαλαίου 4, εφαρμοσμένος στους ελέγχους αντί στα δεδομένα.


@pytest.mark.parametrize("text,acronym", [
    ("PEFT (PyTorch Elastic Framework) για κατανεμημένη εκπαίδευση", "PEFT"),
    ("TRL για transfer learning/reasoning tasks", "TRL"),
    ("RAG (Retrovirus Activation Gene) — τμήμα του DNA", "RAG"),
    ("LoRA = Layer of Recurrent Attention", "LoRA"),
])
def test_wrong_acronym_expansions_are_caught(text, acronym):
    """A real tool with an invented meaning passes every name-based check.

    "PEFT" is in the vocabulary and in the facts, so the allowlist is
    satisfied; nothing in the denylist mentions it, so that is satisfied
    too. The falsehood is in the gloss, and the more confident the gloss the
    more likely a listener takes it on trust.
    """
    from jarvis.inference.thesis_facts import check_acronym_expansions

    issues = check_acronym_expansions(text)
    assert issues and acronym in issues[0]


@pytest.mark.parametrize("text", [
    "PEFT (Parameter-Efficient Fine-Tuning)",
    "TRL = Transformer Reinforcement Learning",
    "QLoRA (Quantized Low-Rank Adaptation) σε 4-bit",
    "Το RAG (Retrieval-Augmented Generation) συνδυάζει ανάκτηση και παραγωγή.",
    # Naming without glossing is correct and must not be punished.
    "χρησιμοποίησα PEFT και TRL στην εκπαίδευση",
    "το BM25 μαζί με dense embeddings",
])
def test_correct_or_absent_expansions_pass(text):
    from jarvis.inference.thesis_facts import check_acronym_expansions

    assert check_acronym_expansions(text) == []


def test_corrupted_tool_names_are_caught():
    """"ChromeDB" is the right tool with the wrong letters.

    Invisible to the allowlist — it is not in the vocabulary, so there is
    nothing to look up — and not an invented tool either. In speech it slips
    past; in a written thesis it is simply wrong.
    """
    from jarvis.inference.thesis_facts import check_corrupted_names

    issues = check_corrupted_names("ChromeDB ως data store")
    assert issues and "chromadb" in issues[0].lower()


@pytest.mark.parametrize("text", [
    "ChromaDB ως data store",
    "Ray, PyTorch, Databricks, Ollama, FastAPI",
    "Hugging Face transformers και bitsandbytes",
    "το frontend είναι React με Vite",
    "Απάντησα στον καθηγητή για την κατανεμημένη εκπαίδευση.",
])
def test_correct_names_are_not_called_corruptions(text):
    """A near-miss check that flags real names is worse than none."""
    from jarvis.inference.thesis_facts import check_corrupted_names

    assert check_corrupted_names(text) == []


# ── Έβδομη κατηγορία: κλίμακα και πλαίσιο ────────────────────


@pytest.mark.parametrize("text", [
    "Ray για κατανεμημένη εκπαίδευση σε GPU clusters",
    "n8n workflows που τρέχουν τοπικά ή στον server μας μέσω Docker",
    "είναι η ραχοκοκαλιά του συστήματος (και τρέχει 24/7)",
    "ασχολήθηκα με Ray, κυρίως στο κομμάτι των reinforcement learning agents",
    "έκανα ένα chatbot πού απαντούσε σε ερωτήσεις για τα προϊόντα της εταιρείας",
    "εκπαίδευσα σε multi-GPU setup",
])
def test_scale_and_context_claims_are_caught(text):
    """The names were all right; the infrastructure around them was invented.

    Harder than the previous six because there is no keyword to look up: the
    same words ("server", "cluster") are correct in a different sentence.
    One GPU is not a cluster, and a laptop is not a server.
    """
    assert check_technical_claims(text), f"missed: {text}"


@pytest.mark.parametrize("text", [
    "Ray για κατανεμημένη εκπαίδευση με data parallelism",
    "Το QLoRA επιλέχθηκε για να χωρέσει η εκπαίδευση σε ένα GPU.",
    "το n8n τρέχει τοπικά σε Docker",
    "Ollama ως runtime, τρέχει στο Mac μου",
    "δούλευα στην εταιρεία ως manager",
])
def test_correct_scale_statements_pass(text):
    assert check_technical_claims(text) == []


# ── Όγδοη κατηγορία: βρέθηκε από το ίδιο το εργαλείο μέτρησης ─
#
# Ο μετρητής ανέφερε «0,0% επινόηση» και ταξινόμησε τις παρακάτω ως «χωρίς
# περιεχόμενο». Δύο σφάλματα μαζί: κανένα μοτίβο δεν τις έπιανε, και οι
# άξονες «δεν απάντησε» / «είπε ψέματα» ήταν συγχωνευμένοι σε έναν, οπότε
# ο πρώτος έκρυβε τον δεύτερο.


@pytest.mark.parametrize("text", [
    "Εχω φτιάξει ένα σύστημα που χρησιμοποιεί το Ray και τρέχει σε 3 "
    "διαφορετικούς servers.",
    "Ο server Α κάνει την συλλογή δεδομένων από διάφορες πηγές (π.χ. twitter)",
    "τα δεδομένα από την βάση της Αθηνάς",
    "το dataset ήρθε από την Αθηνά",
    "Το έτρεξα για 3-4 μέρες μέχρι που τελείωσε",
])
def test_the_measurement_tool_found_these(text):
    assert check_technical_claims(text), f"missed: {text}"


@pytest.mark.parametrize("text", [
    "Το Krikri-8B είναι από το ΙΕΛ / Αθηνά Ε.Κ.",
    "Το βασικό μοντέλο δημοσιεύτηκε από το Αθηνά Ερευνητικό Κέντρο.",
    "τα δεδομένα είναι 13.289 ζεύγη από προσωπικές συνομιλίες",
    "η εκπαίδευση σταμάτησε στο checkpoint 650",
    "το n8n τρέχει τοπικά σε Docker",
])
def test_the_true_version_of_each_still_passes(text):
    """The ΙΕΛ really did publish the model; only the *data* claim is false.

    A check that cannot tell those apart would flag the correct answer about
    provenance, which is one the examiner is likely to ask.
    """
    assert check_technical_claims(text) == []


def test_greek_inflection_moves_the_accent():
    """«Αθήν\\w+» matched only the nominative.

    Greek inflection shifts the accent — Αθήνα, Αθηνάς — so a pattern with a
    fixed accent matches one form and silently misses the rest. Same family
    as the final-sigma bug: a rule that looks right and fires on almost
    nothing.
    """
    assert check_technical_claims("τα δεδομένα από τη βάση της Αθηνάς")
    assert check_technical_claims("το dataset από την Αθήνα")


def test_abbreviations_do_not_truncate_the_window():
    """"(π.χ. twitter)" contains full stops.

    The window was written as ``[^.!?;]`` to avoid crossing sentences, and
    the abbreviation cut it short — the claim landed just past the boundary.
    """
    assert check_technical_claims(
        "συλλογή δεδομένων από διάφορες πηγές (π.χ. twitter)"
    )


# ── Ένατη κατηγορία: αντικατάσταση αντικειμένου ──────────────


@pytest.mark.parametrize("text", [
    "κάνω μια εργασία που λέγεται «Ανίχνευση και Αντιμετώπιση Λογικών "
    "Σφαλμάτων σε Συστήματα Αυτόνομων Οχημάτων»",
    "η διπλωματική μου είναι πρόβλεψη τιμών ενέργειας μέσω deep learning",
    "Το RAG? Είναι ένα σύστημα αξιολόγησης. Red- Amber - Green",
    "Ροή Άμεσης Γέφυρας?",
])
def test_a_different_thesis_entirely(text):
    """No tool name is wrong; the whole subject is.

    Every check so far compares *names*. Here the names are absent and the
    object of the work has been replaced with something else plausible for a
    student of this department. Reported clean by all four checks, and by
    the tool written to measure exactly this.
    """
    from jarvis.inference.thesis_facts import check_acronym_expansions

    assert check_technical_claims(text) or check_acronym_expansions(text), text


@pytest.mark.parametrize("text", [
    "Η εργασία αφορά την ανάπτυξη ενός αυτόνομου ψηφιακού διδύμου με "
    "εξατομικευμένη επικοινωνία μέσω μεγάλων ανοιχτών γλωσσικών μοντέλων",
    "Η διπλωματική μου είναι ένα ψηφιακό δίδυμο βασισμένο στο Krikri-8B.",
    "Το RAG (Retrieval-Augmented Generation) συνδυάζει ανάκτηση και παραγωγή.",
])
def test_the_actual_title_passes(text):
    """The first of these is the thesis title, recited correctly."""
    from jarvis.inference.thesis_facts import check_acronym_expansions

    assert check_technical_claims(text) == []
    assert check_acronym_expansions(text) == []


def test_punctuation_between_acronym_and_gloss():
    """«Το RAG? Είναι…» — a bare \\s* does not cross the question mark.

    Restating the term as a question before explaining it is ordinary spoken
    Greek, not an edge case.
    """
    from jarvis.inference.thesis_facts import check_acronym_expansions

    assert check_acronym_expansions("Το RAG? Είναι σύστημα αξιολόγησης")
    assert check_acronym_expansions("RAG — είναι Red Amber Green")
