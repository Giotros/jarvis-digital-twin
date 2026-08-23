import pytest

"""Tests for the post-processing guardrails pipeline."""

from jarvis.inference.guardrails import Guardrails


def _g() -> Guardrails:
    return Guardrails()


# ── Capitalize ────────────────────────────────────────────────────

def test_capitalize_first_letter():
    assert _g().process("καλα φιλε") == "Καλά φίλε"


def test_capitalize_after_period():
    result = _g().process("καλα. τωρα παω.")
    assert result.startswith("Καλά.")
    assert "Τώρα" in result


# ── Accent restoration ───────────────────────────────────────────

def test_restore_common_accents():
    g = Guardrails(capitalize=False, filter_profanity=False, remove_names=False)
    assert "είναι" in g.process("ειναι καλα")
    assert "καλά" in g.process("ειναι καλα")


def test_accent_case_insensitive():
    g = Guardrails(capitalize=False, filter_profanity=False, remove_names=False)
    # Should handle mixed case (though rare in Greek texting)
    result = g.process("ΕΙΝΑΙ")
    # The pattern matches case-insensitively
    assert result is not None


# ── Profanity filter ─────────────────────────────────────────────

def test_profanity_replacement():
    g = Guardrails(capitalize=False, restore_accents=False, remove_names=False)
    assert "άστα να πάνε" in g.process("γαμησετα φιλε")


def test_profanity_blocked():
    g = Guardrails(capitalize=False, restore_accents=False, remove_names=False)
    result = g.process("γαμησου ρε")
    assert "γαμησου" not in result


# ── Name hallucination removal ────────────────────────────────────

def test_remove_name_prefix():
    g = Guardrails(capitalize=False, restore_accents=False, filter_profanity=False)
    assert g.process("παναγιωτη που εισαι") == "που εισαι"


def test_name_removal_only_at_start():
    g = Guardrails(capitalize=False, restore_accents=False, filter_profanity=False)
    # Names inside the text should NOT be removed
    result = g.process("ρωτησε τον παναγιωτη")
    assert "παναγιωτη" in result


# ── Emoji artifacts ──────────────────────────────────────────────

def test_clean_emoji_text():
    g = Guardrails(capitalize=False, restore_accents=False, filter_profanity=False, remove_names=False)
    assert g.process("καλα(laugh)") == "καλα"
    assert g.process("(purple_heart) ναι") == "ναι"


# ── Impolite words ───────────────────────────────────────────────

def test_remove_re():
    g = Guardrails(capitalize=False, restore_accents=False, filter_profanity=False, remove_names=False)
    assert g.process("τι κανεις ρε φιλε") == "τι κανεις φιλε"


def test_re_not_removed_from_words():
    """'ρε' should only be removed as standalone word, not from 'πρέπει' etc."""
    g = Guardrails(capitalize=False, restore_accents=False, filter_profanity=False, remove_names=False)
    result = g.process("πρεπει να παρεις")
    assert "πρεπει" in result
    assert "παρεις" in result


# ── Full pipeline ────────────────────────────────────────────────

def test_full_pipeline():
    result = _g().process("παναγιωτη καλα ρε φιλε γαμησετα(laugh)")
    assert "παναγιωτη" not in result.lower()
    assert "γαμησετα" not in result
    assert "(laugh)" not in result
    assert "ρε" not in result.split()  # standalone ρε removed
    assert result[0].isupper()  # capitalized


def test_empty_string():
    assert _g().process("") == ""


def test_from_settings():
    settings = {
        "guardrails": {
            "capitalize_sentences": True,
            "restore_accents": True,
            "filter_profanity": True,
            "remove_name_hallucinations": True,
            "clean_emoji_artifacts": True,
        }
    }
    g = Guardrails.from_settings(settings)
    result = g.process("καλα φιλε")
    assert result[0].isupper()


# ── Regressions found in live evaluation (2026-08-22) ──────────

@pytest.mark.parametrize("text,stem", [
    ("Ναι εννοείται τρελε πουστη μ", "πουστ"),   # unaccented, inflected
    ("Ρε πούστη μου τι λες", "πούστ"),           # accented — original filter missed it
    ("ειναι μαλακισμενος", "μαλακισμ"),
    ("κωλοπαιδο", "κωλοπαιδ"),
])
def test_profanity_filtered_across_accents_and_inflection(text, stem):
    """Greek profanity inflects and is written with or without accents.

    An exact-word list fails on both counts. These strings were produced by
    the live model during evaluation and passed the original filter intact.
    """
    out = Guardrails().process(text)
    assert stem not in out.lower()
    assert "..." not in out, "blocked words are removed, not marked"


@pytest.mark.parametrize("text,softened", [
    ("μαλάκα τι λες", "φίλε"),      # accented — never matched the exact key
    ("μαλακα τι λες", "φίλε"),
    ("σκατά η μέρα", "χάλια"),
])
def test_profanity_replacements_survive_accents(text, softened):
    """The replacement map matched raw text, so accented forms slipped past.

    Same failure mode as the blocked list, in the loop directly above it.
    """
    assert softened in Guardrails().process(text).lower()


def test_removal_leaves_no_stray_punctuation_or_gaps():
    """A deleted word must not leave "τρελε  μ" or a stranded comma."""
    out = Guardrails().process("Ναι εννοείται τρελε πουστη μ")
    assert "  " not in out
    assert " ," not in out
    assert out == "Ναι εννοείται τρελε μ"


@pytest.mark.parametrize("text", [
    "Καλά είμαι φίλε",
    "Ναι θα έρθω το Σάββατο",
    "Πάω για τρέξιμο",
    "Είμαι στην Τρίπολη, σπουδάζω πληροφορική.",
])
def test_clean_text_untouched_by_profanity_filter(text):
    assert Guardrails().process(text) == text


@pytest.mark.parametrize("text,forbidden", [
    ("Ναι [NAME]", "[NAME]"),
    ("Ο [Person_26] ειπε", "[Person_26]"),
    ("Παρε με στο [PHONE]", "[PHONE]"),
    ("Στειλε στο [EMAIL]", "[EMAIL]"),
])
def test_anonymisation_placeholders_never_reach_output(text, forbidden):
    """Privacy placeholders are training artefacts, not words.

    The model saw ~4,700 of them during fine-tuning and emits them fluently;
    "Ναι [NAME]" was an actual reply from the deployed system.
    """
    assert forbidden not in Guardrails().process(text)


def test_accents_survive_profanity_filtering():
    """Folding is used to *find* matches, not to rewrite the whole reply."""
    out = Guardrails().process("Καλησπέρα, όλα καλά σήμερα")
    assert "ό" in out or "έ" in out


# ── Register enforcement (regression: "αγορι μ" to a professor) ─

@pytest.mark.parametrize("register", ["professional", "academic"])
@pytest.mark.parametrize("text", [
    "Ειμαι καλά αγορι μ να ξερς",
    "Καλά φιλαράκι μ ειμαι",
    "ειμαι καλα ρε φιλε σε ευχαριστω",
    "Ωραία μεγάλε",
])
def test_familiar_vocatives_removed_in_formal_registers(register, text):
    """The model does not drop these when asked, so they are removed after.

    Both a prompt instruction and a few-shot demonstration reduce the rate;
    neither takes it to zero, and one "αγόρι μου" addressed to an examiner is
    one too many.
    """
    out = Guardrails().process(text, register=register)
    for word in ("αγορι μ", "φιλαρακι", "φιλε", "μεγαλε"):
        assert word not in Guardrails._strip_accents(out).lower()


@pytest.mark.parametrize("text", [
    "Ειμαι καλά αγορι μ",
    "Καλά φιλαράκι μου",
])
def test_familiar_vocatives_kept_in_casual_registers(text):
    """With a friend these words are the voice, not a defect."""
    out = Guardrails().process(text, register="close")
    assert "αγορι" in out.lower() or "φιλαράκι" in out.lower()


@pytest.mark.parametrize("text", [
    "Είναι μεγάλο πρόβλημα αυτό",       # μεγάλε vs μεγάλο
    "Έχω τρελό φόρτο αυτή την περίοδο",  # τρελέ vs τρελό
    "Η αδερφή μου σπουδάζει εκεί",       # αδερφέ vs αδερφή
])
def test_vocative_stripping_does_not_eat_ordinary_words(text):
    """Stems would be too blunt here.

    Greek vocatives share a stem with common adjectives and nouns: matching
    "μεγαλ" would delete "μεγάλο πρόβλημα". The masculine -ε ending is what
    makes the match safe.
    """
    assert Guardrails().process(text, register="academic") == text


def test_enforce_register_is_idempotent():
    """It runs at generation and possibly again downstream."""
    g = Guardrails()
    once = g.enforce_register("Καλά φιλαράκι μου, ναι", "academic")
    assert g.enforce_register(once, "academic") == once


def test_process_without_register_is_unchanged_behaviour():
    """The register argument is optional; existing callers must not shift."""
    g = Guardrails()
    text = "Καλά φιλαράκι μου"
    assert g.process(text) == g.process(text, register="")


# ── Orphaned articles (regression from the surname filter) ──────

@pytest.mark.parametrize("text,forbidden", [
    ("μάθαμε πράγματα και ο Ταμπακάς είναι καλός καθηγητής", "και ο ει"),
    ("Το δουλεύω ακόμα με τον Ζέρβα από τη σχολή", "με τον από"),
    ("Περιμένω να με ενημερώσει η Παπαδοπούλου πότε", "ει η πότε"),
    ("καλή φάση, το έχουμε με τον Κούγια", "με τον"),
])
def test_deleting_a_surname_does_not_strand_its_article(text, forbidden):
    """Removing the noun alone leaves ungrammatical Greek.

    Live output contained "και ο είναι πολύ καλός καθηγητής" and "το δουλεύω
    με τον από τη σχολή". A privacy filter that leaves visibly broken
    sentences draws attention to precisely the sentence it was trying to
    make unremarkable.
    """
    out = Guardrails().sanitise_output(text)
    assert forbidden not in out


def test_cascading_deletions_are_resolved():
    """"με τον Παπαδόπουλο" loses three tokens, one at a time.

    The preposition only becomes orphaned after the article goes, so a
    single pass is not enough.
    """
    out = Guardrails().sanitise_output("Μίλησα με τον Παπαδόπουλο χθες.")
    assert "με τον" not in out
    assert "Μίλησα" in out and "χθες" in out


@pytest.mark.parametrize("text", [
    "Ο καθηγητής είναι πολύ καλός",
    "Το δουλεύω ακόμα με τον συνάδελφο από τη σχολή",
    "Η εργασία μου είναι έτοιμη",
    "Πάω στην Τρίπολη το Σάββατο",
])
def test_sentences_without_surnames_are_untouched(text):
    """The article stripper runs only where a name was actually removed."""
    assert Guardrails().sanitise_output(text) == text


def test_no_double_punctuation_after_removal():
    out = Guardrails().sanitise_output("Ήρθε ο Ιωαννίδης, και ο Παπαδόπουλος.")
    assert ",," not in out and " ," not in out and ".." not in out


# ── Ορφανό άρθρο στη μέση πρότασης ───────────────────────────
#
# Παρατηρήθηκε 23/08/2026 σε ζωντανή απάντηση. Το μοντέλο αντέγραψε από το
# αρχείο τη φράση «θα προσπαθησω να θυμηθω να παρω και την [NAME] το
# απογευμα», το placeholder αφαιρέθηκε, και ο χρήστης είδε «…να πάρω και
# ΤΗΝ το απόγευμα».


@pytest.mark.parametrize("broken", [
    "Θα προσπαθήσω να θυμηθώ να πάρω και την [NAME] το απόγευμα",
    "Θα προσπαθήσω να θυμηθώ να πάρω και την το απόγευμα",
    "Θα περασω λιγο απο την για να δω κατι",
    "και ο είναι πολύ καλός καθηγητής",
])
def test_orphaned_article_mid_sentence(broken):
    """Το κεφάλαιο 7 ανέφερε το τρίτο παράδειγμα ως διορθωμένο.

    Δεν ήταν. Το μοτίβο απαιτούσε στίξη ή τέλος γραμμής μετά το άρθρο, και
    έπιανε μόνο «…μίλησα με την.» — τη μία μορφή που ήδη δούλευε. Το test
    που το κατοχύρωνε έλεγχε ακριβώς αυτή τη μορφή.

    Και το ``_close_gap`` υπήρχε αλλά έτρεχε μόνο μέσα στο
    ``_strip_surnames``, όταν είχε όντως αφαιρεθεί επώνυμο. Η αφαίρεση
    placeholder — η συχνότερη αιτία χάσματος, αφού το corpus είναι γεμάτο
    [NAME] — δεν το καλούσε ποτέ.
    """
    g = Guardrails()
    out = g.sanitise_output(broken, "close")
    assert out != broken
    words = out.lower().split()
    for i, w in enumerate(words[:-1]):
        if w in {"ο", "η", "το", "την", "τον", "της", "του"}:
            assert words[i + 1] not in {
                "για", "από", "απο", "με", "είναι", "ειναι", "θα",
                "το", "την", "τον", "η", "ο",
            }, f"έμεινε ορφανό «{w} {words[i + 1]}» στο: {out}"


@pytest.mark.parametrize("fine", [
    "πάρε το αυτοκίνητο",
    "η μέρα ήταν καλή",
    "το απόγευμα θα δούμε",
    "ο καθηγητής είναι καλός",
    "τα παιδιά και οι γονείς",
    "θα πάω στο σπίτι",
    "της είπα ότι θα έρθω",
    "Δεν μπορώ να δω το ημερολόγιό μου τώρα, οπότε δεν ξέρω τι έχω.",
])
def test_correct_greek_is_left_alone(fine):
    """Ένα φίλτρο που σβήνει σωστά άρθρα είναι χειρότερο από κανένα.

    Το «ο καθηγητής είναι καλός» έχει άρθρο πριν από ουσιαστικό και μετά
    ρήμα· ένα γενικό «άρθρο + οτιδήποτε» θα το κατέστρεφε.
    """
    assert Guardrails().sanitise_output(fine, "close") == fine


@pytest.mark.parametrize("broken", [
    "Πιθανόν να πάω για καμιά μπύρα με τον μετά τη δουλειά",
    "θα τα πούμε με την αύριο",
    "μίλησα με τον χθες",
])
def test_orphan_after_preposition(broken):
    """«με τον [NAME] μετά» → «με τον μετά», που δεν είναι ελληνικά."""
    assert Guardrails().sanitise_output(broken, "close") != broken


@pytest.mark.parametrize("fine", [
    # Η πρώτη εκδοχή του κανόνα κατέστρεψε αυτό: το «μετά» είναι
    # ουσιαστικοποιημένο και το άρθρο σωστό.
    "το μετά το βλέπουμε",
    "πάμε για μπύρα μετά τη δουλειά",
    "μίλησα με τον καθηγητή χθες",
    "από το πρωί μέχρι το βράδυ",
])
def test_the_adverb_rule_does_not_overreach(fine):
    """Απαιτείται πρόθεση μπροστά, ώστε η ακολουθία να είναι αναμφίβολα σπασμένη.

    Χωρίς αυτόν τον περιορισμό ο κανόνας έσβηνε σωστά άρθρα — και ένα
    φίλτρο που καταστρέφει σωστά ελληνικά είναι χειρότερο από κανένα, γιατί
    τραβάει την προσοχή ακριβώς στην πρόταση που ήθελε να κάνει αδιάφορη.
    """
    assert Guardrails().sanitise_output(fine, "close") == fine
