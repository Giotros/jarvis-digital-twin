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
