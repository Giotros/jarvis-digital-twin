"""Tests for RAG context builder."""

import json
import tempfile
from pathlib import Path

from jarvis.rag.context_builder import (
    build_searcher,
    format_context,
    load_corpus_from_json,
)
from jarvis.rag.hybrid_search import SearchResult


def _make_corpus_file(pairs: list[dict]) -> Path:
    """Create a temp JSON file with test pairs."""
    p = Path(tempfile.mktemp(suffix=".json"))
    p.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    return p


SAMPLE_PAIRS = [
    {
        "instruction_clean": "[Person_1]: Πάμε για καφέ αύριο;",
        "response_clean": "ναι εννοειται, που θες να παμε;",
    },
    {
        "instruction_clean": "[Person_2]: Τι γίνεται με την Κέρκυρα;",
        "response_clean": "ολα καλα εδω, ομορφα ησυχα",
    },
    {
        "instruction_clean": "[Person_3]: Έχω πρόβλημα με το ίντερνετ",
        "response_clean": "δοκιμασε να κανεις restart το router",
    },
    {
        "instruction_clean": "[Person_4]: Τι δουλειά κάνεις;",
        "response_clean": "δουλευω στις τηλεπικοινωνιες, eshop και τεχνικη υποστηριξη",
    },
    {
        "instruction_clean": "[Person_5]: Πόσο χρονών είσαι;",
        "response_clean": "28 αδερφε",
    },
]


def test_load_corpus_from_json():
    path = _make_corpus_file(SAMPLE_PAIRS)
    texts, metadata = load_corpus_from_json(path)
    assert len(texts) == 5
    assert len(metadata) == 5
    assert "Κέρκυρα" in texts[1]
    assert "ομορφα ησυχα" in texts[1]
    assert metadata[1]["response"] == "ολα καλα εδω, ομορφα ησυχα"
    path.unlink()


def test_load_corpus_max_pairs():
    path = _make_corpus_file(SAMPLE_PAIRS)
    texts, metadata = load_corpus_from_json(path, max_pairs=2)
    assert len(texts) == 2
    path.unlink()


def test_load_corpus_supports_old_keys():
    """Works with instruction/response (not just _clean variants)."""
    pairs = [{"instruction": "γεια", "response": "τι λέει"}]
    path = _make_corpus_file(pairs)
    texts, metadata = load_corpus_from_json(path)
    assert len(texts) == 1
    assert "γεια" in texts[0]
    path.unlink()


def test_build_searcher_and_search():
    path = _make_corpus_file(SAMPLE_PAIRS)
    searcher = build_searcher(path)
    results = searcher.search("Κέρκυρα", top_k=2)
    # BM25-only mode: should find the Corfu conversation
    assert len(results) >= 1
    assert any("Κέρκυρα" in r.text for r in results)
    path.unlink()


def test_build_searcher_internet_query():
    path = _make_corpus_file(SAMPLE_PAIRS)
    searcher = build_searcher(path)
    results = searcher.search("ίντερνετ πρόβλημα", top_k=2)
    assert len(results) >= 1
    assert any("router" in r.text for r in results)
    path.unlink()


def test_format_context_conversation_style():
    results = [
        SearchResult(
            doc_id=0, text="test", score=1.0,
            metadata={
                "instruction": "Πάμε για καφέ;",
                "response": "ναι εννοειται",
            },
        )
    ]
    ctx = format_context(results, style="conversation")
    assert "Ερώτηση: Πάμε για καφέ;" in ctx
    assert "Απάντηση Γιώργου: ναι εννοειται" in ctx


def test_format_context_empty():
    assert format_context([]) == ""


def test_format_context_max_results():
    results = [
        SearchResult(doc_id=i, text=f"doc{i}", score=1.0, metadata={
            "instruction": f"q{i}", "response": f"a{i}"
        })
        for i in range(5)
    ]
    ctx = format_context(results, max_results=2)
    assert "q0" in ctx
    assert "q1" in ctx
    assert "q2" not in ctx


# ── Context framing (regression: ungrounded_rate 0.91) ─────────

from jarvis.rag.context_builder import frame_context  # noqa: E402


def test_framing_marks_the_material_as_past():
    """Retrieved pairs look exactly like a few-shot prompt.

    format_context emits "Ερώτηση: … / Απάντηση Γιώργου: …", so under a
    neutral heading the model continues the pattern and copies an archived
    answer. It produced "Ναι, 06/10 θα είμαι εκεί για την εγκατάσταση του
    νέου server" in reply to a question about nothing of the sort.
    """
    out = frame_context("Ερώτηση: θα ερθεις;\nΑπάντηση Γιώργου: Ναι 06/10")
    assert "ΔΕΝ είναι η τωρινή κουβέντα" in out
    assert "ΜΗΝ αντιγράφεις" in out


def test_framing_delimits_the_retrieved_block():
    """The model must be able to tell where the archive ends."""
    out = frame_context("κάτι")
    assert out.index("<<<ΑΡΧΕΙΟ") < out.index("κάτι") < out.index("ΤΕΛΟΣ ΑΡΧΕΙΟΥ>>>")


def test_framing_says_it_is_allowed_to_ignore_the_archive():
    """Retrieval always returns *something*, relevant or not.

    Without explicit permission to discard it, the model treats an unrelated
    top-scoring match as evidence — which is how casual questions came back
    answered with e-shop chatter.
    """
    assert "αγνόησέ το" in frame_context("άσχετο κείμενο")


def test_empty_context_produces_no_section():
    """Callers concatenate unconditionally; an empty frame would be noise."""
    assert frame_context("") == ""
    assert frame_context("   ") == ""
