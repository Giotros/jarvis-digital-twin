"""Tests for BM25 + vector + RRF hybrid retrieval."""

import pytest

from jarvis.rag import BM25, HybridSearcher, cosine_similarity, rrf_fuse


def test_rrf_math():
    """RRF(d) = Σ 1/(k + rank). Doc appearing top in both lists must win."""
    scores = rrf_fuse([[1, 2, 3], [1, 3, 2]], k=60)
    assert scores[1] == pytest.approx(2 / 61)
    assert scores[1] > scores[2] and scores[1] > scores[3]


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0        # zero-vector guard


def test_bm25_ranks_keyword_doc_first():
    corpus = [
        "ο σκύλος γαβγίζει όλη νύχτα",
        "η γάτα κοιμάται στον καναπέ",
        "completely unrelated english text",
    ]
    ranking = BM25(corpus).rank("σκύλος")
    assert ranking[0] == 0


def test_hybrid_threshold_gates_irrelevant_docs():
    """Docs below the vector-relevance threshold must not be returned
    even if BM25 ranks them — the anti-hallucination gate."""
    texts = ["πληρωμή τιμολογίου project alpha", "συνάντηση για καφέ"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    searcher = HybridSearcher(
        texts,
        embeddings=embeddings,
        embed_fn=lambda q: [1.0, 0.0],          # query ≈ doc 0
        threshold=0.62,
    )
    results = searcher.search("τιμολόγιο project", top_k=5)
    assert [r.doc_id for r in results] == [0]   # doc 1: cos=0 < 0.62 → gated


def test_translate_fn_is_used_for_embedding_leg():
    calls = []

    def translate(q: str) -> str:
        calls.append(q)
        return "invoice project"

    searcher = HybridSearcher(
        ["invoice for project alpha", "coffee meetup"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        embed_fn=lambda q: [1.0, 0.0],
        translate_fn=translate,
    )
    searcher.search("τιμολόγιο για το project")
    assert calls == ["τιμολόγιο για το project"]
