"""RAG context builder — loads sanitized conversations and builds a searchable corpus.

This module bridges the training data (sanitized Viber pairs) with the
HybridSearcher to provide relevant context at inference time. It handles:

1. Loading the sanitized JSON into a searchable corpus
2. Building conversation-aware documents (instruction + response together)
3. Formatting retrieved results into context strings for the model

For production (Databricks), embeddings come from gold.george_embeddings.
For local/Colab use, BM25-only mode works without any embedding model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jarvis.rag.hybrid_search import HybridSearcher, SearchResult


def load_corpus_from_json(
    path: str | Path,
    max_pairs: int | None = None,
) -> tuple[list[str], list[dict]]:
    """Load sanitized training data as a searchable corpus.

    Each document = the user's message + George's response, concatenated.
    This way, searching for "Κέρκυρα" finds conversations WHERE George
    talked about Corfu, giving the model both the question pattern and
    George's actual response style for that topic.

    Returns:
        texts:    list of searchable document strings
        metadata: list of dicts with original instruction/response
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")

    texts = []
    metadata = []
    for i, pair in enumerate(data):
        instruction = pair.get("instruction_clean") or pair.get("instruction") or ""
        response = pair.get("response_clean") or pair.get("response") or ""
        if not instruction or not response:
            continue

        # Document = both sides of the conversation for richer search
        doc_text = f"{instruction}\n{response}"
        texts.append(doc_text)
        metadata.append({
            "index": i,
            "instruction": instruction,
            "response": response,
        })

        if max_pairs and len(texts) >= max_pairs:
            break

    return texts, metadata


def build_searcher(
    corpus_path: str | Path,
    embeddings: list[list[float]] | None = None,
    embed_fn: Any = None,
    settings: dict | None = None,
    max_pairs: int | None = None,
) -> HybridSearcher:
    """Build a HybridSearcher from the sanitized training data.

    Args:
        corpus_path: path to sanitized JSON file
        embeddings:  pre-computed vectors (from Databricks export)
        embed_fn:    function to embed a query string
        settings:    settings dict (uses rag.* keys for thresholds)
        max_pairs:   limit corpus size (for testing)
    """
    texts, metadata = load_corpus_from_json(corpus_path, max_pairs)

    rag_settings = (settings or {}).get("rag", {})

    searcher = HybridSearcher(
        texts=texts,
        embeddings=embeddings,
        embed_fn=embed_fn,
        metadata=metadata,
        rrf_k=rag_settings.get("rrf_k", 60),
        threshold=rag_settings.get("relevance_threshold", 0.62),
        bm25_k1=rag_settings.get("bm25_k1", 1.5),
        bm25_b=rag_settings.get("bm25_b", 0.75),
    )

    return searcher


def format_context(
    results: list[SearchResult],
    max_results: int = 3,
    style: str = "conversation",
) -> str:
    """Format search results into a context string for the model.

    Args:
        results:     search results from HybridSearcher
        max_results: maximum results to include
        style:       "conversation" shows Q&A pairs; "flat" shows raw text

    Returns:
        Formatted context string to inject into the system prompt
    """
    if not results:
        return ""

    parts = []
    for r in results[:max_results]:
        if style == "conversation":
            instruction = r.metadata.get("instruction", "")
            response = r.metadata.get("response", "")
            parts.append(f"Ερώτηση: {instruction}\nΑπάντηση Γιώργου: {response}")
        else:
            parts.append(r.text)

    return "\n---\n".join(parts)
