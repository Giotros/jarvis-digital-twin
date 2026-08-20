"""Phase 2 — RAG: hybrid retrieval (BM25 + vectors + RRF) + context builder."""

from jarvis.rag.hybrid_search import BM25, HybridSearcher, cosine_similarity, rrf_fuse
from jarvis.rag.context_builder import build_searcher, format_context, load_corpus_from_json

__all__ = [
    "BM25", "HybridSearcher", "cosine_similarity", "rrf_fuse",
    "build_searcher", "format_context", "load_corpus_from_json",
]
