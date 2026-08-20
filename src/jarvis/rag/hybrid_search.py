"""Hybrid retrieval: BM25 (lexical) + dense vectors (semantic) fused with RRF.

Design rationale (thesis §2.2.4): BM25 catches exact tokens (names, technical
terms) that embeddings blur; embeddings catch paraphrases that BM25 misses;
Reciprocal Rank Fusion (Cormack et al., SIGIR 2009) combines both rankings
robustly without score calibration:  RRF(d) = Σ_r 1 / (k + rank_r(d)).

Production retrieval runs on Databricks over gold.george_embeddings
(databricks-gte-large-en, 1024-dim). This module is the reference
implementation used for local development, unit tests and evaluation —
the algorithm is identical.

Greek queries: embeddings are strongest in English, so when
settings.rag.translate_greek_queries is on, pass a translate_fn
(on Databricks: ai_query against the generation endpoint).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-zA-Z0-9Ͱ-Ͽἀ-῿]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; handles Greek and Latin scripts."""
    return _TOKEN_RE.findall(text.lower())


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Fuse ranked lists of doc ids → {doc_id: rrf_score}.

    *k* dampens the impact of top ranks (k=60 is the published default).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class BM25:
    """Okapi BM25 over an in-memory corpus (k1=1.5, b=0.75 defaults)."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [tokenize(doc) for doc in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0
        self.doc_freqs: list[Counter[str]] = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str, doc_id: int) -> float:
        freqs, dl = self.doc_freqs[doc_id], self.doc_len[doc_id]
        s = 0.0
        for term in tokenize(query):
            if term not in freqs:
                continue
            tf = freqs[term]
            s += self.idf.get(term, 0.0) * (
                tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            )
        return s

    def rank(self, query: str, top_k: int | None = None) -> list[int]:
        order = sorted(range(len(self.docs)), key=lambda i: self.score(query, i), reverse=True)
        return order[:top_k] if top_k else order


@dataclass
class SearchResult:
    doc_id: int
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class HybridSearcher:
    """BM25 + vector search + RRF, with relevance thresholding.

    Args:
        texts:        the document corpus.
        embeddings:   optional pre-computed vectors aligned with *texts*.
        embed_fn:     query → vector (required for the semantic leg).
        translate_fn: optional Greek→English query rewrite before embedding.
        rrf_k:        RRF dampening constant (settings.rag.rrf_k).
        threshold:    minimum *vector* similarity for a doc to be eligible —
                      the anti-hallucination gate (settings.rag.relevance_threshold).
    """

    def __init__(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]] | None = None,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        translate_fn: Callable[[str], str] | None = None,
        metadata: Sequence[dict] | None = None,
        rrf_k: int = 60,
        threshold: float = 0.62,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.texts = list(texts)
        self.embeddings = list(embeddings) if embeddings is not None else None
        self.embed_fn = embed_fn
        self.translate_fn = translate_fn
        self.metadata = list(metadata) if metadata else [{} for _ in self.texts]
        self.rrf_k, self.threshold = rrf_k, threshold
        self.bm25 = BM25(self.texts, k1=bm25_k1, b=bm25_b)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        rankings: list[list[int]] = [self.bm25.rank(query, top_k=max(top_k * 4, 20))]
        similarities: dict[int, float] = {}

        if self.embeddings is not None and self.embed_fn is not None:
            embed_query = query
            if self.translate_fn is not None:
                embed_query = self.translate_fn(query)
            q_vec = self.embed_fn(embed_query)
            similarities = {
                i: cosine_similarity(q_vec, vec) for i, vec in enumerate(self.embeddings)
            }
            vector_ranking = sorted(similarities, key=similarities.get, reverse=True)
            rankings.append(vector_ranking[: max(top_k * 4, 20)])

        fused = rrf_fuse(rankings, k=self.rrf_k)
        results = []
        for doc_id, score in sorted(fused.items(), key=lambda kv: -kv[1]):
            # Anti-hallucination gate: if we have vector evidence and it is
            # below threshold, the doc is not relevant enough to ground on.
            if similarities and similarities.get(doc_id, 0.0) < self.threshold:
                continue
            results.append(
                SearchResult(doc_id, self.texts[doc_id], score, self.metadata[doc_id])
            )
            if len(results) >= top_k:
                break
        return results
