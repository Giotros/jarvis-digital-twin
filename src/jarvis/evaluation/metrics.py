"""Quantitative metrics for the experimental-validation chapter.

The thesis specification asks for measurement along three axes — *ακρίβεια*
(accuracy), *φυσικότητα* (naturalness) and *αξιοπιστία* (reliability). This
module implements a proxy for each that runs without a GPU, without network
access, and without model downloads, so the numbers can be regenerated on any
machine while writing up.

Scope and honesty about limitations
-----------------------------------
These are *lexical and structural* proxies, not semantic judgements:

* ``style_distance`` compares measurable surface features (length, lexical
  diversity, script mix, punctuation habits). Two texts can score close while
  differing in meaning. It answers "does this sound like the same writer?",
  not "does it mean the same thing".
* ``grounding_score`` checks whether specific claims appear in the supplied
  context. It detects *unsupported* specifics, which is the dominant failure
  mode of a persona twin; it cannot detect a fluent claim that happens to be
  wrong but is present in the context.

The thesis should state this plainly and position an NLI-based faithfulness
check and a human preference study as the validation these proxies stand in
for. Reporting a weak metric honestly is worth more than an impressive one
that does not measure what it claims.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, asdict
from typing import Any

__all__ = [
    "StyleProfile",
    "style_profile",
    "style_distance",
    "grounding_score",
    "refusal_rate",
    "distinct_n",
    "repetition_rate",
    "aggregate_report",
]

_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT = re.compile(r"[.!;?…]+")

#: Greek first-person singular is marked morphologically, not lexically: active
#: verbs end in -ω (κάνω, πάω, έρθω) and mediopassive in -μαι (είμαι, θυμάμαι).
#: A closed word list misses most of the inventory, so match the inflection —
#: plus the explicit first-person pronouns, which carry no verb ending.
_FIRST_PERSON_VERB = re.compile(r"\b\w{2,}(?:ω|ώ|μαι)\b", re.UNICODE)
_FIRST_PERSON_PRONOUN = re.compile(r"\b(?:εγω|μου|μενα|μας|εμεις)\b", re.UNICODE)

#: Words ending in -ω that are *not* first-person verbs. Small and closed:
#: Greek has very few such nouns/adverbs in conversational register.
_OMEGA_EXCEPTIONS = frozenset({"κατω", "πανω", "εδω", "πισω", "εξω", "μεσω", "ηχω"})

#: Phrases that betray assistant register rather than a person talking.
_ASSISTANT_TELLS = (
    "πώς μπορώ να σας βοηθήσω", "πως μπορω να σας βοηθησω",
    "είμαι εδώ για να", "ειμαι εδω για να",
    "ως τεχνητή νοημοσύνη", "ως τεχνητη νοημοσυνη",
    "ως ai", "as an ai", "i'm an ai", "language model",
    "δεν έχω τη δυνατότητα", "δεν εχω τη δυνατοτητα",
)

#: Acceptable ways of admitting ignorance. Counting these separately matters:
#: a refusal is a *success* when the model lacks grounding, and a *failure*
#: when the answer was retrievable.
_REFUSALS = (
    "δεν ξέρω", "δεν ξερω", "δεν θυμάμαι", "δεν θυμαμαι",
    "δεν είμαι σίγουρος", "δεν ειμαι σιγουρος", "δεν έχω ιδέα", "δεν εχω ιδεα",
    "i don't know", "i do not know", "not sure",
)


def _normalise(text: str) -> str:
    """Casefold and strip diacritics so 'θυμάμαι' == 'θυμαμαι'."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# ── Naturalness: style profiling ────────────────────────────────


@dataclass(frozen=True)
class StyleProfile:
    """Measurable surface features of a body of text.

    Every field is normalised to a comparable scale so that
    :func:`style_distance` can treat them uniformly.
    """

    mean_words_per_response: float
    mean_words_per_sentence: float
    mean_sentences_per_response: float
    type_token_ratio: float
    greek_ratio: float
    question_rate: float
    exclamation_rate: float
    first_person_rate: float
    assistant_tell_rate: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def style_profile(texts: Sequence[str]) -> StyleProfile:
    """Profile the writing style of a set of responses.

    ``type_token_ratio`` is computed over the pooled vocabulary and is
    length-sensitive by nature, so only compare profiles built from a similar
    number of samples — in practice, always the same probe set.
    """
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise ValueError("style_profile requires at least one non-empty text")

    per_response_words = [len(_words(t)) for t in texts]
    per_response_sents = [max(1, len(_sentences(t))) for t in texts]

    all_words = [w for t in texts for w in _words(t)]
    pooled = " ".join(texts)
    greek = len(_GREEK.findall(pooled))
    latin = len(_LATIN.findall(pooled))

    norm_texts = [_normalise(t) for t in texts]

    def _rate(markers: tuple[str, ...]) -> float:
        hits = sum(
            1 for t in norm_texts if any(_normalise(m) in t for m in markers)
        )
        return hits / len(texts)

    def _first_person_rate() -> float:
        hits = 0
        for t in norm_texts:
            verbs = [
                w for w in _FIRST_PERSON_VERB.findall(t)
                if w not in _OMEGA_EXCEPTIONS
            ]
            if verbs or _FIRST_PERSON_PRONOUN.search(t):
                hits += 1
        return hits / len(texts)

    return StyleProfile(
        mean_words_per_response=statistics.fmean(per_response_words),
        mean_words_per_sentence=statistics.fmean(
            w / s for w, s in zip(per_response_words, per_response_sents)
        ),
        mean_sentences_per_response=statistics.fmean(per_response_sents),
        type_token_ratio=(len(set(all_words)) / len(all_words)) if all_words else 0.0,
        greek_ratio=(greek / (greek + latin)) if (greek + latin) else 0.0,
        question_rate=sum(1 for t in texts if "?" in t or ";" in t) / len(texts),
        exclamation_rate=sum(1 for t in texts if "!" in t) / len(texts),
        first_person_rate=_first_person_rate(),
        assistant_tell_rate=_rate(_ASSISTANT_TELLS),
        n_samples=len(texts),
    )


#: Relative importance of each feature when computing style distance.
#: ``assistant_tell_rate`` dominates because assistant register is the single
#: most persona-breaking failure for a digital twin.
_STYLE_WEIGHTS: dict[str, float] = {
    "mean_words_per_response": 1.0,
    "mean_words_per_sentence": 1.0,
    "mean_sentences_per_response": 1.0,
    "type_token_ratio": 1.0,
    "greek_ratio": 1.5,
    "question_rate": 0.5,
    "exclamation_rate": 0.5,
    "first_person_rate": 2.0,
    "assistant_tell_rate": 3.0,
}

#: Scale factors turning unbounded counts into 0..1 before differencing.
_STYLE_SCALES: dict[str, float] = {
    "mean_words_per_response": 40.0,
    "mean_words_per_sentence": 20.0,
    "mean_sentences_per_response": 5.0,
}


def style_distance(a: StyleProfile, b: StyleProfile) -> float:
    """Weighted distance between two style profiles, in ``[0, 1]``.

    ``0.0`` means indistinguishable on these features. Interpret loosely:
    below ~0.15 is a close stylistic match, above ~0.40 is a different voice.
    Report the per-feature breakdown alongside it — the aggregate hides which
    dimension actually drifted.
    """
    total = 0.0
    weight_sum = 0.0
    for field_name, weight in _STYLE_WEIGHTS.items():
        scale = _STYLE_SCALES.get(field_name, 1.0)
        va = getattr(a, field_name) / scale
        vb = getattr(b, field_name) / scale
        total += weight * min(1.0, abs(va - vb))
        weight_sum += weight
    return total / weight_sum


def style_breakdown(a: StyleProfile, b: StyleProfile) -> dict[str, float]:
    """Per-feature absolute differences, for the discussion section."""
    return {
        name: round(abs(getattr(a, name) - getattr(b, name)), 4)
        for name in _STYLE_WEIGHTS
    }


# ── Reliability: grounding and refusals ─────────────────────────


def grounding_score(response: str, context: str, min_len: int = 4) -> float:
    """Fraction of the response's content words that appear in ``context``.

    Content words are those of length ``>= min_len``, which cheaply filters
    Greek articles and particles without a stopword list. Returns ``1.0`` for
    a response with no content words (a bare "ναι" cannot hallucinate).

    A low score means the model asserted specifics the retrieved context does
    not support — the signature of hallucination in a RAG pipeline.
    """
    resp_words = {w for w in _words(response) if len(w) >= min_len}
    if not resp_words:
        return 1.0
    ctx_norm = _normalise(context)
    supported = sum(1 for w in resp_words if _normalise(w) in ctx_norm)
    return supported / len(resp_words)


def refusal_rate(responses: Sequence[str]) -> float:
    """Share of responses that explicitly admit ignorance.

    Read against grounding: high refusal *with* empty context is correct
    behaviour; high refusal *with* rich context means the model is failing to
    use what it retrieved.
    """
    if not responses:
        return 0.0
    norm = [_normalise(r) for r in responses]
    hits = sum(1 for r in norm if any(_normalise(m) in r for m in _REFUSALS))
    return hits / len(responses)


def assistant_drift_rate(responses: Sequence[str]) -> float:
    """Share of responses that slip into assistant register.

    This is the metric to minimise: a digital twin that says "πώς μπορώ να σας
    βοηθήσω" has failed at its only job, regardless of fluency.
    """
    if not responses:
        return 0.0
    norm = [_normalise(r) for r in responses]
    hits = sum(1 for r in norm if any(_normalise(m) in r for m in _ASSISTANT_TELLS))
    return hits / len(responses)


# ── Fluency proxies ─────────────────────────────────────────────


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Distinct-n: unique n-grams over total n-grams (Li et al., 2016).

    Standard diversity measure for dialogue generation. Low values indicate
    a model falling back on a handful of canned phrasings.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    grams: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        w = _words(text)
        grams.update(tuple(w[i : i + n]) for i in range(len(w) - n + 1))
    total = sum(grams.values())
    return (len(grams) / total) if total else 0.0


def repetition_rate(text: str, n: int = 3) -> float:
    """Share of repeated n-grams *within* a single response.

    Degenerate repetition is the classic symptom of a bad decoding
    configuration or an over-fitted adapter.
    """
    w = _words(text)
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def mean_pairwise_similarity(texts: Sequence[str]) -> float:
    """Mean Jaccard similarity across all response pairs.

    High values mean the model answers everything the same way — persona
    collapse. Requires at least two texts.
    """
    sets = [set(_words(t)) for t in texts if _words(t)]
    if len(sets) < 2:
        return 0.0
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if union:
                sims.append(len(sets[i] & sets[j]) / len(union))
    return statistics.fmean(sims) if sims else 0.0


# ── Aggregate report ────────────────────────────────────────────


def aggregate_report(
    responses: Sequence[str],
    reference_style: StyleProfile | None = None,
    contexts: Sequence[str] | None = None,
    latencies_s: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Single call producing every number the results chapter needs.

    ``reference_style`` should be profiled from George's *real* messages
    (the held-out Viber split), giving the target the twin is measured
    against. ``contexts`` must align 1:1 with ``responses`` when supplied.
    """
    if not responses:
        raise ValueError("aggregate_report requires at least one response")
    if contexts is not None and len(contexts) != len(responses):
        raise ValueError(
            f"contexts/responses length mismatch: "
            f"{len(contexts)} vs {len(responses)}"
        )

    profile = style_profile(responses)

    report: dict[str, Any] = {
        "n_responses": len(responses),
        "naturalness": {
            "style_profile": profile.to_dict(),
            "distinct_1": round(distinct_n(responses, 1), 4),
            "distinct_2": round(distinct_n(responses, 2), 4),
            "mean_repetition_rate": round(
                statistics.fmean(repetition_rate(r) for r in responses), 4
            ),
            "mean_pairwise_similarity": round(mean_pairwise_similarity(responses), 4),
        },
        "reliability": {
            "refusal_rate": round(refusal_rate(responses), 4),
            "assistant_drift_rate": round(assistant_drift_rate(responses), 4),
            "empty_response_rate": round(
                sum(1 for r in responses if not r.strip()) / len(responses), 4
            ),
        },
    }

    if reference_style is not None:
        report["naturalness"]["style_distance_to_reference"] = round(
            style_distance(profile, reference_style), 4
        )
        report["naturalness"]["style_breakdown"] = style_breakdown(
            profile, reference_style
        )

    if contexts is not None:
        scores = [grounding_score(r, c) for r, c in zip(responses, contexts)]
        report["accuracy"] = {
            "mean_grounding_score": round(statistics.fmean(scores), 4),
            "ungrounded_rate": round(
                sum(1 for s in scores if s < 0.5) / len(scores), 4
            ),
        }

    if latencies_s:
        report["performance"] = {
            "mean_latency_s": round(statistics.fmean(latencies_s), 3),
            "median_latency_s": round(statistics.median(latencies_s), 3),
            "p95_latency_s": round(
                sorted(latencies_s)[min(len(latencies_s) - 1,
                                        math.ceil(0.95 * len(latencies_s)) - 1)], 3
            ),
        }

    return report


def report_markdown(report: dict[str, Any]) -> str:
    """Flatten :func:`aggregate_report` into a thesis-ready table."""
    lines = ["| Metric | Value |", "|---|---:|"]

    for section, body in report.items():
        if isinstance(body, dict):
            for key, value in body.items():
                if isinstance(value, dict):
                    for sub, subval in value.items():
                        lines.append(f"| {section}.{key}.{sub} | {subval} |")
                else:
                    lines.append(f"| {section}.{key} | {value} |")
        else:
            lines.append(f"| {section} | {body} |")
    return "\n".join(lines)
