"""Phase 3 — training-data utilities (Krikri + Mistral instruction formats)."""

from jarvis.training.dataset import (
    CHAT_FORMAT_LLAMA3,
    CHAT_FORMAT_MISTRAL,
    MISTRAL_TEMPLATE,
    corpus_stats,
    format_pair,
    format_pair_as_messages,
    format_records,
    load_pairs,
    train_val_split,
)

__all__ = [
    "CHAT_FORMAT_LLAMA3", "CHAT_FORMAT_MISTRAL",
    "MISTRAL_TEMPLATE", "format_pair", "format_pair_as_messages",
    "format_records", "load_pairs", "train_val_split", "corpus_stats",
]
