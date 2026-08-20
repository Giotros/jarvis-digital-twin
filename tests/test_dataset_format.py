"""Tests for training-pair formatting — both Krikri and Mistral formats."""

from jarvis.training import (
    corpus_stats,
    format_pair,
    format_pair_as_messages,
    format_records,
    train_val_split,
)


# ── Mistral format (baseline, byte-identical to Part A run) ────────

def test_mistral_template_is_byte_identical_to_part_a_run():
    """Changing this string would break compatibility with the trained adapters."""
    assert format_pair("γεια", "τι λέει", "mistral") == "<s>[INST] γεια [/INST] τι λέει</s>"


def test_mistral_format_pair_strips_whitespace():
    assert format_pair("  hi \n", "\tyo ", "mistral") == "<s>[INST] hi [/INST] yo</s>"


# ── Krikri/Llama3 format ──────────────────────────────────────────

def test_llama3_format_contains_header_tags():
    result = format_pair("γεια", "τι λέει", "llama3")
    assert "<|start_header_id|>user<|end_header_id|>" in result
    assert "<|start_header_id|>assistant<|end_header_id|>" in result
    assert "γεια" in result
    assert "τι λέει" in result


def test_llama3_format_strips_whitespace():
    result = format_pair("  hi \n", "\tyo ", "llama3")
    assert "hi<|eot_id|>" in result
    assert "yo<|eot_id|>" in result


def test_format_pair_as_messages_with_system():
    messages = format_pair_as_messages("γεια", "τι λέει", system_prompt="Είσαι ο Jarvis.")
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Είσαι ο Jarvis."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "γεια"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "τι λέει"


def test_format_pair_as_messages_without_system():
    messages = format_pair_as_messages("γεια", "τι λέει")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"


# ── Common tests ──────────────────────────────────────────────────

def test_format_records_skips_invalid_rows():
    records = [
        {"instruction": "a", "response": "b"},
        {"instruction": "", "response": "b"},
        {"response": "only"},
        {"instruction": "c", "response": "d"},
    ]
    assert len(format_records(records)) == 2


def test_format_records_supports_clean_keys():
    """Sanitized data uses instruction_clean/response_clean keys."""
    records = [
        {"instruction_clean": "a", "response_clean": "b"},
        {"instruction_clean": "c", "response_clean": "d"},
    ]
    assert len(format_records(records)) == 2


def test_train_val_split_deterministic():
    records = [{"instruction": str(i), "response": str(i)} for i in range(100)]
    train1, val1 = train_val_split(records, val_ratio=0.1, seed=42)
    train2, val2 = train_val_split(records, val_ratio=0.1, seed=42)
    assert train1 == train2 and val1 == val2
    assert len(val1) == 10 and len(train1) == 90


def test_corpus_stats_greek_ratio():
    records = [
        {"instruction": "α", "response": "γεια σου"},
        {"instruction": "b", "response": "hello there"},
    ]
    stats = corpus_stats(records)
    assert stats["pairs"] == 2
    assert stats["greek_response_ratio"] == 0.5


def test_corpus_stats_clean_keys():
    """Stats work with sanitized data keys too."""
    records = [
        {"instruction_clean": "α", "response_clean": "γεια σου"},
    ]
    stats = corpus_stats(records)
    assert stats["pairs"] == 1
    assert stats["greek_response_ratio"] == 1.0
