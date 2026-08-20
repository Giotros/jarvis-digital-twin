"""Training-pair formatting for Krikri-8B and Mistral-7B.

Supports two chat formats:
  - Krikri/Llama 3.1: uses tokenizer.apply_chat_template() with system/user/assistant
  - Mistral (baseline): <s>[INST] {instruction} [/INST] {response}</s>

The Mistral template is preserved byte-identical for compatibility with
the existing Persona-Chat and Viber LoRA adapters (trained 2026-06-21).

For Krikri, formatting is done via the tokenizer's built-in chat template
(see the Colab training cell), so the KRIKRI_* constants here are for
reference and testing only.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from pathlib import Path

# ── Mistral format (baseline, frozen) ──────────────────────────────
MISTRAL_TEMPLATE = "<s>[INST] {instruction} [/INST] {response}</s>"

# ── Format identifiers ────────────────────────────────────────────
CHAT_FORMAT_LLAMA3 = "llama3"
CHAT_FORMAT_MISTRAL = "mistral"

_GREEK_CHARS = re.compile(r"[Ͱ-Ͽἀ-῿]")


def format_pair(instruction: str, response: str, chat_format: str = "mistral") -> str:
    """One training example in the specified format.

    For Mistral: returns the full formatted string.
    For Krikri/Llama3: returns a dict-based representation (actual formatting
    happens via tokenizer.apply_chat_template in the training notebook).
    """
    if chat_format == CHAT_FORMAT_MISTRAL:
        return MISTRAL_TEMPLATE.format(
            instruction=instruction.strip(), response=response.strip()
        )
    else:
        # For llama3, return the text that would be produced by apply_chat_template
        # This is a simplified version; actual training uses the tokenizer
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction.strip()}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response.strip()}<|eot_id|>"
        )


def format_pair_as_messages(
    instruction: str, response: str, system_prompt: str = ""
) -> list[dict[str, str]]:
    """Return a message list suitable for tokenizer.apply_chat_template().

    This is the preferred format for Krikri/Llama3 training.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction.strip()})
    messages.append({"role": "assistant", "content": response.strip()})
    return messages


def format_records(
    records: Sequence[dict], chat_format: str = "mistral"
) -> list[str]:
    """Format a list of {"instruction", "response"} records; skips invalid rows."""
    formatted = []
    for rec in records:
        instruction = rec.get("instruction") or rec.get("instruction_clean")
        response = rec.get("response") or rec.get("response_clean")
        if instruction and response:
            formatted.append(format_pair(instruction, response, chat_format))
    return formatted


def load_pairs(path: str | Path) -> list[dict]:
    """Load a JSON list of pair records, validating basic shape."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list of records")

    # Support both key formats: instruction/response and instruction_clean/response_clean
    valid = 0
    for r in records:
        if isinstance(r, dict):
            has_instr = bool(r.get("instruction") or r.get("instruction_clean"))
            has_resp = bool(r.get("response") or r.get("response_clean"))
            if has_instr and has_resp:
                valid += 1

    bad = len(records) - valid
    if bad:
        print(f"WARNING {path}: {bad}/{len(records)} records missing instruction/response")
    return records


def train_val_split(
    records: Sequence[dict], val_ratio: float = 0.05, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle-and-split (fixed seed → reproducible thesis runs)."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def corpus_stats(records: Sequence[dict]) -> dict:
    """Quick corpus profile for the thesis (sizes, Greek/Latin mix)."""
    n = len(records)
    if not n:
        return {"pairs": 0}
    resp_lens = [
        len(r.get("response") or r.get("response_clean") or "")
        for r in records
    ]
    greek = sum(
        1 for r in records
        if _GREEK_CHARS.search(r.get("response") or r.get("response_clean") or "")
    )
    return {
        "pairs": n,
        "avg_response_chars": round(sum(resp_lens) / n, 1),
        "max_response_chars": max(resp_lens),
        "greek_response_ratio": round(greek / n, 3),
    }
