#!/usr/bin/env python3
"""Produce jarvis_training_data_sanitized.json from the raw Viber export.

This is the missing pipeline step that blocks Part B (Viber) training:
the raw jarvis_training_data.json on Google Drive contains PII and must
never reach the trainer.

Run:
    python scripts/run_sanitization.py \
        --input  ~/Downloads/jarvis_training_data.json \
        --output ~/Downloads/jarvis_training_data_sanitized.json \
        --contacts config/contacts.txt

Then upload the *_sanitized.json to Drive (MyDrive root) and run Part B
of notebooks/Jarvis_George_Phase3_QLoRA_v3.ipynb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis.sanitization import PIISanitizer  # noqa: E402
from jarvis.sanitization.patterns import PII_PATTERNS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw JSON (list of records)")
    parser.add_argument("--output", required=True, help="sanitized JSON destination")
    parser.add_argument("--contacts", default="config/contacts.txt")
    parser.add_argument(
        "--fields", nargs="+", default=["instruction", "response"],
        help="record fields to sanitize",
    )
    args = parser.parse_args()

    contacts_path = Path(args.contacts)
    if not contacts_path.exists():
        print(f"ERROR: contacts file not found: {contacts_path}")
        print("Copy config/contacts.example.txt → config/contacts.txt and fill it in.")
        return 1

    sanitizer = PIISanitizer.from_contacts_file(contacts_path)
    print(f"Loaded {sanitizer.n_contacts} contact-name variants.")

    report = sanitizer.sanitize_json_file(args.input, args.output, fields=args.fields)
    print(report.summary())

    # Post-check: rescan output for any structural PII that slipped through.
    residual = 0
    text = Path(args.output).read_text(encoding="utf-8")
    for category, pattern, validator in PII_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            if validator is None or validator("".join(value.split())):
                residual += 1
                print(f"WARNING residual {category}: …{value[:4]}**** ")
    if residual:
        print(f"\n⚠ {residual} residual matches — review before training!")
        return 2
    print("\n✓ Post-scan clean. Safe to upload for Part B training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
