#!/usr/bin/env python3
"""Produce a corpus with surnames removed, and refuse to ship a dirty one.

The output filter in :mod:`jarvis.inference.guardrails` covers a surname
that the model has already learned. It does not unlearn it. Two real
surnames sat in v4 through 34 occurrences, went into the adapter, and came
back out of the deployed model — and no amount of post-processing changes
what is in the weights. Only retraining on clean data does.

This is the finding worth putting in the thesis: an anonymisation failure is
not fixable after training. It is maskable. The distinction matters because
masking looks identical from outside until the mask has a gap.

Usage:
    python3 scripts/resanitise_surnames.py                    # dry run
    python3 scripts/resanitise_surnames.py --write            # write v5
    python3 scripts/resanitise_surnames.py --check data/x.json

``--check`` exits non-zero when the file still contains surnames, so it can
gate a training run rather than being remembered by a person at the right
moment.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jarvis.sanitization.greek_surnames import (  # noqa: E402
    find_surnames,
    gazetteer_status,
    load_gazetteer,
    redact_surnames,
)

#: Text fields in a corpus record. Every one is fed to the model, so every
#: one has to be cleaned — the first sanitiser missed a field and the
#: omission was invisible until inference.
_TEXT_FIELDS = (
    "instruction_clean", "response_clean", "formatted_prompt",
    "instruction", "response", "conversation_with",
)

#: Fields to *count* in. Deliberately narrower than the set to clean.
#:
#: ``formatted_prompt`` is a concatenation of the other two, so every
#: redaction appears in it a second time. Counting across all fields doubles
#: the figures — which is how "592 phone numbers" and "1,187 surnames"
#: entered the project's notes when the true counts are 296 and 589. The
#: cleaning must still cover the duplicate; only the arithmetic must not.
_COUNT_FIELDS = ("instruction_clean", "response_clean")

PLACEHOLDER = "[SURNAME]"


def scan(rows: list[dict], fields: tuple[str, ...] = _COUNT_FIELDS
         ) -> collections.Counter:
    """Count surnames, by default without double-counting.

    ``--check`` passes the full field set: for a gate, finding a name in the
    duplicated field still means the file is contaminated, and a missed
    duplicate would let a dirty corpus through.
    """
    hits: collections.Counter = collections.Counter()
    for row in rows:
        for field in fields:
            for name in find_surnames(row.get(field) or ""):
                hits[name.lower()] += 1
    return hits


def affected_records(rows: list[dict]) -> int:
    """Records containing at least one surname — the figure to report.

    Occurrences overstate the reach of a leak when one message names the
    same person four times.
    """
    return sum(
        1 for row in rows
        if any(find_surnames(row.get(f) or "") for f in _COUNT_FIELDS)
    )


def clean(rows: list[dict]) -> tuple[list[dict], int]:
    total = 0
    out: list[dict] = []
    for row in rows:
        new_row = dict(row)
        for field in _TEXT_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                continue
            cleaned, count = redact_surnames(value, PLACEHOLDER)
            if count:
                new_row[field] = cleaned
                total += count
        out.append(new_row)
    return out, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/jarvis_training_data_v4.json")
    ap.add_argument("--out", default="data/jarvis_training_data_v5.json")
    ap.add_argument("--write", action="store_true",
                    help="γράψε το καθαρό αρχείο (χωρίς αυτό, μόνο αναφορά)")
    ap.add_argument("--check", metavar="FILE",
                    help="έξοδος ≠0 αν το αρχείο περιέχει ακόμα επώνυμα")
    args = ap.parse_args()

    load_gazetteer(force=True)
    print(f"  {gazetteer_status()}\n")

    # ── Gate mode ───────────────────────────────────────────────
    if args.check:
        path = ROOT / args.check
        rows = json.load(open(path, encoding="utf-8"))
        hits = scan(rows, _TEXT_FIELDS)
        if hits:
            print(f"✗ {path.name}: {sum(hits.values())} επώνυμα σε "
                  f"{len(hits)} μοναδικές μορφές")
            for name, n in hits.most_common(10):
                print(f"    {n:4}  {name}")
            print("\n  ΜΗΝ εκπαιδεύσεις σε αυτό το αρχείο.")
            sys.exit(1)
        print(f"✓ {path.name}: κανένα επώνυμο")
        sys.exit(0)

    corpus_path = ROOT / args.corpus
    if not corpus_path.exists():
        print(f"δεν βρέθηκε: {corpus_path}")
        sys.exit(1)

    rows = json.load(open(corpus_path, encoding="utf-8"))
    hits = scan(rows)

    affected = affected_records(rows)
    print(f"{corpus_path.name}: {len(rows)} εγγραφές")
    print(f"επώνυμα προς αφαίρεση: {sum(hits.values())} εμφανίσεις σε "
          f"{affected} εγγραφές ({affected / len(rows):.1%}), "
          f"{len(hits)} μοναδικές μορφές")
    print("(μετρημένα σε μοναδικά πεδία — το formatted_prompt επαναλαμβάνει "
          "τα άλλα δύο)\n")
    for name, n in hits.most_common(15):
        print(f"  {n:4}  {name}")
    if len(hits) > 15:
        print(f"  … και {len(hits) - 15} ακόμα")

    if not args.write:
        print("\n(δοκιμή — δεν γράφτηκε τίποτα. Πρόσθεσε --write)")
        return

    cleaned_rows, replaced = clean(rows)

    # Verify on the output, not on the intention. The sanitiser is the thing
    # under test here, and a pass that reports success without re-reading
    # what it wrote is how v4 came to be trusted.
    remaining = scan(cleaned_rows, _TEXT_FIELDS)
    if remaining:
        print(f"\n✗ Μετά τον καθαρισμό απομένουν {sum(remaining.values())}. "
              "Δεν γράφτηκε αρχείο.")
        sys.exit(1)

    out_path = ROOT / args.out
    out_path.write_text(
        json.dumps(cleaned_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n✓ {replaced} αντικαταστάσεις σε όλα τα πεδία → {out_path.name}")
    print(f"  επαλήθευση στο γραμμένο αρχείο: 0 επώνυμα")
    print("\nΕπόμενο: δείξε το training και το JARVIS_CORPUS στο v5.")
    print("Το φίλτρο εξόδου καλύπτει· μόνο η επανεκπαίδευση αφαιρεί.")


if __name__ == "__main__":
    main()
