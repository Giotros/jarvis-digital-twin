#!/usr/bin/env python3
"""Surface likely surnames in the corpus for a human to confirm.

Morphology catches surnames with distinctive endings. It cannot catch the
rest: Ταμπακάς ends in -ας, and so do hundreds of ordinary Greek words. No
rule separates them, so a person has to look.

This script narrows "every word in the corpus" down to a list short enough
to read. It does not decide anything — the output is a shortlist, and
nothing is redacted until a name is written into ``config/surnames.txt``.

The evidence used, in order of usefulness:

**Name-adjacent frames.** "στείλε στον X", "του X", "ο X είπε", "κ. X". A
word that repeatedly appears where a person's name goes is probably a
person's name, even in lower case with no article.

**Absence from ordinary Greek.** A token that occurs in this corpus but
matches no common-word pattern is either a name, a typo, or a loanword.

**Frequency band.** A real acquaintance is mentioned more than once and
fewer times than a function word.

Usage:
    python3 scripts/mine_surnames.py
    python3 scripts/mine_surnames.py --min-count 3 --top 60

Output is printed, never written. Writing the gazetteer is a decision about
identifiable third parties and stays a human action.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jarvis.sanitization.greek_names import find_given_names  # noqa: E402
from jarvis.sanitization.greek_surnames import (  # noqa: E402
    SELF_NAMES,
    is_surname,
    load_gazetteer,
    strip_accents,
)

#: Frames where a personal name is the expected filler.
#:
#: Case-insensitive and accent-folded, because the corpus is casual Greek
#: written without either. That is the whole reason capitalisation-based
#: detection failed on this data in the first place.
_NAME_FRAMES: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"\b(?:στον|στην|στου|στης)\s+([^\W\d_]{4,})",
        r"\b(?:του|της|τον|την)\s+([^\W\d_]{4,})",
        r"\b(?:με|απο|για)\s+τον?\s+([^\W\d_]{4,})",
        # The dot is required. Written as `\bκ\.?` it matched the initial
        # kappa of any word — "κινητή" yielded the candidate "ινητη", 412
        # times, and drowned every real name in the ranking.
        r"\bκ\.\s*([^\W\d_]{4,})",
        r"\b(?:κυριο|κυριε|κυρια)\s+([^\W\d_]{4,})",
        r"\b(?:ο|η)\s+([^\W\d_]{4,})\s+(?:ειπε|λεει|ελεγε|εστειλε|ρωτησε)",
        r"\b(?:στειλε|στειλ|γραψε|ρωτα|ρωτησε|πες)\s+(?:στον|στην|του|της)?\s*([^\W\d_]{4,})",
    )
)

#: Words that fill name frames without being names. Grammatical fillers
#: mostly — "του εαυτού μου", "στην αρχή".
_FRAME_NOISE = frozenset(strip_accents(w) for w in {
    "εαυτο", "εαυτου", "αρχη", "αρχης", "ωρα", "ωρας", "μερα", "μερας",
    "δουλεια", "δουλειας", "σχολη", "σχολης", "σπιτι", "σπιτιου",
    "αλλο", "αλλη", "αλλον", "ιδιο", "ιδια", "καθε", "οποιο", "οποια",
    "πρωτο", "πρωτη", "τελευταιο", "τελευταια", "θεμα", "θεματος",
    "εβδομαδα", "εβδομαδας", "μηνα", "χρονο", "χρονια", "βραδυ",
    "πρωι", "απογευμα", "τηλεφωνο", "μαιλ", "email", "μηνυμα",
    "παιδι", "παιδια", "κοσμο", "κοσμος", "ανθρωπο", "ανθρωπος",
    "εταιρεια", "εταιρειας", "γραφειο", "μαγαζι", "νοσοκομειο",
    "πανεπιστημιο", "τραπεζα", "τραπεζας", "αυτοκινητο", "μηχανη",
})


def load_corpus(path: Path) -> list[str]:
    rows = json.load(open(path, encoding="utf-8"))
    return [
        (row.get(field) or "")
        for row in rows
        for field in ("instruction", "response_clean", "response")
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/jarvis_training_data_v4.json")
    ap.add_argument("--min-count", type=int, default=2,
                    help="ignore words seen fewer times than this")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    corpus_path = ROOT / args.corpus
    if not corpus_path.exists():
        print(f"δεν βρέθηκε το corpus: {corpus_path}")
        sys.exit(1)

    texts = load_corpus(corpus_path)
    print(f"{len(texts)} πεδία κειμένου από {corpus_path.name}\n")

    known = load_gazetteer()
    frame_hits: collections.Counter = collections.Counter()
    total: collections.Counter = collections.Counter()

    word_re = re.compile(r"[^\W\d_]+", re.UNICODE)
    for text in texts:
        for word in word_re.findall(text):
            total[strip_accents(word)] += 1
        # Counted once per position, not once per pattern. Several frames
        # overlap ("του X" is inside "με τον X"), and counting each match
        # separately produced ratios above 100% — a number that cannot mean
        # anything and undermines the column it sits in.
        seen_spans: set[tuple[int, int]] = set()
        for pattern in _NAME_FRAMES:
            for match in pattern.finditer(text):
                span = match.span(1)
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                frame_hits[strip_accents(match.group(1))] += 1

    candidates = []
    for word, frames in frame_hits.items():
        if frames < args.min_count:
            continue
        if word in _FRAME_NOISE:
            continue
        # His own name fills name frames more than anyone else's. Surfacing
        # it as a candidate invites the one edit that would make the twin
        # unable to say who it is.
        if word in SELF_NAMES:
            continue
        # Already handled by one of the two existing detectors.
        if is_surname(word) or find_given_names(word):
            continue
        if any(word.startswith(stem) for stem in known):
            continue
        # A word that appears mostly inside name frames is a better
        # candidate than one that happens to be common everywhere.
        occurrences = total[word] or 1
        ratio = frames / occurrences
        candidates.append((frames, ratio, word, occurrences))

    candidates.sort(key=lambda c: (c[1], c[0]), reverse=True)

    print("Υποψήφια επώνυμα — ΔΕΝ έχει σβηστεί τίποτα.")
    print("Γράψε όσα είναι πραγματικά ονόματα στο config/surnames.txt.\n")
    print(f"  {'σε πλαίσιο':>10}  {'σύνολο':>7}  {'αναλογία':>8}  λέξη")
    print("  " + "─" * 46)
    for frames, ratio, word, occurrences in candidates[:args.top]:
        print(f"  {frames:>10}  {occurrences:>7}  {ratio:>7.0%}  {word}")

    if not candidates:
        print("  (κανένα νέο υποψήφιο)")

    print(f"\n{len(candidates)} υποψήφια συνολικά, {len(known)} ήδη επιβεβαιωμένα.")
    print("\nΤο αρχείο config/surnames.txt είναι στο .gitignore — είναι")
    print("λίστα ταυτοποιήσιμων τρίτων, ακριβώς ό,τι αφαιρεί η ροή.")


if __name__ == "__main__":
    main()
