#!/usr/bin/env python3
"""Μετράει πόσο συχνά το σύστημα επινοεί τεχνικά στοιχεία.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ
-------------
Επτά κατηγορίες σφάλματος βρέθηκαν σε μία ημέρα, καθεμία **αφού** είχε
διορθωθεί η προηγούμενη:

  1. γνωστές επινοήσεις εργαλείων      (Kubernetes, Django, MongoDB)
  2. άγνωστες επινοήσεις εργαλείων     (Rust, WebAssembly, actix-web)
  3. ισχυρισμοί ικανοτήτων             («μαθαίνει από τις αλληλεπιδράσεις»)
  4. ψευδοακρίβεια αριθμών             («75% του κώδικα»)
  5. λάθος αναπτύγματα ακρωνυμίων      («PEFT = PyTorch Elastic Framework»)
  6. παραφθαρμένα ονόματα              («ChromeDB»)
  7. ισχυρισμοί κλίμακας και πλαισίου  («GPU clusters», «ο server μας»)

Κάθε φορά, οι υπάρχοντες έλεγχοι ανέφεραν καθαρό αποτέλεσμα και το σφάλμα
το εντόπισε άνθρωπος διαβάζοντας την έξοδο. Η ακολουθία είναι το ίδιο
φαινόμενο με το 8,1% → 14,2% του κεφαλαίου 4: **δεν γνωρίζεις την ανάκλησή
σου**, ούτε στα δεδομένα ούτε στους ελέγχους.

Το εύλογο επόμενο βήμα δεν είναι όγδοη κατηγορία μοτίβων. Είναι να πάψει η
συζήτηση να στηρίζεται στο τελευταίο ✗ που έτυχε να δει κάποιος και να
αποκτήσει **αριθμό**: σε τι ποσοστό απαντήσεων, με πόση διασπορά, και ποιος
έλεγχος τον πιάνει. Αυτός ο αριθμός είναι αποτέλεσμα της εργασίας — και
είναι, από όσο ξέρω, το είδος μέτρησης που ούτε το WeClone ούτε το
PersonaTwin αναφέρουν.

ΤΙ ΜΕΤΡΑΕΙ
----------
ΔΥΟ ανεξάρτητοι άξονες, όχι ένας:

  ακρίβεια  — clean | confabulated | refusal
  κάλυψη    — απάντησε στην ερώτηση, ναι ή όχι

Η πρώτη έκδοση τους συγχώνευε, και το αποτέλεσμα ήταν ότι μια απάντηση με
«τρέχει σε 3 διαφορετικούς servers … συλλογή από twitter» καταγράφηκε ως
«χωρίς περιεχόμενο» και η αναφορά έδειξε **0,0% επινόηση**. Το εργαλείο
μέτρησης της επινόησης ανέφερε μηδέν επινόηση, με τον ίδιο ακριβώς
μηχανισμό που περιγράφει η ενότητα από πάνω. Αυτή είναι η όγδοη εμφάνιση,
και η πρώτη που παρήχθη από κώδικα γραμμένο για να την αποτρέψει.

Χρήση:
    ./scripts/measure_confabulation.py            # 3 επαναλήψεις
    ./scripts/measure_confabulation.py --runs 10  # για το κεφάλαιο 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _reexec_with_yaml() -> None:
    """Restart under an interpreter that has PyYAML, or explain and stop.

    macOS ships several Pythons and ``python3`` on PATH is rarely the one
    with the project's packages. This exact gap already cost a measurement
    once: the ablation script ran without PyYAML, both models came out
    ungrounded, and the run looked like a *result* — the facts file had
    silently failed to load and nothing said so.

    So: find an interpreter that works, or refuse. Never measure with no
    source of truth, because a check with nothing to compare against reports
    zero problems and zero is indistinguishable from success.
    """
    try:
        import yaml  # noqa: F401
        return
    except ImportError:
        pass

    if os.environ.get("_JARVIS_REEXEC"):  # already tried; do not loop
        return

    root = Path(__file__).resolve().parents[1]
    candidates = [
        "python3.13", "python3.12", "python3.11", "python3.10",
        "/opt/homebrew/bin/python3", "/usr/local/bin/python3",
        "/usr/bin/python3",
    ]
    for candidate in candidates:
        try:
            probe = subprocess.run(
                [candidate, "-c", "import yaml"],
                capture_output=True, timeout=10,
                env={**os.environ, "PYTHONPATH": str(root / "src")},
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            print(f"  (χρησιμοποιώ {candidate} — το PyYAML λείπει από το "
                  f"{sys.executable})")
            os.execve(
                candidate, [candidate, *sys.argv],
                {**os.environ, "_JARVIS_REEXEC": "1",
                 "PYTHONPATH": str(root / "src")},
            )

    print("✗ Κανένας διερμηνέας δεν έχει PyYAML. Χωρίς αυτό το αρχείο\n"
          "  στοιχείων δεν φορτώνεται, ο έλεγχος δεν έχει πηγή αλήθειας,\n"
          "  και η μέτρηση θα ανέφερε ψευδώς μηδέν επινοήσεις.\n\n"
          f"      {sys.executable} -m pip install pyyaml\n",
          file=sys.stderr)
    raise SystemExit(1)


_reexec_with_yaml()

import urllib.error
import urllib.request

from jarvis.inference.thesis_facts import (  # noqa: E402
    check_acronym_expansions,
    check_corrupted_names,
    check_technical_claims,
    load_thesis_facts,
    unsupported_technologies,
)

API = "http://localhost:8000/orchestration/generate"
REFUSAL = "Δεν θέλω να πω κάτι λάθος"

#: Ερωτήσεις που θα κάνει πραγματικά ένας εξεταστής. Επίτηδες ανομοιογενείς:
#: η στενή («ποιο μοντέλο») είναι εύκολη, η ανοιχτή («τι έκανες φέτος») είναι
#: αυτή που παρήγαγε κάθε μία από τις επτά κατηγορίες.
QUESTIONS = [
    "με τι τεχνολογίες δούλεψες φέτος;",
    "ποιο μοντέλο χρησιμοποίησες και γιατί;",
    "πώς έγινε η εκπαίδευση;",
    "πώς αντιμετώπισες τα προσωπικά δεδομένα;",
    "τι είναι το RAG στο σύστημά σου;",
    "πες μου για τη διπλωματική σου",
    "ποιοι είναι οι περιορισμοί της δουλειάς σου;",
    "πού τρέχει το σύστημα;",
]

ROLES = {"φίλος": "close", "συνάδελφος": "professional", "καθηγητής": "academic"}

#: Όροι που δείχνουν ότι η απάντηση **αφορά** την ερώτηση, ανά ερώτηση.
#:
#: Η πρώτη έκδοση χρησιμοποιούσε μία ενιαία λίστα ονομάτων εργαλείων για
#: όλες τις ερωτήσεις. Οι μισές ερωτήσεις δεν αφορούν εργαλεία: η σωστή
#: απάντηση στο «πώς αντιμετώπισες τα προσωπικά δεδομένα» δεν περιέχει
#: κανένα όνομα εργαλείου, και η σωστή απάντηση «Krikri-8B (ΙΕΛ/Αθηνά),
#: είναι ελληνικό» περιέχει ένα. Και οι δύο βαθμολογήθηκαν «χωρίς
#: περιεχόμενο», δηλαδή η μετρική τιμώρησε τις σωστές απαντήσεις.
#:
#: Το σφάλμα είναι το ίδιο που το εργαλείο υπάρχει για να μετρήσει, και
#: κατέγραψα να το κάνω ενώ το έγραφα.
EXPECTED: dict[str, tuple[str, ...]] = {
    "με τι τεχνολογίες δούλεψες φέτος;": (
        "python", "ray", "pytorch", "ollama", "n8n", "docker", "fastapi"),
    "ποιο μοντέλο χρησιμοποίησες και γιατί;": (
        "krikri", "8b", "ιελ", "ilsp", "ελλην", "tokeniz", "mistral"),
    "πώς έγινε η εκπαίδευση;": (
        "qlora", "lora", "4-bit", "4 bit", "ray", "colab", "adapter",
        "fine-tun", "epoch", "checkpoint"),
    "πώς αντιμετώπισες τα προσωπικά δεδομένα;": (
        "ανωνυμ", "τηλέφων", "email", "ονόματ", "ονοματ", "επώνυμ",
        "αφαίρ", "αφαιρ", "gdpr", "προσωπικ", "σβήσ", "καθάρ"),
    "τι είναι το RAG στο σύστημά σου;": (
        "ανάκτησ", "ανακτησ", "αναζήτησ", "bm25", "embedding", "chromadb",
        "corpus", "συνομιλ", "retrieval"),
    # «ψηφιακ», «εξατομικευ», «γλωσσικ» προστέθηκαν αφού η μετρική σήμανε
    # ως εκτός θέματος δύο απαντήσεις που απήγγειλαν τον ΤΙΤΛΟ της
    # εργασίας: «ανάπτυξη ενός αυτόνομου ψηφιακού διδύμου με εξατομικευμένη
    # επικοινωνία μέσω μεγάλων ανοιχτών γλωσσικών μοντέλων». Δεύτερη φορά σε
    # δύο εκδόσεις που η ίδια μετρική τιμωρεί τη σωστή απάντηση.
    "πες μου για τη διπλωματική σου": (
        "δίδυμο", "διδυμο", "twin", "ύφος", "υφος", "μηνύματ", "μηνυματ",
        "krikri", "συνομιλ", "ψηφιακ", "εξατομικευ", "γλωσσικ"),
    "ποιοι είναι οι περιορισμοί της δουλειάς σου;": (
        "epoch", "quota", "gpu", "δεν ", "περιορισ", "μικρ", "ένα ", "μόνο"),
    "πού τρέχει το σύστημα;": (
        "τοπικ", "mac", "ollama", "apple", "silicon", "metal", "docker"),
}

#: Πόσοι όροι αρκούν για να θεωρηθεί ότι η απάντηση αφορά την ερώτηση.
COVERAGE_THRESHOLD = 1


def ask(message: str, role: str, timeout: float = 120.0) -> str | None:
    """Μία κλήση στο /generate. None αν το API δεν απαντά."""
    payload = json.dumps({
        "message": message,
        "speaker_name": "Παναγιώτης",
        "speaker_role": role,
    }).encode()
    req = urllib.request.Request(
        API, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp).get("reply", "")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def classify(reply: str, question: str) -> dict:
    """Δύο ΑΝΕΞΑΡΤΗΤΟΙ άξονες: ακρίβεια και κάλυψη.

    Η πρώτη έκδοση τους συγχώνευε σε έναν, με σειρά ελέγχου
    ``confabulated → empty → clean``. Το αποτέλεσμα ήταν ότι μια απάντηση
    που έλεγε *«τρέχει σε 3 διαφορετικούς servers … συλλογή από twitter»*
    καταγράφηκε ως **«χωρίς περιεχόμενο»** — και η συγκεντρωτική αναφορά
    έδειξε **0,0% επινόηση**.

    Ένα σφάλμα ήταν ότι κανένα μοτίβο δεν έπιανε αυτούς τους ισχυρισμούς·
    το δεύτερο, βαθύτερο, ότι «δεν απάντησε» και «είπε ψέματα» δεν είναι
    εναλλακτικές. Μια απάντηση μπορεί να είναι και τα δύο, και σε αυτή την
    περίπτωση ήταν. Δύο ιδιότητες σε έναν άξονα σημαίνει ότι η μία κρύβει
    την άλλη.

    Το ότι το εργαλείο μέτρησης της επινόησης ανέφερε μηδέν επινόηση, με
    τον ίδιο μηχανισμό που περιγράφει, δεν είναι ειρωνεία — είναι το
    ισχυρότερο τεκμήριο για το ότι η κατηγορία σφάλματος είναι δομική.
    """
    checks: list[tuple[str, list[str]]] = [
        ("denylist", check_technical_claims(reply)),
        ("acronyms", check_acronym_expansions(reply)),
        ("near_miss", check_corrupted_names(reply)),
    ]
    unsupported = unsupported_technologies(reply)
    if unsupported:
        checks.append(("allowlist", [f"εκτός στοιχείων: {', '.join(unsupported)}"]))

    problems = [msg for _, msgs in checks for msg in msgs]
    fired = [name for name, msgs in checks if msgs]

    lowered = reply.lower()
    expected = EXPECTED.get(question, ())
    hits = [t for t in expected if t in lowered]
    refused = REFUSAL in reply

    if refused:
        accuracy = "refusal"
    elif problems:
        accuracy = "confabulated"
    else:
        accuracy = "clean"

    return {
        "accuracy": accuracy,
        "answered": (not refused) and len(hits) >= COVERAGE_THRESHOLD,
        "problems": problems,
        "checks": fired,
        "hits": hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3,
                        help="επαναλήψεις ανά ερώτηση και register")
    parser.add_argument("--out", default="docs/confabulation_rate.json")
    args = parser.parse_args()

    if not load_thesis_facts(force_reload=True):
        print("✗ Δεν βρέθηκε το config/thesis_facts.yaml — χωρίς αυτό ο "
              "έλεγχος δεν έχει πηγή αλήθειας και θα ανέφερε ψευδώς μηδέν.",
              file=sys.stderr)
        return 1

    if ask("δοκιμή", "φίλος", timeout=10) is None:
        print("✗ Το API στο localhost:8000 δεν απαντά. "
              "docker compose up -d jarvis-api", file=sys.stderr)
        return 1

    total = len(QUESTIONS) * len(ROLES) * args.runs
    print(f"\n{total} κλήσεις "
          f"({len(QUESTIONS)} ερωτήσεις × {len(ROLES)} registers × "
          f"{args.runs} επαναλήψεις)\n")

    records: list[dict] = []
    done = 0
    for role, register in ROLES.items():
        for question in QUESTIONS:
            for run in range(args.runs):
                reply = ask(question, role)
                done += 1
                if reply is None:
                    print(f"  [{done}/{total}] ✗ το API σταμάτησε να απαντά")
                    continue
                verdict = classify(reply, question)
                records.append({
                    "register": register, "question": question, "run": run,
                    **verdict, "words": len(reply.split()), "reply": reply,
                })
                mark = {"clean": "✓", "confabulated": "✗",
                        "refusal": "⚠"}[verdict["accuracy"]]
                cover = "" if verdict["answered"] else "  (εκτός θέματος)"
                print(f"  [{done}/{total}] {mark} {register:12} "
                      f"{question[:36]}{cover}")
                for p in verdict["problems"]:
                    print(f"              → {p[:72]}")

    if not records:
        print("\n✗ Καμία απάντηση.", file=sys.stderr)
        return 1

    counts = Counter(r["accuracy"] for r in records)
    n = len(records)
    answered = sum(1 for r in records if r["answered"])
    by_check = Counter(c for r in records for c in r["checks"])
    by_register = {
        reg: Counter(r["accuracy"] for r in records if r["register"] == reg)
        for reg in ROLES.values()
    }

    print("\n" + "═" * 60)
    print(f"  {n} απαντήσεις\n")
    print("  Ακρίβεια — ΚΑΤΩ ΦΡΑΓΜΑ, όχι εκτίμηση:")
    for status, label in (("clean", "χωρίς εντοπισμένο λάθος"),
                          ("confabulated", "με επινόηση"),
                          ("refusal", "αρνήσεις")):
        c = counts[status]
        print(f"    {label:26} {c:3}  ({100 * c / n:.1f}%)")
    print("\n  Κάλυψη (ανεξάρτητος άξονας):")
    print(f"    {'απάντησαν στην ερώτηση':26} {answered:3}  "
          f"({100 * answered / n:.1f}%)")

    # Ο συνδυασμός είναι το νούμερο που έχει σημασία: χρήσιμη ΚΑΙ σωστή.
    useful = sum(1 for r in records
                 if r["answered"] and r["accuracy"] == "clean")
    both_bad = sum(1 for r in records
                   if not r["answered"] and r["accuracy"] == "confabulated")
    print(f"\n    {'χρήσιμες και σωστές':26} {useful:3}  "
          f"({100 * useful / n:.1f}%)")
    print(f"    {'εκτός θέματος ΚΑΙ λάθος':26} {both_bad:3}  "
          f"({100 * both_bad / n:.1f}%)")
    print("      ↑ η δεύτερη γραμμή ήταν αόρατη όσο οι δύο άξονες ήταν ένας")

    print("\n  Ποιος έλεγχος πιάνει τι (συνολικές πυροδοτήσεις):")
    if by_check:
        for name, c in by_check.most_common():
            print(f"    {name:20} {c:3}")
        # Ο πιο χρήσιμος αριθμός εδώ: πόσα θα είχαν διαφύγει αν υπήρχε μόνο
        # ο αρχικός denylist. Είναι η ποσοτικοποίηση του «κάθε νέος έλεγχος
        # βρήκε κατηγορία που οι προηγούμενοι ανέφεραν καθαρή».
        only_denylist = sum(
            1 for r in records
            if r["checks"] and "denylist" not in r["checks"]
        )
        print(f"\n    Θα διέφευγαν με μόνο τον denylist: {only_denylist}")
    else:
        print("    καμία")

    print("\n  Ανά register:")
    for reg, c in by_register.items():
        m = sum(c.values()) or 1
        print(f"    {reg:14} καθαρές {100 * c['clean'] / m:5.1f}%   "
              f"επινόηση {100 * c['confabulated'] / m:5.1f}%   "
              f"άρνηση {100 * c['refusal'] / m:5.1f}%")

    if args.runs > 1:
        # Η διασπορά έχει σημασία όσο και ο μέσος όρος: αν το ίδιο ερώτημα
        # άλλοτε περνά και άλλοτε όχι, ένα μεμονωμένο καθαρό τρέξιμο δεν
        # σημαίνει τίποτα — και σε αυτό το έργο ένα καθαρό τρέξιμο έχει ήδη
        # τρεις φορές αναφέρει επιτυχία που δεν υπήρχε.
        per_q = []
        for reg in ROLES.values():
            for q in QUESTIONS:
                rs = [r for r in records
                      if r["register"] == reg and r["question"] == q]
                if len(rs) > 1:
                    per_q.append(
                        sum(1 for r in rs
                            if r["accuracy"] == "clean" and r["answered"])
                        / len(rs)
                    )
        if per_q:
            unstable = sum(1 for p in per_q if 0 < p < 1)
            print(f"\n  Αστάθεια: {unstable} από {len(per_q)} συνδυασμούς "
                  f"άλλοτε καθαροί και άλλοτε όχι")
            print(f"  Τυπική απόκλιση καθαρότητας: "
                  f"{statistics.pstdev(per_q):.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "date": date.today().isoformat(),
        "runs_per_cell": args.runs,
        "n": n,
        "counts": dict(counts),
        "answered": answered,
        "useful": useful,
        "by_check": dict(by_check),
        "by_register": {k: dict(v) for k, v in by_register.items()},
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n  ⚠ ΠΩΣ ΔΙΑΒΑΖΕΤΑΙ ΑΥΤΟΣ Ο ΑΡΙΘΜΟΣ")
    print("    Το «με επινόηση» είναι ΚΑΤΩ ΦΡΑΓΜΑ. Μετράει όσα ξέρουν να")
    print("    αναγνωρίσουν οι έλεγχοι, και οι έλεγχοι είναι λίστες.")
    print("    Στην εργασία αυτή, ποσοστό 0,0% έχει αναφερθεί ΤΡΕΙΣ φορές")
    print("    και ήταν λάθος και τις τρεις — τα σφάλματα βρέθηκαν από")
    print("    ανάγνωση της εξόδου, όχι από το εργαλείο.")
    print("    Διάβασε τα records του JSON πριν αναφέρεις οποιοδήποτε")
    print("    ποσοστό σε κείμενο που θα κριθεί.")
    print(f"\n  Γράφτηκε: {out}")
    print("═" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
