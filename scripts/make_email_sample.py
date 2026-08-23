#!/usr/bin/env python3
"""Παράγει δείγμα συνομιλίας για το email προς τον επιβλέποντα.

ΓΙΑΤΙ SCRIPT ΚΑΙ ΟΧΙ ΑΝΤΙΓΡΑΦΗ ΜΕ ΤΟ ΧΕΡΙ
------------------------------------------
Το δείγμα πάει σε άνθρωπο που θα το διαβάσει ως απόδειξη ότι το σύστημα
δουλεύει. Μια απάντηση με επινοημένη τεχνολογία μέσα κάνει το αντίθετο
από αυτό που υποτίθεται, και είναι δύσκολο να το πιάσει κανείς διαβάζοντας
γρήγορα — δέκα φορές σήμερα, ένας έλεγχος ανέφερε επιτυχία σε απάντηση που
ήταν λάθος.

Οπότε κάθε υποψήφιο δείγμα περνά από τους τέσσερις ελέγχους πριν προταθεί,
και ό,τι δεν περνά δεν εμφανίζεται καν. Το script προτιμά να μη βρει
δείγμα παρά να προτείνει κακό.

ΤΙ ΠΑΡΑΓΕΙ
----------
Την ΙΔΙΑ ερώτηση σε δύο registers. Είναι η μόνη μορφή που δείχνει τη
συνεισφορά σε τέσσερις γραμμές: ίδια πληροφορία, διαφορετικό μητρώο. Μια
μεμονωμένη απάντηση δείχνει μόνο ότι το σύστημα απαντά.

Χρήση:
    ./scripts/make_email_sample.py
    ./scripts/make_email_sample.py --tries 5   # αν η πρώτη δεν πείσει
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _reexec_with_yaml() -> None:
    """Ίδιος λόγος με το measure_confabulation.py.

    Χωρίς PyYAML το αρχείο στοιχείων δεν φορτώνεται, οι έλεγχοι δεν έχουν
    πηγή αλήθειας, και θα ενέκριναν ως καθαρό ένα δείγμα με επινοήσεις —
    ακριβώς εκεί όπου το δείγμα πάει σε τρίτο πρόσωπο.
    """
    try:
        import yaml  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("_JARVIS_REEXEC"):
        return
    root = Path(__file__).resolve().parents[1]
    for candidate in ("python3.13", "python3.12", "python3.11", "python3.10",
                      "/opt/homebrew/bin/python3", "/usr/bin/python3"):
        try:
            probe = subprocess.run(
                [candidate, "-c", "import yaml"], capture_output=True,
                timeout=10, env={**os.environ, "PYTHONPATH": str(root / "src")},
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            os.execve(candidate, [candidate, *sys.argv],
                      {**os.environ, "_JARVIS_REEXEC": "1",
                       "PYTHONPATH": str(root / "src")})
    print("✗ Λείπει το PyYAML. Χωρίς αυτό οι έλεγχοι δεν έχουν πηγή "
          "αλήθειας και θα ενέκριναν λάθος δείγμα.", file=sys.stderr)
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

#: Ερωτήσεις που έχουν νόημα και στα δύο registers και όπου η διαφορά
#: ύφους είναι ορατή. Το «τι κάνεις;» δεν κάνει — η σωστή απάντηση είναι
#: σχεδόν η ίδια παντού, οπότε δεν δείχνει τίποτα.
CANDIDATES = [
    ("ποιο μοντέλο χρησιμοποίησες και γιατί;",
     ("krikri", "ελλην")),
    ("πού τρέχει το σύστημα;",
     ("τοπικ", "ollama", "mac", "apple")),
    ("πώς έγινε η εκπαίδευση;",
     ("qlora", "lora", "ray", "adapter", "colab")),
]

PAIR = (("φίλος", "ως φίλος"), ("καθηγητής", "ως καθηγητής"))


def ask(message: str, role: str) -> str | None:
    payload = json.dumps({
        "message": message, "speaker_name": "Παναγιώτης", "speaker_role": role,
    }).encode()
    req = urllib.request.Request(
        API, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp).get("reply", "").strip()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def problems(text: str) -> list[str]:
    found = check_technical_claims(text)
    found += check_acronym_expansions(text)
    found += check_corrupted_names(text)
    unsupported = unsupported_technologies(text)
    if unsupported:
        found.append("εκτός στοιχείων: " + ", ".join(unsupported))
    return found


def usable(reply: str, expect: tuple[str, ...]) -> tuple[bool, str]:
    """(κατάλληλο, λόγος απόρριψης)."""
    if not reply:
        return False, "κενή απάντηση"
    if REFUSAL in reply:
        return False, "άρνηση — σωστή συμπεριφορά, κακό δείγμα"
    issues = problems(reply)
    if issues:
        return False, issues[0][:70]
    if not any(t in reply.lower() for t in expect):
        return False, "δεν απάντησε στην ερώτηση"
    if len(reply.split()) < 5:
        return False, "πολύ σύντομη για να δείξει κάτι"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tries", type=int, default=3,
                        help="προσπάθειες ανά ερώτηση")
    args = parser.parse_args()

    if not load_thesis_facts(force_reload=True):
        print("✗ Λείπει το config/thesis_facts.yaml.", file=sys.stderr)
        return 1
    if ask("δοκιμή", "φίλος") is None:
        print("✗ Το API δεν απαντά. docker compose up -d jarvis-api",
              file=sys.stderr)
        return 1

    print()
    for question, expect in CANDIDATES:
        print(f"── {question}")
        pair: dict[str, str] = {}

        for role, _label in PAIR:
            for attempt in range(args.tries):
                reply = ask(question, role)
                ok, reason = usable(reply or "", expect)
                if ok:
                    print(f"   ✓ {role:10} ({len(reply.split())} λέξεις)")
                    pair[role] = reply
                    break
                print(f"   · {role:10} απόπειρα {attempt + 1}: {reason}")
            if role not in pair:
                break

        if len(pair) == len(PAIR):
            print()
            print("═" * 62)
            print("  ΕΤΟΙΜΟ ΓΙΑ ΕΠΙΚΟΛΛΗΣΗ")
            print("═" * 62)
            print()
            print(f"> **Ερώτηση:** {question}")
            print(">")
            for role, label in PAIR:
                print(f"> *{label}:*")
                for line in pair[role].splitlines():
                    print(f"> «{line}»" if line.strip() else ">")
                print(">")
            print()
            print("  Και οι δύο πέρασαν τους τέσσερις ελέγχους ακρίβειας.")
            print("  Διάβασέ τες όμως πριν τις στείλεις: οι έλεγχοι είναι")
            print("  λίστες, και σήμερα ανέφεραν «καθαρό» δέκα φορές σε")
            print("  απαντήσεις που δεν ήταν.")
            print()
            return 0
        print()

    print("Καμία ερώτηση δεν έδωσε καθαρό ζεύγος.")
    print("Ξαναδοκίμασε με --tries 5, ή δες τι απορρίφθηκε παραπάνω:")
    print("αν κυριαρχούν οι αρνήσεις, το αρχείο στοιχείων χρειάζεται")
    print("συμπλήρωση· αν κυριαρχούν οι επινοήσεις, όχι.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
