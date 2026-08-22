#!/bin/bash
# ============================================================
# Ablation study: ίδιο base μοντέλο, με και χωρίς τον LoRA adapter.
#
# Απαντά με νούμερα στην ερώτηση που θα κάνει η επιτροπή:
# «τι σου έδωσε και τι σου κόστισε το fine-tuning;»
#
# Μετρώνται δύο πράγματα σε αντίθετες κατευθύνσεις:
#   ύφος        — πόσο μοιάζει με τα golden examples του Γιώργου
#   ακρίβεια    — πόσοι τεχνικοί ισχυρισμοί αντιφάσκουν με το project
#
# Η υπόθεση είναι ότι ο adapter κερδίζει στο πρώτο και χάνει στο
# δεύτερο. Αν βγει αλλιώς, αυτό είναι το εύρημα.
#
# Προϋπόθεση: ./scripts/setup_base_model.sh
#
# Χρήση:  ./scripts/compare_adapter.sh [--out docs]
# ============================================================

cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
OLLAMA="http://localhost:11434"
OUT="${2:-docs}"

echo ""
echo -e "${BLUE}═══ Ablation: με και χωρίς adapter ═══${NC}"
echo ""

for m in jarvis jarvis-base; do
    if ! curl -s --max-time 3 "$OLLAMA/api/tags" | grep -q "\"$m"; then
        echo -e "  ${RED}✗${NC} λείπει το μοντέλο '$m'"
        [[ "$m" == "jarvis-base" ]] && echo "    Τρέξε: ./scripts/setup_base_model.sh"
        exit 1
    fi
done
echo -e "  ${GREEN}✓${NC} και τα δύο μοντέλα διαθέσιμα"

# ── Εύρεση κατάλληλου διερμηνέα ─────────────────────────────
#
# Χωρίς PyYAML δεν φορτώνονται ούτε τα στοιχεία της διπλωματικής ούτε τα
# golden examples, και τα δύο μοντέλα τρέχουν αθεμελίωτα. Η πρώτη εκτέλεση
# αυτού του script το έκανε ακριβώς αυτό: το ανέφερε σε μία γραμμή, συνέχισε,
# και παρήγαγε πίνακα που έμοιαζε έγκυρος. Μια σύγκριση που μετράει κάτι
# άλλο από αυτό που δηλώνει είναι χειρότερη από καμία σύγκριση.
#
# Σε macOS συνυπάρχουν πολλές Python. Το PyYAML εγκαταστάθηκε σε μία και το
# `python3` του PATH δείχνει σε άλλη, οπότε ένας έλεγχος `python3 -c "import
# yaml"` απαντά «λείπει» ενώ το πακέτο υπάρχει — και η οδηγία εγκατάστασης
# απαντά «already satisfied». Ατέρμονος βρόχος για τον χρήστη.
#
# Ελέγχονται και τα δύο μαζί: yaml ΚΑΙ το πακέτο jarvis. Ένας διερμηνέας που
# έχει το ένα αλλά όχι το άλλο δεν εξυπηρετεί.
PY=""
for candidate in python3 python3.13 python3.12 python3.11 python3.10 \
                 /opt/homebrew/bin/python3 /usr/local/bin/python3 \
                 /usr/bin/python3 \
                 /Library/Developer/CommandLineTools/usr/bin/python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    # Δοκιμάζονται όλα όσα χρειάζεται το script, όχι μόνο το yaml. Ένας
    # διερμηνέας που έχει το yaml αλλά δεν φορτώνει το πακέτο jarvis θα
    # περνούσε τον έλεγχο και θα έσκαγε δύο γραμμές πιο κάτω.
    if PYTHONPATH=src "$candidate" -c "import yaml
import jarvis.inference.thesis_facts
import jarvis.orchestration.persona
import jarvis.evaluation.metrics" 2>/dev/null; then
        PY="$candidate"
        break
    fi
done

if [[ -z "$PY" ]]; then
    echo ""
    echo -e "  ${RED}✗ Δεν βρέθηκε Python με PyYAML.${NC}"
    echo "    Χωρίς αυτό δεν φορτώνονται τα στοιχεία της διπλωματικής ούτε"
    echo "    τα golden examples, και η σύγκριση μετράει δύο αθεμελίωτα μοντέλα."
    echo ""
    echo "    Δοκίμασε, με τη Python που θα τρέξει το script:"
    echo -e "      ${BLUE}$(command -v python3) -m pip install pyyaml${NC}"
    echo ""
    exit 1
fi
echo -e "  ${GREEN}✓${NC} PyYAML διαθέσιμο ($("$PY" -V 2>&1) — $PY)"

PYTHONPATH=src "$PY" - "$OUT" << 'PYEOF'
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")
from jarvis.evaluation.metrics import style_profile, style_distance
from jarvis.inference.thesis_facts import check_technical_claims
from jarvis.orchestration.persona import ACADEMIC, CLOSE, build_system_prompt

OLLAMA = "http://localhost:11434/api/chat"
OUT = Path(sys.argv[1])

# Δύο ομάδες ερωτήσεων, γιατί το ζητούμενο διαφέρει.
# Στις τεχνικές μετράει η ακρίβεια· στις προσωπικές το ύφος.
# Κάθε τεχνική ερώτηση συνοδεύεται από τους όρους που ΠΡΕΠΕΙ να εμφανιστούν.
#
# Ο αριθμός αντιφάσεων από μόνος του είναι μετρική που κερδίζεται με σιωπή:
# στην πρώτη εκτέλεση το μοντέλο με adapter βγήκε 0 αντιφάσεις απαντώντας
# «εχεις καποιο demo?» και «η προσαρμογη στην εξεταση». Μηδέν λάθη επειδή
# μηδέν περιεχόμενο. Χρειάζεται δεύτερη μετρική που τιμωρεί τη μη-απάντηση,
# αλλιώς το τεστ ανταμείβει το να αποφεύγεις την ερώτηση.
TECHNICAL: list[tuple[str, list[str]]] = [
    ("με τι τεχνολογιες δουλεψες;",
     ["krikri", "qlora", "ray"]),
    ("γιατι διαλεξες αυτο το μοντελο;",
     ["ελλην", "krikri", "token"]),
    ("πως αντιμετωπισες το θεμα των προσωπικων δεδομενων;",
     ["ανωνυμ", "gdpr", "ονοματ", "δεδομεν"]),
    ("τι ακριβως κανει το RAG κομματι;",
     ["ανακτ", "αναζητ", "bm25", "embedding", "συνομιλ"]),
    ("ποια ηταν η μεγαλυτερη δυσκολια;",
     ["ανωνυμ", "ονοματ", "εκπαιδευ", "gpu", "ελλην"]),
]
PERSONAL = [
    "τι κανεις;",
    "θα ερθεις το σαββατο;",
    "πως πας με τη σχολη;",
    "τι εγινε χθες;",
]


def ask(model: str, message: str, system: str, max_tokens: int) -> tuple[str, float]:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        OLLAMA, data=payload, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data.get("message", {}).get("content", ""), time.perf_counter() - started
    except Exception as exc:
        print(f"    ! {type(exc).__name__}: {exc}")
        return "", time.perf_counter() - started


def reference_style():
    """Ο Γιώργος όπως γράφει πραγματικά, από τα golden examples."""
    path = Path("config/golden_examples.yaml")
    if not path.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    responses = [
        (e.get("response") or "").strip()
        for e in data.get("examples", [])
        if (e.get("response") or "").strip()
    ]
    return style_profile(responses) if responses else None


def fold(text: str) -> str:
    import unicodedata
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def coverage(reply: str, expected: list[str]) -> float:
    """Πόσοι από τους αναμενόμενους όρους εμφανίστηκαν.

    Δεν κρίνει αν η απάντηση είναι σωστή — κρίνει αν είναι *απάντηση*.
    Μαζί με τις αντιφάσεις σχηματίζουν ζεύγος: το ένα τιμωρεί το λάθος,
    το άλλο τη σιωπή. Καμία από τις δύο δεν στέκει μόνη της.
    """
    if not reply.strip():
        return 0.0
    folded = fold(reply)
    hits = sum(1 for term in expected if fold(term) in folded)
    return hits / len(expected) if expected else 0.0


REF = reference_style()
academic_prompt, _ = build_system_prompt("Παναγιώτης", "καθηγητής")
close_prompt, _ = build_system_prompt("Νίκος", "φίλος")

if "Krikri" not in academic_prompt:
    print("\n  ✗ Τα στοιχεία της διπλωματικής ΔΕΝ φορτώθηκαν στο prompt.")
    print("    Η σύγκριση θα μετρούσε δύο αθεμελίωτα μοντέλα. Διακοπή.")
    sys.exit(1)

results: dict[str, dict] = {}

for model in ("jarvis", "jarvis-base"):
    print(f"\n── {model}")
    tech_replies, tech_issues, latencies, coverages = [], 0, [], []
    for q, expected in TECHNICAL:
        reply, secs = ask(model, q, academic_prompt, ACADEMIC.max_new_tokens)
        issues = check_technical_claims(reply)
        cov = coverage(reply, expected)
        tech_replies.append(reply)
        tech_issues += len(issues)
        coverages.append(cov)
        latencies.append(secs)
        mark = "✗" if issues else ("○" if cov == 0 else "✓")
        print(f"  {mark} [{secs:4.1f}s] κάλυψη {cov:.0%} — {q}")
        print(f"      {reply[:110] or '(κενή)'}")
        for i in issues:
            print(f"        → {i}")

    personal_replies = []
    for q in PERSONAL:
        reply, secs = ask(model, q, close_prompt, CLOSE.max_new_tokens)
        personal_replies.append(reply)
        latencies.append(secs)

    non_empty = [r for r in personal_replies if r.strip()]
    dist = None
    if REF and non_empty:
        dist = style_distance(style_profile(non_empty), REF)

    results[model] = {
        "technical_contradictions": tech_issues,
        "technical_probes": len(TECHNICAL),
        "mean_coverage": statistics.mean(coverages) if coverages else None,
        "non_answers": sum(1 for c in coverages if c == 0),
        "style_distance": dist,
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "mean_words_technical": (
            statistics.mean(len(r.split()) for r in tech_replies if r.strip())
            if any(r.strip() for r in tech_replies) else None
        ),
    }

# ── Πίνακας ─────────────────────────────────────────────────
def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"

a, b = results["jarvis"], results["jarvis-base"]
n = a["technical_probes"]
table = "\n".join([
    "| Μετρική | Με adapter | Χωρίς adapter | Κατεύθυνση |",
    "|---|---:|---:|:---|",
    f"| Κάλυψη αναμενόμενων όρων | {fmt(a['mean_coverage'] and a['mean_coverage']*100, 0)}% | "
    f"{fmt(b['mean_coverage'] and b['mean_coverage']*100, 0)}% | ↑ καλύτερο |",
    f"| Μη-απαντήσεις (μηδενική κάλυψη) | {a['non_answers']}/{n} | "
    f"{b['non_answers']}/{n} | ↓ καλύτερο |",
    f"| Αντιφάσεις σε {n} τεχνικές ερωτήσεις | "
    f"{a['technical_contradictions']} | {b['technical_contradictions']} | ↓ καλύτερο |",
    f"| Απόσταση ύφους από golden examples | {fmt(a['style_distance'], 3)} | "
    f"{fmt(b['style_distance'], 3)} | ↓ καλύτερο |",
    f"| Μέσο μήκος τεχνικής απάντησης (λέξεις) | {fmt(a['mean_words_technical'], 1)} | "
    f"{fmt(b['mean_words_technical'], 1)} | — |",
    f"| Μέσος χρόνος (s) | {fmt(a['mean_latency_s'], 2)} | {fmt(b['mean_latency_s'], 2)} | ↓ καλύτερο |",
])

print("\n" + "=" * 62)
print(table)
print("=" * 62)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ablation_adapter.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / "ablation_adapter.md").write_text("\n".join([
    "# Ablation: ο ρόλος του LoRA adapter",
    "",
    "Ίδιο base μοντέλο (Krikri-8B Q4_K_M), ίδια prompts, ίδιες παράμετροι.",
    "Η μόνη διαφορά είναι η directive `ADAPTER` στο Modelfile.",
    "",
    table,
    "",
    "## Πώς διαβάζεται",
    "",
    "Οι δύο πρώτες γραμμές πρέπει να διαβάζονται μαζί. Ο αριθμός αντιφάσεων",
    "από μόνος του είναι μετρική που κερδίζεται με σιωπή: σε πρώιμη εκτέλεση",
    "το μοντέλο με adapter βγήκε μηδέν αντιφάσεις απαντώντας «εχεις καποιο",
    "demo?» σε ερώτηση για τις τεχνολογίες. Μηδέν λάθη επειδή μηδέν",
    "περιεχόμενο. Η κάλυψη τιμωρεί ακριβώς αυτό.",
    "",
    "Οι δύο στήλες δεν έχουν νικητή. Ο adapter είναι ο λόγος που το σύστημα",
    "ακούγεται σαν συγκεκριμένο άτομο· είναι επίσης ο λόγος που δυσκολεύεται",
    "να ακολουθήσει οδηγία ύφους, γιατί 13.289 παραδείγματα εκπαίδευσης",
    "βαραίνουν περισσότερο από μία πρόταση system prompt.",
    "",
    "Το συμπέρασμα δεν είναι «κρατάμε το ένα». Είναι ότι η επιλογή εξαρτάται",
    "από το ποιος ρωτάει — που είναι ακριβώς ο λόγος ύπαρξης των registers.",
    "",
    "## Περιορισμοί",
    "",
    "Η κάλυψη μετράει αν εμφανίστηκαν οι σωστοί όροι, όχι αν ειπώθηκαν σωστά.",
    "Μια απάντηση που αναφέρει «Krikri» σε λάθος πρόταση βαθμολογείται ίδια",
    "με μία που το εξηγεί. Οι αντιφάσεις πιάνονται από λίστα παρατηρημένων",
    "σφαλμάτων, άρα εξ ορισμού δεν πιάνουν ό,τι δεν έχει ξανασυμβεί —",
    "το «RAG (Retrovirus Activation Gene)» πέρασε ανέγγιχτο.",
    "Και οι δύο είναι προσεγγιστές· η ανθρώπινη κρίση δεν αντικαθίσταται.",
], ), encoding="utf-8")

print(f"\n→ {OUT/'ablation_adapter.md'}")
print(f"→ {OUT/'ablation_adapter.json'}")
PYEOF
echo ""
