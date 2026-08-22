#!/bin/bash
# ============================================================
# Έλεγχος ότι όλη η στοίβα είναι σωστά ρυθμισμένη και ζωντανή.
#
# Τρέξ' το πριν από κάθε παρουσίαση, και μετά από κάθε αλλαγή
# στο docker-compose.yml ή στο μοντέλο.
#
# Χρήση:  ./scripts/smoke_test.sh
# Έξοδος: 0 = όλα καλά, 1 = υπάρχει πρόβλημα
# ============================================================

cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
PASS=0; FAIL=0; WARN=0

ok()   { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS+1)); }
bad()  { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "  ${YELLOW}!${NC} $1"; WARN=$((WARN+1)); }

echo ""
echo -e "${BLUE}═══ Jarvis — Έλεγχος συστήματος ═══${NC}"

# ── 1. Αρχεία & ρυθμίσεις ───────────────────────────────────
echo ""
echo -e "${BLUE}1. Αρχεία${NC}"

CORPUS=$(grep -oE 'JARVIS_CORPUS=[^ ]+' docker-compose.yml | cut -d= -f2)
CORPUS_LOCAL="./data/$(basename "$CORPUS")"
if [[ -f "$CORPUS_LOCAL" ]]; then
    ok "corpus: $(basename "$CORPUS") ($(du -h "$CORPUS_LOCAL" | cut -f1))"
else
    bad "corpus λείπει: $CORPUS_LOCAL"
fi

# Το corpus δεν πρέπει να είναι κάποιο από τα γνωστά ΜΗ καθαρά αρχεία.
#
# Ελέγχεται με άρνηση, όχι με λίστα επιτρεπτών εκδόσεων. Η προηγούμενη
# μορφή ήταν `== *_v4.json`, δηλαδή καρφωμένος αριθμός έκδοσης: μόλις
# παρήχθη το v5 — καθαρότερο από το v4 — ο έλεγχος το κατήγγειλε ως
# μολυσμένο. Ένας έλεγχος που αποτυγχάνει όταν κάτι βελτιώνεται εκπαιδεύει
# τον χρήστη να τον αγνοεί.
#
# Η πραγματική εγγύηση δίνεται από τους ελέγχους περιεχομένου στην
# ενότητα 2, που μετράνε ονόματα αντί να διαβάζουν ονόματα αρχείων.
case "$(basename "$CORPUS")" in
    *sanitized*|*_v4.json)
        bad "δείχνει σε παλιό corpus με γνωστές διαρροές: $(basename "$CORPUS")"
        echo "        τρέξε: python3 scripts/resanitise_surnames.py --write"
        ;;
    *raw*|*jarvis_training_data.json)
        bad "δείχνει στο ΑΚΑΘΑΡΤΟ corpus: $(basename "$CORPUS")"
        ;;
    *)
        ok "δείχνει σε καθαρό corpus: $(basename "$CORPUS")"
        ;;
esac

[[ -f models/jarvis-adapter.gguf ]] && ok "adapter GGUF" || bad "λείπει models/jarvis-adapter.gguf"
[[ -f models/llama-krikri-8b-instruct-q4_k_m.gguf ]] && ok "base GGUF" || bad "λείπει το base GGUF"
[[ -f config/identity.yaml ]] && ok "identity.yaml" || warn "λείπει config/identity.yaml (cp από .example)"

# ── 2. GDPR ─────────────────────────────────────────────────
echo ""
echo -e "${BLUE}2. GDPR${NC}"
if [[ -f "$CORPUS_LOCAL" ]]; then
    LEAKS=$(PYTHONPATH=src python3 -c "
import json,sys
try:
    from jarvis.sanitization.greek_names import redact_given_names
except Exception:
    print('SKIP'); sys.exit()
r = json.load(open('$CORPUS_LOCAL', encoding='utf-8'))
n = sum(redact_given_names((x.get('response_clean') or ''))[1] for x in r)
print(n)
" 2>/dev/null)
    if [[ "$LEAKS" == "0" ]]; then
        ok "μηδέν ονόματα τρίτων στο corpus"
    elif [[ "$LEAKS" == "SKIP" ]]; then
        warn "δεν μπόρεσα να ελέγξω (λείπει το πακέτο jarvis)"
    else
        bad "$LEAKS ονόματα τρίτων στο corpus"
    fi
fi
# Επώνυμα. Ξεχωριστός έλεγχος από τα βαφτιστικά, γιατί είναι ξεχωριστός
# ανιχνευτής — και το κενό φάνηκε μόνο όταν το εκπαιδευμένο μοντέλο
# εξέφερε δύο πραγματικά επώνυμα σε αξιολόγηση.
if [[ -f "$CORPUS_LOCAL" ]]; then
    SURNAMES=$(PYTHONPATH=src python3 -c "
import json,sys
try:
    from jarvis.sanitization.greek_surnames import find_surnames
except Exception:
    print('SKIP'); sys.exit()
r = json.load(open('$CORPUS_LOCAL', encoding='utf-8'))
n = sum(len(find_surnames(x.get(f) or ''))
        for x in r for f in ('instruction_clean','response_clean','formatted_prompt'))
print(n)
" 2>/dev/null)
    if [[ "$SURNAMES" == "0" ]]; then
        ok "μηδέν επώνυμα τρίτων στο corpus"
    elif [[ "$SURNAMES" == "SKIP" ]]; then
        warn "δεν μπόρεσα να ελέγξω επώνυμα"
    else
        bad "$SURNAMES επώνυμα τρίτων — τρέξε scripts/resanitise_surnames.py --write"
    fi
fi

git check-ignore -q data/ 2>/dev/null && ok "data/ εκτός git" || warn "το data/ ίσως μπαίνει στο git"
git ls-files 2>/dev/null | grep -q "config/surnames.txt" && bad "surnames.txt ΕΙΝΑΙ στο git" || ok "surnames.txt εκτός git"
git ls-files 2>/dev/null | grep -q "config/identity.yaml" && bad "identity.yaml ΕΙΝΑΙ στο git" || ok "identity.yaml εκτός git"

# ── 3. Υπηρεσίες ────────────────────────────────────────────
echo ""
echo -e "${BLUE}3. Υπηρεσίες${NC}"

for svc in jarvis-api:8000 jarvis-n8n:5678 jarvis-chromadb:8100 jarvis-frontend:3001; do
    name="${svc%%:*}"
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${name}$" \
        && ok "$name τρέχει" || bad "$name ΔΕΝ τρέχει"
done

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^jarvis-ollama$'; then
    warn "ο Ollama σε Docker τρέχει — θα συγκρουστεί με τον native (πόρτα 11434)"
else
    ok "ο Ollama σε Docker είναι σταματημένος (σωστό)"
fi

# ── 4. Ollama & μοντέλο ─────────────────────────────────────
echo ""
echo -e "${BLUE}4. Μοντέλο${NC}"

if curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama απαντά στο :11434"
    if curl -s --max-time 3 http://localhost:11434/api/tags | grep -q '"jarvis'; then
        ok "το μοντέλο 'jarvis' είναι καταχωρημένο"
    else
        bad "το μοντέλο 'jarvis' ΔΕΝ βρέθηκε"
    fi
    # native ή Docker;
    if pgrep -x ollama >/dev/null 2>&1; then
        ok "native Ollama (Metal — γρήγορο)"
    else
        warn "δεν φαίνεται native διεργασία — ίσως τρέχει σε Docker (αργό)"
    fi
else
    bad "ο Ollama δεν απαντά στο :11434"
fi

# ── 5. Ροή end-to-end ───────────────────────────────────────
echo ""
echo -e "${BLUE}5. End-to-end${NC}"

API=$(curl -s --max-time 5 http://localhost:8000/health 2>/dev/null)
echo "$API" | grep -q '"status"' && ok "jarvis-api /health" || bad "η jarvis-api δεν απαντά"

RAG=$(curl -s --max-time 10 -X POST http://localhost:8000/orchestration/rag \
      -H "Content-Type: application/json" -d '{"query":"δουλεια","top_k":2}' 2>/dev/null)
RAG_N=$(echo "$RAG" | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('num_results', -1))
except Exception: print(-1)
" 2>/dev/null)

if [[ "$RAG_N" -gt 0 ]] 2>/dev/null; then
    ok "RAG λειτουργεί ($RAG_N αποτελέσματα από το corpus)"
    # Το context πρέπει να περιέχει placeholders, όχι πραγματικά ονόματα.
    if echo "$RAG" | grep -q '\[Person_'; then
        ok "το ανακτημένο context είναι ανωνυμοποιημένο"
    fi
elif [[ "$RAG_N" == "0" ]]; then
    warn "το RAG δεν βρήκε αποτελέσματα για τη δοκιμαστική ερώτηση"
elif echo "$RAG" | grep -q 'corpus_missing'; then
    bad "RAG: το corpus δεν βρέθηκε μέσα στο container"
else
    bad "το RAG endpoint δεν απαντά σωστά"
fi

echo -e "  ${BLUE}→${NC} δοκιμή webhook (μπορεί να πάρει λίγο)..."
START=$(date +%s)
WH=$(curl -s --max-time 60 http://localhost:5678/webhook/twin-chat \
     -H "Content-Type: application/json" -d '{"message":"τι κανεις;"}' 2>/dev/null)
ELAPSED=$(( $(date +%s) - START ))

if echo "$WH" | grep -q '"reply"'; then
    REPLY=$(echo "$WH" | python3 -c "import json,sys; print(json.load(sys.stdin)['reply'][:70])" 2>/dev/null)
    INTENT=$(echo "$WH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('intent','?'))" 2>/dev/null)
    ok "webhook → \"$REPLY\" [$INTENT] σε ${ELAPSED}s"
    if [[ $ELAPSED -le 5 ]]; then
        ok "ταχύτητα κατάλληλη για παρουσίαση (${ELAPSED}s)"
    else
        warn "αργό για παρουσίαση: ${ELAPSED}s — προθέρμανε το μοντέλο πρώτα"
    fi
elif echo "$WH" | grep -q 'not registered'; then
    bad "το n8n workflow δεν είναι ενεργό (publish + restart n8n)"
else
    bad "το webhook δεν απάντησε"
fi

# ── 5b. Registers ───────────────────────────────────────────
# Το χαρακτηριστικό αποτυγχάνει σιωπηλά: αν η ιδιότητα δεν φτάσει
# στο μοντέλο, το pop-up εξακολουθεί να εμφανίζεται και οι
# απαντήσεις απλώς είναι ίδιες. Ελέγχεται εδώ ώστε να μη
# φτάσει έτσι στην παρουσίαση.
PERSONA=$(curl -s --max-time 5 -X POST http://localhost:8000/orchestration/persona \
          -H "Content-Type: application/json" \
          -d '{"speaker_name":"Παναγιώτης","speaker_role":"καθηγητής"}' 2>/dev/null)
if echo "$PERSONA" | grep -q '"academic"'; then
    ok "registers: «καθηγητής» → academic"
elif [[ -n "$PERSONA" ]]; then
    bad "registers: λάθος επιλογή → $(echo "$PERSONA" | head -c 60)"
else
    warn "το /orchestration/persona δεν απαντά (χρειάζεται rebuild της jarvis-api;)"
fi

# ── 6. Tests ────────────────────────────────────────────────
echo ""
echo -e "${BLUE}6. Unit tests${NC}"
if command -v python3 >/dev/null && [[ -d tests ]]; then
    OUT=$(PYTHONPATH=src python3 -m pytest tests/ -q 2>&1 | tail -3)
    if echo "$OUT" | grep -qi "fail\|error"; then
        bad "$(echo "$OUT" | grep -i 'fail' | head -1)"
    else
        ok "όλα τα tests περνάνε"
    fi
fi

# ── Σύνοψη ──────────────────────────────────────────────────
echo ""
echo -e "${BLUE}═══════════════════════════════${NC}"
echo -e "  ${GREEN}$PASS πέρασαν${NC}   ${YELLOW}$WARN προειδοποιήσεις${NC}   ${RED}$FAIL απέτυχαν${NC}"
echo ""
if [[ $FAIL -eq 0 ]]; then
    echo -e "  ${GREEN}Το σύστημα είναι έτοιμο.${NC}"
    exit 0
else
    echo -e "  ${RED}Υπάρχουν προβλήματα — δες τα ✗ παραπάνω.${NC}"
    exit 1
fi
