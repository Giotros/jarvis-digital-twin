#!/bin/bash
# ============================================================
# Ελέγχει ότι η ιδιότητα του συνομιλητή αλλάζει πραγματικά την
# απάντηση — και, αν δεν την αλλάζει, ΠΟΥ χάνεται.
#
# Τρία επίπεδα, με αυτή τη σειρά:
#   1. /orchestration/persona  — επιλέγεται σωστό register;
#   2. /orchestration/generate — το ακούει το μοντέλο;
#   3. webhook του n8n         — φτάνουν τα πεδία ως εκεί;
#
# Αν το 1 και το 2 περνούν αλλά το 3 όχι, φταίει η ροή του n8n.
# Αν το 1 περνάει και το 2 όχι, το μοντέλο αγνοεί την οδηγία.
# Χωρίς αυτόν τον διαχωρισμό, ένα «οι απαντήσεις είναι ίδιες»
# μπορεί να σημαίνει τρία εντελώς διαφορετικά πράγματα.
#
# Χρήση:  ./scripts/check_registers.sh
# ============================================================

cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
API="http://localhost:8000"
WEBHOOK="http://localhost:5678/webhook/twin-chat"

# Μια ερώτηση που έχει νόημα και στα τρία registers και όπου η διαφορά
# ύφους είναι ορατή. Το «τι κάνεις;» δεν κάνει: σε όλα τα registers η
# σωστή απάντηση είναι σχεδόν η ίδια, οπότε δεν διακρίνει τίποτα.
Q="με τι τεχνολογιες δουλεψες φετος;"
FAIL=0

echo ""
echo -e "${BLUE}═══ Έλεγχος registers ═══${NC}"

# ── 0. Αναμονή για το API ───────────────────────────────────
# Τρέχει συνήθως αμέσως μετά από docker compose up. Το container
# επιστρέφει τον έλεγχο σε λιγότερο από ένα δευτερόλεπτο, αλλά το
# FastAPI θέλει κάμποσο ακόμα. Χωρίς αναμονή, τα πρώτα βήματα
# αποτυγχάνουν και μοιάζουν με πραγματικά σφάλματα.
printf "\n  αναμονή για το API"
for _ in $(seq 1 40); do
    curl -s --max-time 2 "$API/health" >/dev/null 2>&1 && break
    printf "."
    sleep 1
done
if curl -s --max-time 2 "$API/health" >/dev/null 2>&1; then
    printf " ${GREEN}έτοιμο${NC}\n"
else
    printf " ${RED}δεν απαντά${NC}\n"
    echo -e "  Το jarvis-api δεν σηκώθηκε. Δες: docker compose logs jarvis-api"
    exit 1
fi

# ── 1. Επιλογή register ─────────────────────────────────────
echo ""
echo -e "${BLUE}1. Επιλογή register (χωρίς μοντέλο)${NC}"

check_pick() {
    local role="$1" expect="$2"
    local got
    got=$(curl -s --max-time 5 -X POST "$API/orchestration/persona" \
          -H "Content-Type: application/json" \
          -d "{\"speaker_name\":\"Παναγιώτης\",\"speaker_role\":\"$role\"}" \
          | python3 -c "import json,sys;print(json.load(sys.stdin)['speaker_register'])" 2>/dev/null)
    if [[ "$got" == "$expect" ]]; then
        printf "  ${GREEN}✓${NC} %-14s → %s\n" "$role" "$got"
    else
        printf "  ${RED}✗${NC} %-14s → %s (περίμενα %s)\n" "$role" "${got:-καμία απάντηση}" "$expect"
        FAIL=1
    fi
}

check_pick "φίλος"      "close"
check_pick "συνάδελφος" "professional"
check_pick "καθηγητής"  "academic"
check_pick ""           "neutral"

# ── 2. Απευθείας στο μοντέλο ────────────────────────────────
echo ""
echo -e "${BLUE}2. Απευθείας στο /generate (παρακάμπτει το n8n)${NC}"

declare -a DIRECT=()
for role in "φίλος" "συνάδελφος" "καθηγητής"; do
    R=$(curl -s --max-time 90 -X POST "$API/orchestration/generate" \
        -H "Content-Type: application/json" \
        -d "{\"message\":\"$Q\",\"speaker_name\":\"Παναγιώτης\",\"speaker_role\":\"$role\"}" 2>/dev/null)
    REPLY=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin).get('reply','').replace(chr(10),' '))" 2>/dev/null)
    REG=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin).get('speaker_register','?'))" 2>/dev/null)
    WORDS=$(echo "$REPLY" | wc -w | tr -d ' ')
    DIRECT+=("$REPLY")
    if [[ -n "$REPLY" ]]; then
        printf "  ${GREEN}✓${NC} %-12s [%-12s] %2s λέξεις\n      %s\n" "$role" "$REG" "$WORDS" "$REPLY"
    else
        printf "  ${RED}✗${NC} %-12s → καμία απάντηση\n" "$role"
        FAIL=1
    fi
done

# Το κρίσιμο ερώτημα δεν είναι «απάντησε;» αλλά «απάντησε ΔΙΑΦΟΡΕΤΙΚΑ;».
# Τρεις πανομοιότυπες απαντήσεις σημαίνουν ότι το χαρακτηριστικό είναι
# διακοσμητικό, ακόμα κι αν κάθε κλήση πέτυχε.
echo ""
if [[ "${DIRECT[0]}" == "${DIRECT[1]}" && "${DIRECT[1]}" == "${DIRECT[2]}" ]]; then
    echo -e "  ${RED}✗ Και οι τρεις απαντήσεις ταυτόσημες${NC} — το register δεν επηρεάζει το μοντέλο."
    FAIL=1
else
    echo -e "  ${GREEN}✓ Οι απαντήσεις διαφέρουν${NC} — η οδηγία φτάνει στο μοντέλο."
fi

# «Διαφέρουν» δεν σημαίνει «σωστές». Το μοντέλο απάντησε κάποτε
# «Ειμαι καλά αγορι μ να ξερς» σε καθηγητή — διαφορετικό από τα άλλα δύο,
# και εντελώς λάθος. Ο έλεγχος πρέπει να είναι τι ΔΕΝ λέει στο formal.
echo ""
echo -e "${BLUE}   Οικειότητα σε formal registers${NC}"
LEAK=0
for i in 1 2; do
    for w in "φιλαράκι" "φιλαρακι" "αγορι μ" "αγόρι μ" "ρε φίλε" "ρε φιλε" "μεγάλε"; do
        if echo "${DIRECT[$i]}" | grep -qi "$w"; then
            echo -e "     ${RED}✗${NC} διέρρευσε «$w» → ${DIRECT[$i]}"
            LEAK=1; FAIL=1
        fi
    done
done
[[ $LEAK -eq 0 ]] && echo -e "     ${GREEN}✓${NC} καμία οικεία προσφώνηση σε professional/academic"

# Το ύφος είναι το εύκολο μέρος. Το μοντέλο απάντησε κάποτε «Krikri-12B»,
# «QLoRA πάνω στο BERTweet», «ανάλυση συναισθημάτων» — ρευστά, λεπτομερώς
# και λάθος. Μπροστά σε εξεταστή αυτό κοστίζει πολύ περισσότερο από μια
# άτσαλη προσφώνηση, και είναι δυσκολότερο να το πιάσεις ακούγοντας.
#
# Ελέγχονται και τα τρία registers, όχι μόνο το academic: ρωτημένο ως
# συνάδελφος, το twin περιέγραψε εντελώς άλλη διπλωματική («πρόβλεψη
# τιμών ενέργειας»). Ένας συνάδελφος που ακούει λάθος περιγραφή της
# δουλειάς σου δεν είναι λιγότερο πρόβλημα από έναν εξεταστή.
echo ""
echo -e "${BLUE}   Ακρίβεια τεχνικών ισχυρισμών${NC}"
i=0
for role in "φίλος" "συνάδελφος" "καθηγητής"; do
    CLAIMS=$(PYTHONPATH=src python3 -c "
import sys
from jarvis.inference.thesis_facts import (
    check_technical_claims, unsupported_technologies,
    check_acronym_expansions, check_corrupted_names)
text = sys.argv[1]
for issue in check_technical_claims(text):
    print(issue)
for issue in check_acronym_expansions(text):
    print(issue)
for issue in check_corrupted_names(text):
    print(issue)
extra = unsupported_technologies(text)
if extra:
    print('Δεν αναφέρονται στα στοιχεία της εργασίας: ' + ', '.join(extra))
# Κάλυψη. Μια απάντηση χωρίς περιεχόμενο δεν έχει τι να αντιφάσκει, και
# ένας έλεγχος που μετρά μόνο λάθη τη βαθμολογεί άριστα. Το ίδιο σφάλμα
# κέρδισε κάποτε το ablation: το προσαρμοσμένο μοντέλο πήρε μηδέν λάθη
# απαντώντας «έχεις κάποιο demo;» σε ερώτηση για τεχνολογίες. Εδώ πήρε ✓
# λέγοντας «Μιλάμε για το 2ο εξάμηνο ή και τα δύο μαζί?».
# Η ασφαλής άρνηση ΔΕΝ είναι το ίδιο με τη μουρμούρα.
#
# Όταν και οι δύο απόπειρες παραγωγής αντιφάσκουν, το /generate επιστρέφει
# ρητή άρνηση αντί να σερβίρει ψέμα. Αυτό είναι ο μηχανισμός να δουλεύει,
# και ένα ✗ δίπλα του λέει το αντίθετο από την αλήθεια — το ίδιο σφάλμα
# με το ✓ που δεν ξεχώριζε «ελέγχθηκε» από «δεν ελέγχθηκε».
if 'Δεν θέλω να πω κάτι λάθος' in text:
    print('ΑΡΝΗΘΗΚΕ — δύο απόπειρες αντέφασκαν, το σύστημα δεν είπε ψέμα')
else:
    named = [t for t in ('python','ray','ollama','n8n','databricks','krikri',
                         'qlora','chromadb','docker','fastapi','pytorch')
             if t in text.lower()]
    if len(named) < 2:
        print('Δεν απάντησε στην ερώτηση — %d τεχνολογίες αναφέρθηκαν' % len(named))
" "${DIRECT[$i]}" 2>/dev/null)

    if [[ "$CLAIMS" == ΑΡΝΗΘΗΚΕ* ]]; then
        echo -e "     ${YELLOW}⚠${NC} $role — ${CLAIMS#ΑΡΝΗΘΗΚΕ — }"
    elif [[ -n "$CLAIMS" ]]; then
        echo -e "     ${RED}✗${NC} $role — επινοημένα στοιχεία:"
        echo "$CLAIMS" | sed 's/^/          → /'
        # Το «φίλος» ΔΕΝ εξαιρείται πλέον. Εξαιρούνταν με το σκεπτικό ότι
        # το close register δεν παίρνει τα στοιχεία του project σκόπιμα,
        # οπότε μια ανακρίβεια εκεί είναι γνωστός συμβιβασμός. Η μέτρηση
        # το διέψευσε: στον φίλο το σύστημα ανέφερε OpenAI GPT-4 και
        # Django — δηλαδή ψευδή δήλωση για την ίδια την εργασία, όχι
        # αθώα φλυαρία. Η θεμελίωση αποφασίζεται τώρα από την ερώτηση,
        # άρα και ο έλεγχος ισχύει παντού.
        FAIL=1
    else
        echo -e "     ${GREEN}✓${NC} $role — κανένας ισχυρισμός δεν αντιφάσκει"
    fi
    i=$((i+1))
done

# ── 3. Μέσω n8n ─────────────────────────────────────────────
echo ""
echo -e "${BLUE}3. Μέσω του webhook (πλήρης ροή)${NC}"

declare -a VIA_N8N=()
for role in "φίλος" "συνάδελφος" "καθηγητής"; do
    R=$(curl -s --max-time 90 "$WEBHOOK" -H "Content-Type: application/json" \
        -d "{\"message\":\"$Q\",\"speaker_name\":\"Παναγιώτης\",\"speaker_role\":\"$role\"}" 2>/dev/null)
    REPLY=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin).get('reply','').replace(chr(10),' '))" 2>/dev/null)
    VIA_N8N+=("$REPLY")
    if [[ -n "$REPLY" ]]; then
        printf "  ${GREEN}✓${NC} %-12s %2s λέξεις\n      %s\n" "$role" "$(echo "$REPLY" | wc -w | tr -d ' ')" "$REPLY"
    else
        printf "  ${RED}✗${NC} %-12s → %s\n" "$role" "$(echo "$R" | head -c 100)"
        FAIL=1
    fi
done

echo ""
if [[ "${VIA_N8N[0]}" == "${VIA_N8N[1]}" && "${VIA_N8N[1]}" == "${VIA_N8N[2]}" ]]; then
    echo -e "  ${RED}✗ Ταυτόσημες μέσω n8n${NC}"
    if [[ "${DIRECT[0]}" != "${DIRECT[1]}" ]]; then
        echo -e "    Το /generate ΔΟΥΛΕΥΕΙ, άρα τα πεδία χάνονται στη ροή του n8n:"
        echo -e "    ${YELLOW}./scripts/add_speaker_to_workflow.sh${NC}"
    fi
    FAIL=1
else
    echo -e "  ${GREEN}✓ Οι απαντήσεις διαφέρουν και μέσω n8n${NC}"
fi

# Ο έλεγχος ακρίβειας έτρεχε μόνο στο βήμα 2 και έλειπε από εδώ — από τη
# μόνη διαδρομή που θα δει ο εξεταστής. Οι απαντήσεις της πλήρους ροής
# αποδείχθηκαν οι χειρότερες (Kubernetes, AWS Lambda, blockchain) και το
# script τις τύπωνε με ✓ δίπλα τους, επειδή το ✓ σήμαινε μόνο «ήρθε
# απάντηση». Ένα διαγνωστικό που ελέγχει το εύκολο μισό της διαδρομής
# αναφέρει επιτυχία ακριβώς εκεί που δεν πρέπει.
echo ""
echo -e "${BLUE}   Ακρίβεια τεχνικών ισχυρισμών (πλήρης ροή)${NC}"
i=0
for role in "φίλος" "συνάδελφος" "καθηγητής"; do
    CLAIMS=$(PYTHONPATH=src python3 -c "
import sys
from jarvis.inference.thesis_facts import (
    check_technical_claims, unsupported_technologies,
    check_acronym_expansions, check_corrupted_names)
text = sys.argv[1]
for issue in check_technical_claims(text):
    print(issue)
for issue in check_acronym_expansions(text):
    print(issue)
for issue in check_corrupted_names(text):
    print(issue)
extra = unsupported_technologies(text)
if extra:
    print('Δεν αναφέρονται στα στοιχεία της εργασίας: ' + ', '.join(extra))
# Κάλυψη. Μια απάντηση χωρίς περιεχόμενο δεν έχει τι να αντιφάσκει, και
# ένας έλεγχος που μετρά μόνο λάθη τη βαθμολογεί άριστα. Το ίδιο σφάλμα
# κέρδισε κάποτε το ablation: το προσαρμοσμένο μοντέλο πήρε μηδέν λάθη
# απαντώντας «έχεις κάποιο demo;» σε ερώτηση για τεχνολογίες. Εδώ πήρε ✓
# λέγοντας «Μιλάμε για το 2ο εξάμηνο ή και τα δύο μαζί?».
# Η ασφαλής άρνηση ΔΕΝ είναι το ίδιο με τη μουρμούρα.
#
# Όταν και οι δύο απόπειρες παραγωγής αντιφάσκουν, το /generate επιστρέφει
# ρητή άρνηση αντί να σερβίρει ψέμα. Αυτό είναι ο μηχανισμός να δουλεύει,
# και ένα ✗ δίπλα του λέει το αντίθετο από την αλήθεια — το ίδιο σφάλμα
# με το ✓ που δεν ξεχώριζε «ελέγχθηκε» από «δεν ελέγχθηκε».
if 'Δεν θέλω να πω κάτι λάθος' in text:
    print('ΑΡΝΗΘΗΚΕ — δύο απόπειρες αντέφασκαν, το σύστημα δεν είπε ψέμα')
else:
    named = [t for t in ('python','ray','ollama','n8n','databricks','krikri',
                         'qlora','chromadb','docker','fastapi','pytorch')
             if t in text.lower()]
    if len(named) < 2:
        print('Δεν απάντησε στην ερώτηση — %d τεχνολογίες αναφέρθηκαν' % len(named))
" "${VIA_N8N[$i]}" 2>/dev/null)

    if [[ "$CLAIMS" == ΑΡΝΗΘΗΚΕ* ]]; then
        echo -e "     ${YELLOW}⚠${NC} $role — ${CLAIMS#ΑΡΝΗΘΗΚΕ — }"
    elif [[ -n "$CLAIMS" ]]; then
        echo -e "     ${RED}✗${NC} $role — επινοημένα στοιχεία:"
        echo "$CLAIMS" | sed 's/^/          → /'
        FAIL=1
    else
        echo -e "     ${GREEN}✓${NC} $role — κανένας ισχυρισμός δεν αντιφάσκει"
    fi
    i=$((i+1))
done

echo ""
echo -e "${BLUE}═══════════════════════════${NC}"
if [[ $FAIL -eq 0 ]]; then
    echo -e "  ${GREEN}Τα registers λειτουργούν end-to-end.${NC}"
    exit 0
fi
echo -e "  ${RED}Δες τα ✗ παραπάνω.${NC}"
exit 1
