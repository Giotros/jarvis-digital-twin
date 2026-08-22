#!/bin/bash
# ============================================================
# Περνάει τα speaker_name / speaker_role από το webhook στους
# κόμβους Generate, ώστε το μοντέλο να ξέρει σε ποιον μιλάει.
#
# Το frontend τα στέλνει ήδη στο σώμα του αιτήματος, αλλά το n8n
# προωθεί μόνο ό,τι αναφέρεται ρητά στο jsonBody κάθε κόμβου.
# Χωρίς αυτό το βήμα η φόρμα δουλεύει και δεν αλλάζει τίποτα —
# σιωπηλά, που είναι το χειρότερο είδος αστοχίας.
#
# Εξάγει το ΤΡΕΧΟΝ workflow (άρα κρατά τα credentials), το
# τροποποιεί, και το επαναφέρει.
#
# Χρήση:  ./scripts/add_speaker_to_workflow.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
WF_ID="tBPEVTfphTTV0xLr"
TMP="/tmp/jarvis_wf_speaker_$$.json"

echo ""
echo -e "${BLUE}Προσθήκη συνομιλητή στο n8n workflow${NC}"
echo ""

echo -e "  ${BLUE}→${NC} Εξαγωγή τρέχοντος workflow..."
docker exec jarvis-n8n n8n export:workflow --id="$WF_ID" --output=/tmp/wf.json >/dev/null 2>&1
docker cp jarvis-n8n:/tmp/wf.json "$TMP" >/dev/null
echo -e "  ${GREEN}✓${NC} εξήχθη"

python3 - "$TMP" << 'PYEOF'
import json, re, sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
wf = data[0] if isinstance(data, list) else data

WEBHOOK = "$('Webhook: Receive Message').item.json.body"
ADDITION = (
    '  "speaker_name": {{ JSON.stringify(%s.speaker_name || "") }},\n'
    '  "speaker_role": {{ JSON.stringify(%s.speaker_role || "") }},\n'
) % (WEBHOOK, WEBHOOK)

patched = skipped = 0

for node in wf.get("nodes", []):
    p = node.get("parameters", {})
    body = p.get("jsonBody")
    if not isinstance(body, str) or not node.get("name", "").startswith("Generate"):
        continue
    if "speaker_name" in body:
        skipped += 1
        continue
    # Insert straight after the opening brace so the fields exist even if the
    # rest of the body is later rewritten by another patch script.
    new_body, n = re.subn(r"(=\{\n)", r"\1" + ADDITION, body, count=1)
    if n == 0:
        print(f"  ! δεν αναγνώρισα τη μορφή του {node['name']}")
        continue
    p["jsonBody"] = new_body
    patched += 1

json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
print(f"  ✓ {patched} κόμβοι Generate ενημερώθηκαν"
      + (f" ({skipped} είχαν ήδη το πεδίο)" if skipped else ""))
PYEOF

echo -e "  ${BLUE}→${NC} Επαναφορά..."
docker cp "$TMP" jarvis-n8n:/tmp/wf_speaker.json >/dev/null
docker exec jarvis-n8n n8n import:workflow --input=/tmp/wf_speaker.json >/dev/null 2>&1

# ΑΠΑΡΑΙΤΗΤΟ: το import:workflow επαναφέρει το workflow σε ανενεργή
# κατάσταση. Χωρίς το publish το webhook παύει να είναι καταχωρημένο και
# κάθε αίτημα γυρίζει κενό — χωρίς σφάλμα, που το κάνει δύσκολο να το δεις.
docker exec jarvis-n8n n8n publish:workflow --id="$WF_ID" >/dev/null 2>&1

docker restart jarvis-n8n >/dev/null
rm -f "$TMP"

echo -e "  ${BLUE}→${NC} Αναμονή εκκίνησης n8n..."
for _ in $(seq 1 30); do
    curl -s --max-time 2 http://localhost:5678/healthz >/dev/null 2>&1 && break
    sleep 1
done
sleep 5

# ── Επαλήθευση: ίδια ερώτηση, τρεις ιδιότητες ───────────────
# Αν οι τρεις απαντήσεις είναι πανομοιότυπες, η ρύθμιση δεν έφτασε
# στο μοντέλο και το χαρακτηριστικό είναι διακοσμητικό.
echo ""
echo -e "${BLUE}Ίδια ερώτηση, τρεις ιδιότητες${NC}"
echo ""
Q="τι κανεις;"
FAILED=0
for pair in "Παναγιώτης|φίλος" "Παναγιώτης|συνάδελφος" "Παναγιώτης|καθηγητής"; do
    NAME="${pair%%|*}"; ROLE="${pair##*|}"
    R=$(curl -s --max-time 90 http://localhost:5678/webhook/twin-chat \
        -H "Content-Type: application/json" \
        -d "{\"message\":\"$Q\",\"speaker_name\":\"$NAME\",\"speaker_role\":\"$ROLE\"}" 2>/dev/null)

    if echo "$R" | grep -q '"reply"'; then
        REPLY=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin)['reply'])" 2>/dev/null)
        printf "  ${GREEN}✓${NC} %-12s → %s\n" "$ROLE" "$REPLY"
    else
        FAILED=1
        # Δείξε τι πραγματικά γύρισε. Ένα "—" κρύβει τη διαφορά ανάμεσα σε
        # "το workflow δεν είναι ενεργό" και "το μοντέλο άργησε".
        printf "  ${RED}✗${NC} %-12s → %s\n" "$ROLE" "$(echo "$R" | head -c 160)"
        [[ -z "$R" ]] && printf "      (κενή απόκριση — το n8n μάλλον δεν σηκώθηκε ακόμα)\n"
        echo "$R" | grep -q 'not registered' && \
            printf "      (το webhook δεν είναι καταχωρημένο — χρειάζεται publish)\n"
    fi
done
echo ""
if [[ $FAILED -eq 1 ]]; then
    echo -e "  ${RED}Κάποια probes απέτυχαν.${NC} Δες την κατάσταση του workflow:"
    echo "    docker exec jarvis-n8n n8n list:workflow"
    exit 1
fi
