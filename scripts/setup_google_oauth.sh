#!/bin/bash
# ============================================================
# Ρυθμίζει το n8n ώστε να ολοκληρώνεται το Google OAuth, και
# τυπώνει το ακριβές redirect URI για το Google Console.
#
# Δουλεύει και με ngrok και χωρίς. Χωρίς είναι το συνηθισμένο:
# η Google δέχεται http://localhost στα redirect URIs, ρητή
# εξαίρεση στον κανόνα https ώστε να γίνεται τοπική ανάπτυξη.
# Το ngrok χρειάζεται μόνο αν το webhook πρέπει να είναι
# προσβάσιμο από άλλο μηχάνημα.
#
# ΓΙΑΤΙ ΥΠΑΡΧΕΙ
# -------------
# Χωρίς WEBHOOK_URL, το n8n παράγει redirect URI με βάση το
# localhost. Η Google το απορρίπτει, η σύνδεση Gmail και Calendar
# δεν ολοκληρώνεται ποτέ, και το σύστημα απαντά για το αύριο
# χωρίς ημερολόγιο — εκεί γεννήθηκε η επινοημένη εκδρομή στο
# Ναύπλιο, με ώρα αναχώρησης και ώρα επιστροφής.
#
# Γράφεται από script και όχι με το χέρι επειδή η διεύθυνση του
# ngrok αλλάζει σε κάθε εκκίνηση στο δωρεάν επίπεδο. Ένα URL
# πληκτρολογημένο λάθος δεν βγάζει σφάλμα· απλώς το OAuth δεν
# δουλεύει, και αυτό μοιάζει με «δεν το έχω ρυθμίσει ακόμα».
#
# Χρήση:  ./scripts/setup_google_oauth.sh
# ============================================================

set -uo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo ""
echo -e "${BLUE}Ρύθμιση Google OAuth για το n8n${NC}"
echo ""

# ── 1. Βρες το ενεργό τούνελ ────────────────────────────────
# Το ngrok εκθέτει τοπικό API στο 4040 όσο τρέχει.
TUNNELS=$(curl -s --max-time 3 http://localhost:4040/api/tunnels 2>/dev/null)

if [[ -z "$TUNNELS" ]]; then
    # Χωρίς ngrok, το localhost είναι ΑΡΚΕΤΟ για το Google OAuth.
    #
    # Η Google απαιτεί https στα redirect URIs με μία ρητή εξαίρεση: το
    # localhost. Είναι τεκμηριωμένο και σκόπιμο, ώστε να μπορεί κανείς να
    # αναπτύσσει τοπικά. Το ngrok χρειάζεται μόνο αν το webhook πρέπει να
    # είναι προσβάσιμο από άλλο μηχάνημα — για τη σύνδεση Gmail και
    # Calendar δεν χρειάζεται καθόλου.
    echo -e "  ${YELLOW}Το ngrok δεν τρέχει — δεν πειράζει.${NC}"
    echo ""
    echo "  Η Google δέχεται http://localhost στα redirect URIs (ρητή"
    echo "  εξαίρεση στον κανόνα https). Ρυθμίζω για τοπική λειτουργία."
    echo ""

    touch .env
    python3 - <<'PY'
import pathlib

# Οι τιμές ngrok αφαιρούνται εντελώς αντί να μηδενιστούν: μια γραμμή
# NGROK_URL= κενή θα έδινε στο n8n κενό WEBHOOK_URL, που είναι χειρότερο
# από το να λείπει — το compose δεν θα εφάρμοζε το default του.
path = pathlib.Path(".env")
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
kept = [ln for ln in lines
        if ln.split("=", 1)[0].strip() not in
        {"NGROK_URL", "NGROK_HOST", "N8N_PROTOCOL"}]
path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
PY

    docker compose up -d --force-recreate n8n >/dev/null 2>&1
    sleep 8
    ACTUAL=$(docker exec jarvis-n8n printenv WEBHOOK_URL 2>/dev/null)
    if [[ "$ACTUAL" == "http://localhost:5678/" ]]; then
        echo -e "  ${GREEN}✓${NC} Το n8n ρυθμίστηκε για localhost"
    else
        echo -e "  ${RED}✗${NC} Το n8n έχει «${ACTUAL:-κενό}»"
        exit 1
    fi

    echo ""
    echo -e "${BLUE}Τελευταίο βήμα — χειροκίνητο${NC}"
    echo ""
    echo "  Google Cloud Console → APIs & Services → Credentials"
    echo "  → OAuth 2.0 Client (τύπος: Web application)"
    echo "  → Authorized redirect URIs → πρόσθεσε:"
    echo ""
    echo -e "      ${GREEN}http://localhost:5678/rest/oauth2-credential/callback${NC}"
    echo ""
    echo "  Μετά, στο n8n: Credentials → Google Calendar / Gmail → Connect."
    echo ""
    echo -e "  ${YELLOW}Η αλλαγή θέλει 5 λεπτά έως λίγες ώρες για να ισχύσει${NC}"
    echo -e "  ${YELLOW}στη Google. Αν δεις redirect_uri_mismatch, περίμενε.${NC}"
    echo ""
    echo "  Το ngrok χρειάζεται ΜΟΝΟ αν θες το webhook προσβάσιμο από άλλο"
    echo "  μηχάνημα — π.χ. να γράφεις στο twin από το κινητό."
    echo ""
    exit 0
fi

URL=$(echo "$TUNNELS" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
# Προτίμησε https· το Google OAuth δεν δέχεται http σε δημόσιο host.
https = [t['public_url'] for t in data.get('tunnels', [])
         if t.get('public_url', '').startswith('https://')]
print(https[0] if https else '')
" 2>/dev/null)

if [[ -z "$URL" ]]; then
    echo -e "  ${RED}✗${NC} Το ngrok τρέχει αλλά δεν βρέθηκε https τούνελ."
    echo "     Το Google OAuth απαιτεί https."
    exit 1
fi

HOST="${URL#https://}"
echo -e "  ${GREEN}✓${NC} Βρέθηκε: ${BLUE}$URL${NC}"

# ── 2. Γράψε στο .env ───────────────────────────────────────
# Οι υπάρχουσες γραμμές αντικαθίστανται, οι υπόλοιπες μένουν.
touch .env
python3 - "$URL" "$HOST" <<'PY'
import pathlib, sys

url, host = sys.argv[1], sys.argv[2]
values = {"NGROK_URL": url, "NGROK_HOST": host, "N8N_PROTOCOL": "https"}

path = pathlib.Path(".env")
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
out, seen = [], set()
for line in lines:
    key = line.split("=", 1)[0].strip()
    if key in values:
        out.append(f"{key}={values[key]}")
        seen.add(key)
    else:
        out.append(line)
for key, value in values.items():
    if key not in seen:
        out.append(f"{key}={value}")
path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
echo -e "  ${GREEN}✓${NC} Γράφτηκε στο .env"

# ── 3. Επανεκκίνηση n8n ─────────────────────────────────────
echo -e "  ${BLUE}→${NC} Επανεκκίνηση n8n..."
docker compose up -d --force-recreate n8n >/dev/null 2>&1
sleep 8

# ── 4. Επαλήθευση ───────────────────────────────────────────
ACTUAL=$(docker exec jarvis-n8n printenv WEBHOOK_URL 2>/dev/null)
if [[ "$ACTUAL" == "$URL" ]]; then
    echo -e "  ${GREEN}✓${NC} Το n8n βλέπει το σωστό URL"
else
    echo -e "  ${RED}✗${NC} Το n8n έχει «${ACTUAL:-κενό}» αντί για «$URL»"
    exit 1
fi

CALLBACK="$URL/rest/oauth2-credential/callback"
echo ""
echo -e "${BLUE}Τελευταίο βήμα — χειροκίνητο${NC}"
echo ""
echo "  Google Cloud Console → APIs & Services → Credentials"
echo "  → OAuth 2.0 Client → Authorized redirect URIs → πρόσθεσε:"
echo ""
echo -e "      ${GREEN}$CALLBACK${NC}"
echo ""
echo "  Μετά, στο n8n: Credentials → Google Calendar / Gmail → Connect."
echo ""
echo -e "  ${YELLOW}Αν το ngrok ξαναξεκινήσει, η διεύθυνση αλλάζει και πρέπει${NC}"
echo -e "  ${YELLOW}να ενημερωθούν ΚΑΙ τα δύο. Σταθερό domain το λύνει.${NC}"
echo ""
