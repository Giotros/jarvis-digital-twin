#!/bin/bash
# ============================================================
# Διορθώνει δύο πράγματα στο ζωντανό n8n workflow:
#
#   1. JSON escaping στα Generate nodes. Το context από το RAG
#      περιέχει εισαγωγικά και αλλαγές γραμμής από πραγματικά
#      μηνύματα. Μπαίνοντας ωμό μέσα σε "..." σπάει το JSON και
#      η κλήση αποτυγχάνει με 500 — πριν καν φτάσει στο μοντέλο.
#      Λύση: JSON.stringify() χωρίς περιβάλλοντα εισαγωγικά.
#
#   2. Ανοχή σφαλμάτων στους κόμβους εργαλείων. Αν λείπουν
#      credentials για Gmail/Calendar/GitHub, η ροή σταματούσε
#      και ο χρήστης έπαιρνε κενό. Τώρα συνεχίζει και απαντά
#      χωρίς το εργαλείο, αντί να σιωπήσει.
#
# Εξάγει το ΤΡΕΧΟΝ workflow (άρα κρατά τα credentials), το
# διορθώνει, και το επαναφέρει.
#
# Χρήση:  ./scripts/fix_workflow_json_escaping.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
WF_ID="tBPEVTfphTTV0xLr"
TMP="/tmp/jarvis_wf_$$.json"

echo ""
echo -e "${BLUE}Διόρθωση n8n workflow${NC}"
echo ""

# ── 1. Εξαγωγή του ζωντανού workflow ────────────────────────
echo -e "  ${BLUE}→${NC} Εξαγωγή τρέχοντος workflow..."
docker exec jarvis-n8n n8n export:workflow --id="$WF_ID" --output=/tmp/wf.json >/dev/null 2>&1
docker cp jarvis-n8n:/tmp/wf.json "$TMP" >/dev/null
echo -e "  ${GREEN}✓${NC} εξήχθη (τα credentials διατηρούνται)"

# ── 2. Διόρθωση ─────────────────────────────────────────────
python3 - "$TMP" << 'PYEOF'
import json, re, sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
wf = data[0] if isinstance(data, list) else data

fixed_json, fixed_err = 0, 0

for node in wf.get("nodes", []):
    p = node.get("parameters", {})

    # ── JSON escaping ───────────────────────────────────────
    body = p.get("jsonBody")
    if isinstance(body, str) and "{{" in body:
        before = body
        # "field": "{{ expr }}"  →  "field": {{ JSON.stringify(expr) }}
        # Παραλείπονται όσα είναι ήδη τυλιγμένα σε JSON.stringify.
        def wrap(m):
            key, expr = m.group(1), m.group(2).strip()
            if "JSON.stringify" in expr:
                return m.group(0)
            return f'"{key}": {{{{ JSON.stringify({expr}) }}}}'

        body = re.sub(r'"(\w+)":\s*"\{\{([^}]*(?:\}[^}][^}]*)*)\}\}"', wrap, body)
        if body != before:
            p["jsonBody"] = body
            fixed_json += 1

    # ── Ανοχή σφαλμάτων στα εργαλεία ────────────────────────
    # Ένας κόμβος χωρίς credentials δεν πρέπει να σκοτώνει τη ροή:
    # καλύτερα απάντηση χωρίς το εργαλείο παρά σιωπή.
    ntype = node.get("type", "")
    if any(t in ntype for t in ("gmail", "googleCalendar")) or (
        ntype.endswith("httpRequest")
        and "jarvis-api" not in json.dumps(p)
    ):
        if node.get("onError") != "continueRegularOutput":
            node["onError"] = "continueRegularOutput"
            node["retryOnFail"] = False
            fixed_err += 1

json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
print(f"  ✓ {fixed_json} κόμβοι με διορθωμένο JSON escaping")
print(f"  ✓ {fixed_err} κόμβοι εργαλείων με ανοχή σφαλμάτων")
PYEOF

# ── 3. Επαναφορά ────────────────────────────────────────────
echo -e "  ${BLUE}→${NC} Επαναφορά..."
docker cp "$TMP" jarvis-n8n:/tmp/wf_fixed.json >/dev/null
docker exec jarvis-n8n n8n import:workflow --input=/tmp/wf_fixed.json >/dev/null 2>&1
docker exec jarvis-n8n n8n publish:workflow --id="$WF_ID" >/dev/null 2>&1
docker restart jarvis-n8n >/dev/null
rm -f "$TMP"

echo -e "  ${BLUE}→${NC} Αναμονή εκκίνησης n8n..."
for _ in $(seq 1 30); do
    curl -s --max-time 2 http://localhost:5678/healthz >/dev/null 2>&1 && break
    sleep 1
done
sleep 5

# ── 4. Επαλήθευση ───────────────────────────────────────────
echo ""
echo -e "${BLUE}Δοκιμή και των τριών διαδρομών${NC}"
echo ""
for q in "τι κανεις;" "τι σπουδασες;" "θα ερθεις το Σαββατο;"; do
    R=$(curl -s --max-time 60 http://localhost:5678/webhook/twin-chat \
        -H "Content-Type: application/json" \
        -d "{\"message\":\"$q\"}" 2>/dev/null)
    if echo "$R" | grep -q '"reply"'; then
        REPLY=$(echo "$R" | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"[{d.get('intent','?')}] {d['reply'][:60]}\")" 2>/dev/null)
        echo -e "  ${GREEN}✓${NC} $q"
        echo -e "      $REPLY"
    else
        echo -e "  ${RED}✗${NC} $q → $(echo "$R" | head -c 70)"
    fi
done
echo ""
