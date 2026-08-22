#!/bin/bash
# ============================================================
# Δημιουργεί ΔΕΥΤΕΡΟ μοντέλο Ollama: ίδιο base Krikri, ΧΩΡΙΣ τον
# adapter. Δεν κατεβαίνει τίποτα — το GGUF είναι ήδη εκεί.
#
# Γιατί:
#   Ο adapter έμαθε να γράφει σαν τον Γιώργο σε Viber. Αυτό είναι
#   ακριβώς ό,τι θέλεις με φίλο και ακριβώς ό,τι εμποδίζει σε
#   τεχνική ερώτηση: τραβάει προς σύντομα, χαλαρά, αυτοσχεδιαστικά,
#   και υπερισχύει των οδηγιών του system prompt.
#
#   Το ίδιο base χωρίς adapter ακολουθεί οδηγίες πολύ καλύτερα,
#   γιατί δεν έχει περάσει από 13.000 μηνύματα που το τραβάνε αλλού.
#
# Δεν είναι υποχώρηση — είναι ablation study. Μετρήσιμη απάντηση
# στο «τι σου έδωσε και τι σου κόστισε το fine-tuning», που είναι
# ακριβώς η ερώτηση που θα κάνει η επιτροπή.
#
# Χρήση:  ./scripts/setup_base_model.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

MODEL_NAME="jarvis-base"
MODELS="$(pwd)/models"
BASE_GGUF="$MODELS/llama-krikri-8b-instruct-q4_k_m.gguf"

echo ""
echo -e "${BLUE}Jarvis — base μοντέλο χωρίς adapter (ablation)${NC}"
echo ""

if [[ ! -f "$BASE_GGUF" ]]; then
    echo -e "  ${RED}✗${NC} Λείπει το base GGUF: $BASE_GGUF"
    echo "    Τρέξε πρώτα ./scripts/setup_local_model.sh"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} base GGUF ($(du -h "$BASE_GGUF" | cut -f1)) — δεν κατεβαίνει τίποτα"

# ── Modelfile χωρίς ADAPTER ─────────────────────────────────
# Το SYSTEM εδώ είναι σκόπιμα ΟΥΔΕΤΕΡΟ. Χωρίς adapter δεν υπάρχει
# μαθημένο ύφος να ενισχυθεί, και το πραγματικό system prompt το
# στέλνει η εφαρμογή ανά register. Ένα ύφος καρφωμένο στο Modelfile
# θα ακύρωνε ακριβώς αυτό που το πείραμα μετράει.
cat > "$MODELS/Modelfile.base" << 'MODELFILE'
FROM ./llama-krikri-8b-instruct-q4_k_m.gguf

SYSTEM """Είσαι ο Γιώργος Τροχίδης, φοιτητής στο Πανεπιστήμιο Πελοποννήσου.
Μιλάς σε πρώτο ενικό πρόσωπο — εσύ ΕΙΣΑΙ ο Γιώργος, ΟΧΙ AI ή βοηθός.
Ακολουθείς πιστά τις οδηγίες ύφους και μήκους που σου δίνονται.
Μην επινοείς πληροφορίες. Αν δεν ξέρεις, πες το."""

PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER num_predict 320
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
MODELFILE

echo -e "  ${BLUE}→${NC} Δημιουργία μοντέλου '$MODEL_NAME'..."
if pgrep -x ollama >/dev/null 2>&1; then
    (cd "$MODELS" && ollama create "$MODEL_NAME" -f Modelfile.base)
else
    echo -e "  ${RED}✗${NC} Ο native Ollama δεν τρέχει. Τρέξε ./scripts/use_native_ollama.sh"
    exit 1
fi

echo ""
echo -e "  ${GREEN}✓${NC} Έτοιμο. Το '$MODEL_NAME' μοιράζεται το ίδιο base GGUF"
echo "    με το 'jarvis' — δεν καταλαμβάνει επιπλέον 5GB."
echo ""
echo "  Για να το χρησιμοποιεί το academic register, πρόσθεσε στο"
echo "  docker-compose.yml, στο jarvis-api → environment:"
echo ""
echo -e "      ${BLUE}- JARVIS_MODEL_ACADEMIC=$MODEL_NAME${NC}"
echo ""
echo "  και μετά:"
echo "      docker compose up -d --force-recreate jarvis-api"
echo "      ./scripts/compare_adapter.sh"
echo ""
