#!/bin/bash
# ============================================================
# Εγκατάσταση του fine-tuned Jarvis μοντέλου στον τοπικό Ollama.
#
# Προϋπόθεση: το .gguf από το Colab στο ~/jarvis/models/
#
# Χρήση:  ./scripts/setup_local_model.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

MODEL_NAME="jarvis"
MODELS_DIR="$(pwd)/models"

echo ""
echo -e "${BLUE}Jarvis George — τοπικό μοντέλο${NC}"
echo ""

# ── 1. Βρες το .gguf ────────────────────────────────────────
GGUF=$(find "$MODELS_DIR" -maxdepth 1 -name "*.gguf" | head -1)
if [[ -z "$GGUF" ]]; then
    echo -e "${RED}Δεν βρέθηκε .gguf στο $MODELS_DIR${NC}"
    echo ""
    echo "Κατέβασε το από το Google Drive:"
    echo "  MyDrive/jarvis_models/gguf/krikri-jarvis-Q4_K_M.gguf"
    echo "και βάλ' το στο $MODELS_DIR/"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} $(basename "$GGUF") ($(du -h "$GGUF" | cut -f1))"

# ── 2. Συγχρόνισε το Modelfile με το πραγματικό όνομα αρχείου ─
sed -i.bak "s|^FROM .*|FROM ./$(basename "$GGUF")|" "$MODELS_DIR/Modelfile"
rm -f "$MODELS_DIR/Modelfile.bak"

# ── 3. Ollama container ─────────────────────────────────────
if ! docker ps --format '{{.Names}}' | grep -q '^jarvis-ollama$'; then
    echo -e "  ${YELLOW}!${NC} Ο jarvis-ollama δεν τρέχει — τον ξεκινάω"
    docker compose up -d ollama
    sleep 5
fi

# ── 4. Αντιγραφή + create ───────────────────────────────────
echo -e "  ${BLUE}→${NC} Αντιγραφή στο container (μπορεί να πάρει ~1 λεπτό)..."
docker cp "$MODELS_DIR/." jarvis-ollama:/models/

echo -e "  ${BLUE}→${NC} Δημιουργία μοντέλου '$MODEL_NAME'..."
docker exec jarvis-ollama ollama create "$MODEL_NAME" -f /models/Modelfile

# ── 5. Επαλήθευση ───────────────────────────────────────────
echo ""
echo -e "  ${BLUE}→${NC} Δοκιμή..."
REPLY=$(docker exec jarvis-ollama ollama run "$MODEL_NAME" "Γεια σου, τι κάνεις;" 2>/dev/null | head -3)
echo -e "  ${GREEN}Απάντηση:${NC} $REPLY"

# ── 6. Ενημέρωση docker-compose ─────────────────────────────
if grep -q "JARVIS_MODEL=mistral" docker-compose.yml; then
    sed -i.bak "s|JARVIS_MODEL=mistral|JARVIS_MODEL=$MODEL_NAME|" docker-compose.yml
    rm -f docker-compose.yml.bak
    echo ""
    echo -e "  ${GREEN}✓${NC} docker-compose.yml → JARVIS_MODEL=$MODEL_NAME"
    echo -e "  ${YELLOW}→${NC} Τρέξε: docker compose up -d --force-recreate jarvis-api"
fi

echo ""
echo -e "${GREEN}Έτοιμο.${NC} Το twin τρέχει τοπικά — καμία εξάρτηση από Colab/ngrok."
echo ""
