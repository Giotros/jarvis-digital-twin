#!/bin/bash
# ============================================================
# Στήνει το fine-tuned Jarvis μοντέλο τοπικά, χωρίς GPU.
#
# Αντί να κάνουμε merge των LoRA adapters στο base μοντέλο (που
# απαιτεί ~16GB RAM για fp16), κατεβάζουμε το επίσημο quantized
# GGUF από το ILSP και μετατρέπουμε ΜΟΝΟ το adapter σε GGUF.
# Ο Ollama τα συνδυάζει κατά την εκτέλεση με τη directive ADAPTER.
#
# Προϋπόθεση: το adapter από το Colab/Drive στο models/adapter/
#   adapter_model.safetensors
#   adapter_config.json
#
# Χρήση:  ./scripts/setup_local_model.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

MODEL_NAME="jarvis"
MODELS="$(pwd)/models"
ADAPTER_DIR="$MODELS/adapter"
BASE_GGUF="$MODELS/llama-krikri-8b-instruct-q4_k_m.gguf"
ADAPTER_GGUF="$MODELS/jarvis-adapter.gguf"
LLAMA_CPP="$MODELS/.llama.cpp"

echo ""
echo -e "${BLUE}Jarvis George — τοπικό μοντέλο (χωρίς GPU)${NC}"
echo ""

# ── 1. Έλεγχος adapter ──────────────────────────────────────
if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
    echo -e "${RED}Λείπει το adapter.${NC}"
    echo ""
    echo "Κατέβασε από το Google Drive τον φάκελο:"
    echo "  jarvis_models/krikri_qlora_v4/checkpoints/checkpoint-650/"
    echo "και βάλε ΤΟΥΛΑΧΙΣΤΟΝ αυτά τα δύο αρχεία στο $ADAPTER_DIR/ :"
    echo "  adapter_model.safetensors"
    echo "  adapter_config.json"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} adapter ($(du -h "$ADAPTER_DIR/adapter_model.safetensors" | cut -f1))"

# ── 2. Base GGUF από το ILSP ────────────────────────────────
if [[ ! -f "$BASE_GGUF" ]]; then
    echo -e "  ${BLUE}→${NC} Κατέβασμα base GGUF από ILSP (~5GB, μία φορά)..."
    curl -L --progress-bar -o "$BASE_GGUF" \
      "https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-GGUF/resolve/main/llama-krikri-8b-instruct-q4_k_m.gguf?download=true"
fi
echo -e "  ${GREEN}✓${NC} base GGUF ($(du -h "$BASE_GGUF" | cut -f1))"

# ── 3. Μετατροπή adapter → GGUF ─────────────────────────────
if [[ ! -f "$ADAPTER_GGUF" ]]; then
    if [[ ! -d "$LLAMA_CPP" ]]; then
        echo -e "  ${BLUE}→${NC} Λήψη llama.cpp..."
        git clone --depth 1 -q https://github.com/ggerganov/llama.cpp "$LLAMA_CPP"
    fi
    echo -e "  ${BLUE}→${NC} Εγκατάσταση εξαρτήσεων μετατροπής..."
    python3 -m pip install -q --user \
        torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
        python3 -m pip install -q --user torch
    python3 -m pip install -q --user transformers safetensors sentencepiece numpy gguf

    echo -e "  ${BLUE}→${NC} Μετατροπή adapter σε GGUF..."
    python3 "$LLAMA_CPP/convert_lora_to_gguf.py" \
        "$ADAPTER_DIR" \
        --base-model-id "ilsp/Llama-Krikri-8B-Instruct" \
        --outfile "$ADAPTER_GGUF" \
        --outtype f16
fi
echo -e "  ${GREEN}✓${NC} adapter GGUF ($(du -h "$ADAPTER_GGUF" | cut -f1))"

# ── 4. Modelfile ────────────────────────────────────────────
cat > "$MODELS/Modelfile" << 'MODELFILE'
FROM ./llama-krikri-8b-instruct-q4_k_m.gguf
ADAPTER ./jarvis-adapter.gguf

SYSTEM """Είσαι ο Γιώργος Τροχίδης, φοιτητής στο Πανεπιστήμιο Πελοποννήσου στην Τρίπολη.
Μιλάς ΠΑΝΤΑ σε πρώτο ενικό πρόσωπο — εσύ ΕΙΣΑΙ ο Γιώργος, ΟΧΙ AI ή βοηθός.
ΥΦΟΣ: Φιλικό, casual, σαν να μιλάς σε φίλο.
ΠΡΟΣΦΩΝΗΣΗ: ΠΑΝΤΑ δεύτερο ενικό — 'εσύ', 'σου', 'τι κάνεις'. ΠΟΤΕ πληθυντικό ευγενείας.
ΜΗΚΟΣ: 1-3 προτάσεις. Μην είσαι μονολεκτικός.
ΚΑΝΟΝΕΣ: Μην επινοείς πληροφορίες. Αν δεν ξέρεις, πες 'δεν θυμάμαι' ή 'δεν ξέρω'."""

PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 4096
PARAMETER num_predict 150
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
MODELFILE

# ── 5. Φόρτωση στον Ollama ──────────────────────────────────
if ! docker ps --format '{{.Names}}' | grep -q '^jarvis-ollama$'; then
    echo -e "  ${YELLOW}!${NC} Ξεκινάω τον jarvis-ollama..."
    docker compose up -d ollama && sleep 5
fi

echo -e "  ${BLUE}→${NC} Αντιγραφή στο container..."
docker exec jarvis-ollama mkdir -p /models
docker cp "$MODELS/Modelfile" jarvis-ollama:/models/
docker cp "$BASE_GGUF" jarvis-ollama:/models/
docker cp "$ADAPTER_GGUF" jarvis-ollama:/models/

echo -e "  ${BLUE}→${NC} Δημιουργία μοντέλου '$MODEL_NAME'..."
docker exec jarvis-ollama ollama create "$MODEL_NAME" -f /models/Modelfile

# ── 6. Δοκιμή ───────────────────────────────────────────────
echo ""
echo -e "  ${BLUE}→${NC} Δοκιμή..."
docker exec jarvis-ollama ollama run "$MODEL_NAME" "Γεια σου, τι κάνεις;" 2>/dev/null | head -5

# ── 7. Ενημέρωση docker-compose ─────────────────────────────
if grep -q "JARVIS_MODEL=mistral" docker-compose.yml; then
    sed -i.bak "s|JARVIS_MODEL=mistral|JARVIS_MODEL=$MODEL_NAME|" docker-compose.yml
    rm -f docker-compose.yml.bak
    echo ""
    echo -e "  ${GREEN}✓${NC} docker-compose.yml → JARVIS_MODEL=$MODEL_NAME"
fi

echo ""
echo -e "${GREEN}Έτοιμο.${NC} Τρέξε:  docker compose up -d --force-recreate jarvis-api"
echo ""
