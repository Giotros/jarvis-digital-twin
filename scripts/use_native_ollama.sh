#!/bin/bash
# ============================================================
# Μεταφέρει το inference από τον Ollama-σε-Docker στον native
# Ollama του macOS, ώστε να χρησιμοποιηθεί το Metal (GPU).
#
# Γιατί: τα containers στο macOS τρέχουν μέσα σε VM Linux και
# δεν έχουν πρόσβαση στη GPU της Apple. Ο Ollama σε Docker
# κάνει inference σε CPU (~3-8 tok/s για 8B Q4). Ο native
# Ollama χρησιμοποιεί Metal (~30-50 tok/s).
#
# Τα υπόλοιπα services (n8n, chromadb, jarvis-api) μένουν σε
# Docker· μόνο το inference βγαίνει έξω.
#
# Χρήση:  ./scripts/use_native_ollama.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

MODELS="$(pwd)/models"

echo ""
echo -e "${BLUE}Μεταφορά inference σε native Ollama (Metal)${NC}"
echo ""

# ── 1. Εγκατάσταση ──────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
    if ! command -v brew >/dev/null 2>&1; then
        echo -e "${RED}Χρειάζεται Homebrew.${NC} https://brew.sh"
        exit 1
    fi
    echo -e "  ${BLUE}→${NC} Εγκατάσταση Ollama..."
    brew install ollama
fi
echo -e "  ${GREEN}✓${NC} ollama $(ollama --version 2>/dev/null | head -1)"

# ── 2. Σταμάτα ΠΡΩΤΑ τον Docker Ollama ──────────────────────
# Και οι δύο δεσμεύουν την 11434. Αν δεν φύγει πρώτα ο Docker, ο έλεγχος
# υγείας παρακάτω βλέπει ΕΚΕΙΝΟΝ και συμπεραίνει λανθασμένα ότι ο native
# είναι ήδη ενεργός — οπότε δεν τον ξεκινά ποτέ.
if docker ps --format '{{.Names}}' | grep -q '^jarvis-ollama$'; then
    echo -e "  ${BLUE}→${NC} Σταμάτημα του Ollama σε Docker..."
    docker compose stop ollama
    sleep 2
fi

# ── 3. Εκκίνηση υπηρεσίας ───────────────────────────────────
# Κράτα το μοντέλο μόνιμα στη μνήμη. Ο Ollama το ξεφορτώνει μετά από 5'
# αδράνειας· σε παρουσίαση αυτό σημαίνει 15-20s παύση στην πρώτη ερώτηση
# μετά από κάθε διάλειμμα, που μοιάζει με κόλλημα.
export OLLAMA_KEEP_ALIVE=-1
launchctl setenv OLLAMA_KEEP_ALIVE -1 2>/dev/null || true

if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "  ${BLUE}→${NC} Εκκίνηση υπηρεσίας..."
    brew services start ollama
    for _ in $(seq 1 20); do
        curl -s --max-time 1 http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi
echo -e "  ${GREEN}✓${NC} υπηρεσία ενεργή στο :11434"

# ── 4. Δημιουργία μοντέλου ──────────────────────────────────
if [[ ! -f "$MODELS/Modelfile" ]]; then
    echo -e "${RED}Λείπει το $MODELS/Modelfile — τρέξε πρώτα setup_local_model.sh${NC}"
    exit 1
fi

echo -e "  ${BLUE}→${NC} Δημιουργία μοντέλου 'jarvis'..."
( cd "$MODELS" && ollama create jarvis -f Modelfile )

# ── 5. Μέτρηση ταχύτητας ────────────────────────────────────
echo ""
echo -e "  ${BLUE}→${NC} Μέτρηση..."
START=$(date +%s)
REPLY=$(ollama run jarvis "Γεια σου, τι κάνεις;" 2>/dev/null | head -3)
ELAPSED=$(( $(date +%s) - START ))
echo -e "  ${GREEN}Απάντηση:${NC} $REPLY"
echo -e "  ${GREEN}Χρόνος:${NC} ${ELAPSED}s"

# ── 6. Στρέψε την jarvis-api στον host ──────────────────────
if grep -q "JARVIS_OLLAMA_URL=http://ollama:11434" docker-compose.yml; then
    sed -i.bak "s|JARVIS_OLLAMA_URL=http://ollama:11434|JARVIS_OLLAMA_URL=http://host.docker.internal:11434|" docker-compose.yml
    rm -f docker-compose.yml.bak
    echo ""
    echo -e "  ${GREEN}✓${NC} docker-compose.yml → host.docker.internal:11434"
fi

echo ""
echo -e "${GREEN}Έτοιμο.${NC} Τρέξε:  docker compose up -d --force-recreate jarvis-api"
echo ""
echo -e "Επαλήθευση:"
echo -e "  curl -s http://localhost:5678/webhook/twin-chat \\"
echo -e "    -H 'Content-Type: application/json' -d '{\"message\":\"γεια σου\"}' | jq ."
echo ""
echo -e "${YELLOW}Πριν την παρουσίαση:${NC} τρέξε μία ερώτηση 10' νωρίτερα ώστε το"
echo -e "μοντέλο να είναι φορτωμένο. Με OLLAMA_KEEP_ALIVE=-1 μένει στη μνήμη."
echo ""
