#!/bin/bash
# ============================================================
# Jarvis George — Startup Script
#
# Usage:
#   ./jarvis.sh start          Tier 1: Local only (base model)
#   ./jarvis.sh start --colab  Tier 2: Full mode (fine-tuned via Colab)
#   ./jarvis.sh stop           Stop everything
#   ./jarvis.sh status         Check what's running
# ============================================================

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

check_colab() {
    local url=$(grep JARVIS_OLLAMA_URL docker-compose.yml | head -1 | cut -d= -f2)
    if curl -s --max-time 5 "$url/health" | grep -q "ok" 2>/dev/null; then
        echo -e "${GREEN}Colab ngrok is UP${NC} ($url)"
        return 0
    else
        echo -e "${YELLOW}Colab ngrok is DOWN${NC} ($url)"
        return 1
    fi
}

pull_fallback_model() {
    local model=$(grep JARVIS_FALLBACK_MODEL docker-compose.yml | head -1 | cut -d= -f2)
    model=${model:-mistral}
    echo -e "${BLUE}Checking local model '$model'...${NC}"
    if docker exec jarvis-ollama ollama list 2>/dev/null | grep -q "$model"; then
        echo -e "${GREEN}Model '$model' ready${NC}"
    else
        echo -e "${YELLOW}Pulling '$model' (first time only)...${NC}"
        docker exec jarvis-ollama ollama pull "$model"
    fi
}

cmd_start() {
    local use_colab=false
    [[ "$1" == "--colab" ]] && use_colab=true

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Jarvis George — Starting up${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Step 1: Docker stack
    echo -e "${BLUE}[1/4]${NC} Starting Docker services..."
    docker compose up -d
    echo ""

    # Step 2: Check/pull fallback model
    echo -e "${BLUE}[2/4]${NC} Checking local fallback model..."
    sleep 3  # Wait for Ollama to be ready
    pull_fallback_model
    echo ""

    # Step 3: Colab check
    echo -e "${BLUE}[3/4]${NC} Checking Colab connection..."
    if $use_colab; then
        if check_colab; then
            echo -e "${GREEN}  Full mode: fine-tuned model via Colab${NC}"
        else
            echo -e "${YELLOW}  Colab not reachable — will use local fallback${NC}"
            echo -e "${YELLOW}  Start Colab notebook and run again with --colab${NC}"
        fi
    else
        echo -e "  Skipping Colab (local-only mode)"
        echo -e "  Use ${YELLOW}./jarvis.sh start --colab${NC} for fine-tuned model"
    fi
    echo ""

    # Step 4: Activate n8n workflow
    echo -e "${BLUE}[4/4]${NC} Activating n8n workflow..."
    sleep 2
    docker exec jarvis-n8n n8n publish:workflow --id=tBPEVTfphTTV0xLr 2>/dev/null || true
    docker restart jarvis-n8n 2>/dev/null
    sleep 3
    echo ""

    # Health checks
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Status${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # API
    if curl -s --max-time 3 http://localhost:8000/health | grep -q "ok" 2>/dev/null; then
        echo -e "  ${GREEN}API${NC}        http://localhost:8000"
    else
        echo -e "  ${RED}API${NC}        not responding"
    fi

    # n8n
    if curl -s --max-time 3 http://localhost:5678/webhook/twin-chat \
        -H "Content-Type: application/json" \
        -d '{"message":"health"}' 2>/dev/null | grep -q "reply" 2>/dev/null; then
        echo -e "  ${GREEN}Webhook${NC}    http://localhost:5678/webhook/twin-chat"
    else
        echo -e "  ${YELLOW}Webhook${NC}    activating... (wait 10s and retry)"
    fi

    # Frontend
    echo -e "  ${GREEN}Frontend${NC}   http://localhost:3001"
    echo ""

    if $use_colab && check_colab >/dev/null 2>&1; then
        echo -e "  ${GREEN}Mode: FULL${NC} (fine-tuned Krikri via Colab)"
    else
        echo -e "  ${YELLOW}Mode: LOCAL${NC} (base Mistral via Ollama)"
    fi
    echo ""
    echo -e "  Open ${BLUE}http://localhost:3001${NC} to chat"
    echo ""
}

cmd_stop() {
    echo -e "${BLUE}Stopping Jarvis George...${NC}"
    docker compose down
    echo -e "${GREEN}Done.${NC}"
}

cmd_status() {
    echo ""
    echo -e "${BLUE}Jarvis George — Status${NC}"
    echo ""
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || docker compose ps
    echo ""
    check_colab 2>/dev/null || true
    echo ""
}

case "${1:-}" in
    start)  cmd_start "$2" ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    *)
        echo "Usage: ./jarvis.sh {start|stop|status}"
        echo ""
        echo "  start          Local-only mode (base model)"
        echo "  start --colab  Full mode (fine-tuned via Colab)"
        echo "  stop           Stop all services"
        echo "  status         Check what's running"
        ;;
esac
