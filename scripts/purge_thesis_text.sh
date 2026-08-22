#!/bin/bash
# ============================================================
# Αφαιρεί το κείμενο της διπλωματικής από το δημόσιο repo — και
# από το ιστορικό του.
#
# Το .gitignore σταματά τα ΜΕΛΛΟΝΤΙΚΑ commits. Δεν αγγίζει ό,τι
# έχει ήδη ανέβει: τα αρχεία παραμένουν προσβάσιμα σε όποιον
# ξέρει να ψάξει παλιά commits, και το GitHub τα σερβίρει.
#
# Το ίδιο έγινε και για το config/identity.yaml παλιότερα.
#
# Χρήση:  ./scripts/purge_thesis_text.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

PATHS=(
    "docs/bibliography.md"
    "docs/thesis_skeleton.md"
    "docs/chapter_04_anonymisation.md"
    "docs/chapter_05_distributed.md"
    "docs/chapter_08_evaluation.md"
)

echo ""
echo -e "${BLUE}Αφαίρεση κειμένου διπλωματικής από το repo${NC}"
echo ""
echo "Θα αφαιρεθούν από την παρακολούθηση ΚΑΙ από το ιστορικό:"
for p in "${PATHS[@]}"; do echo "  · $p"; done
echo ""
echo -e "${YELLOW}Αυτό ξαναγράφει το ιστορικό και απαιτεί force push.${NC}"
echo "Τα τοπικά αρχεία ΔΕΝ διαγράφονται — μόνο βγαίνουν από το git."
echo ""
read -rp "Συνέχεια; [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Ακυρώθηκε."; exit 0; }

# ── 1. Έξοδος από την παρακολούθηση ─────────────────────────
echo ""
echo -e "  ${BLUE}→${NC} Αφαίρεση από το index..."
for p in "${PATHS[@]}"; do
    git rm --cached --quiet --ignore-unmatch "$p"
done
git commit -q -m "Keep thesis text out of the public repository" || true

# ── 2. Καθαρισμός ιστορικού ─────────────────────────────────
echo -e "  ${BLUE}→${NC} Ξαναγράφω το ιστορικό (μπορεί να πάρει λίγο)..."
FILTER=""
for p in "${PATHS[@]}"; do FILTER+=" '$p'"; done

FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --index-filter \
    "git rm --cached --ignore-unmatch$FILTER" \
    --prune-empty --tag-name-filter cat -- --all >/dev/null 2>&1

# ── 3. Τοπική εκκαθάριση ────────────────────────────────────
echo -e "  ${BLUE}→${NC} Εκκαθάριση αναφορών..."
rm -rf .git/refs/original
git reflog expire --expire=now --all
git gc --prune=now --aggressive >/dev/null 2>&1

# ── 4. Επαλήθευση ───────────────────────────────────────────
echo ""
echo -e "${BLUE}Επαλήθευση στο ιστορικό${NC}"
CLEAN=1
for p in "${PATHS[@]}"; do
    if git log --all --oneline -- "$p" 2>/dev/null | grep -q .; then
        echo -e "  ${RED}✗${NC} $p υπάρχει ακόμα στο ιστορικό"
        CLEAN=0
    else
        echo -e "  ${GREEN}✓${NC} $p καθαρό"
    fi
    [[ -f "$p" ]] && echo -e "      (το τοπικό αρχείο διατηρήθηκε)"
done

echo ""
if [[ $CLEAN -eq 1 ]]; then
    echo -e "  ${GREEN}Το ιστορικό είναι καθαρό.${NC} Τελευταίο βήμα:"
    echo ""
    echo -e "      ${BLUE}git push origin --force --all${NC}"
    echo ""
    echo "  Μετά το push, έλεγξε ότι δεν φαίνονται:"
    echo "      https://github.com/Giotros/jarvis-digital-twin/tree/main/docs"
else
    echo -e "  ${RED}Κάτι έμεινε — μην κάνεις push.${NC}"
    exit 1
fi
