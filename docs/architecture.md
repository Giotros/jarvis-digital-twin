# Jarvis George — System Architecture (v5)

*Updated: July 2026 · n8n orchestration + intent routing + Docker stack*

## 1. Overview

Jarvis George is an autonomous digital twin: an AI agent that replicates George's
communication style and can, under human supervision, handle written communication in his
authentic voice. The system combines **weight-level adaptation** (QLoRA fine-tuning — captures
*how* George writes) with **retrieval grounding** (hybrid RAG — supplies *what* George knows),
and a **guardrails pipeline** (post-processing — ensures clean, polite output), over a
governed medallion data platform with GDPR-compliant PII sanitization.

```
 Viber backup / email (mbox)
        │  src/jarvis/extraction
        ▼
┌─────────────────────────── Databricks (digital_twin) ───────────────────────────┐
│ bronze.viber_messages_raw   →  silver.viber_messages_clean                      │
│      (as-is audit copy)         (dedupe, timestamps, language, threading)       │
│                                        │                                        │
│              ┌─────────────────────────┴──────────────┐                         │
│              ▼                                        ▼                         │
│ gold.viber_training_pairs (13,289 — RAW/PII)   gold.george_embeddings (13,785)  │
│              │ 05_gold_sanitized_export               │ ai_query(gte-large-en)  │
│              ▼                                        │                         │
│ gold.viber_training_pairs_sanitized ──► export JSON   │                         │
└──────────────│────────────────────────────────────────│─────────────────────────┘
               ▼                                        ▼
   Colab A100 — QLoRA fine-tuning              Hybrid RAG (BM25 + vector + RRF,
   Krikri-8B direct training ✅                 threshold 0.62, GR→EN translation)
   (Mistral sequential = baseline)                      │
               │                                        │
               └───────────► Phase 4: FastAPI ◄─────────┘
                             (Krikri + RAG + Guardrails)
                             (toggle ON/OFF, PII leak guard)
                                      │
                        Phase 5: n8n orchestration + agentic workflows
```

## 2. Technology stack

| Component | Technology |
|---|---|
| **Primary model** | **Krikri-8B** (`ilsp/Llama-Krikri-8B-Instruct`) — Greek-native, 128K vocab |
| Baseline model | Mistral-7B-Instruct-v0.2 (for evaluation comparison) |
| Fine-tuning | QLoRA — PEFT + TRL, LoRA r=64, α=128, 4-bit NF4 (bitsandbytes) |
| Training infra | Google Colab Pro, A100 40GB |
| Distributed training | Ray Train (thesis requirement — scalability demonstration) |
| Data platform | Databricks Free Edition, Unity Catalog `digital_twin`, Delta tables |
| Embeddings | databricks-gte-large-en (1024-dim) |
| RAG generation | databricks-meta-llama-3-3-70b-instruct |
| PII removal | Custom Greek `PIISanitizer` (src/jarvis/sanitization) |
| Inference | FastAPI + Guardrails pipeline (Phase 4) |
| Orchestration | n8n — Guided Flows + Guardrails nodes (Phase 5) |
| Local deployment | Ollama + Open WebUI on Mac Mini M4 |

## 3. Training design

### 3.1 Primary: Krikri-8B Direct Training

**Direct fine-tuning** on 13,289 sanitized Viber pairs. Krikri is already instruction-tuned
in Greek (56.7B Greek tokens pretraining, 857K instruction pairs by ILSP/Athena RC), so no
Persona-Chat warm-up step is needed. The model's 128K extended Greek tokenizer means
clean Greek output without the token fragmentation issues seen in Mistral.

Training result: loss 5.52 → 0.27 over 2,493 steps (~2h on A100).
Adapter saved at: `Drive/jarvis_models/krikri_qlora/` (320.1 MB).

**Chat format** (Llama 3.1 template via `tokenizer.apply_chat_template`):
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>
```

### 3.2 Baseline: Mistral-7B Sequential Training

**Sequential fine-tuning** (validated by PersonaGPT, Tang et al. 2021): Part A trained on
~109K Synthetic-Persona-Chat pairs, Part B continued on 13,289 sanitized Viber pairs with
lower LR (1e-4 vs 2e-4). Loss 2.70 → 0.25 over 2,493 steps.

Known issue: Mistral's 32K vocabulary tokenizer fragments Greek text into small subword
pieces, causing mixed-language output ("συντεθigma", "στ Greek Κυριακη"). This validates
the switch to Krikri-8B.

**Data format** (Mistral instruction format):
```
<s>[INST] {instruction} [/INST] {response}</s>
```

## 4. Inference pipeline

```
User message
  → RAG retrieval (HybridSearcher: BM25 + vector + RRF)
  → Krikri-8B generation (QLoRA adapter, system prompt + RAG context)
  → Guardrails pipeline:
      1. Clean emoji artifacts: (laugh), (purple_heart) → removed
      2. Remove hallucinated names: παναγιωτη, χρηστο → removed
      3. Profanity filter: γαμησετα → "άστα να πάνε"
      4. Remove impolite words: ρε → removed
      5. Restore Greek accents: ειναι → είναι (200+ word dictionary)
      6. Capitalize sentences: first letter uppercase
  → PII leak guard (defence-in-depth)
  → Clean response
```

## 5. RAG design

Hybrid retrieval fuses BM25 (exact tokens: names, terms) with dense vectors (paraphrase)
via **Reciprocal Rank Fusion** `RRF(d) = Σ 1/(k + rank)`, k=60. A **relevance threshold
(0.62)** on vector similarity acts as the anti-hallucination gate: below it, the twin
answers without grounding rather than grounding on noise. Greek queries are translated to
English before embedding (gte-large-en is English-optimised). Production retrieval runs on
Databricks; `src/jarvis/rag/hybrid_search.py` is the identical-algorithm reference
implementation used for tests and local evaluation.

## 6. Privacy pipeline (GDPR-critical)

**Threat model**: LLMs memorise training data (Carlini et al. 2021) → any PII present at
training time is recoverable from weights. Therefore sanitization happens **before**
training; the post-generation leak guard in the API is defence-in-depth only.

Detection order (v2-validated, kept): IBAN → ΑΜΚΑ (11 digits + birth-date check) → ΑΦΜ
(9 digits + mod-11 **checksum**) → ταυτότητα → phones (mobile 69…, landline 2…, optional
+30) → emails → known contacts (324-name dictionary, all grammatical cases + accent-stripped
variants) → generic `Firstname Surname` pattern with a non-name guard list.

**v3 fix**: the Greek lowercase class now includes final sigma **ς** — names like Γιώργος,
Νίκος, Βασίλης were escaping detection in v2. Unit-tested (`tests/test_pii_sanitizer.py`).

Sanitization results: 31,715 name replacements + 1,046 regex replacements across 13,289
pairs. Zero remaining PII confirmed via post-scan assert.

## 7. Evaluation plan (thesis §Evaluation)

1. **Multi-model comparison** (Mistral baseline vs Krikri primary):
   identical probe prompts answered by both models.
2. **Perplexity & F1** on held-out pairs.
3. **Greek language quality**: token fragmentation analysis, accent correctness,
   vocabulary coherence (key differentiator: Krikri vs Mistral).
4. **NLI-based faithfulness** (Synthetic-Persona-Chat method) — planned.
5. **Human Turing-style test** with people who know George — planned.

## 8. Phase 5 — n8n Orchestration & Agentic Control

### 8.1 Architecture overview

The monolithic `/chat` endpoint (api.py v5) still works for direct use. For agentic
orchestration, the API exposes **granular endpoints** under `/orchestration/*` that n8n
calls as separate workflow nodes:

```
n8n Workflow: "Jarvis George — Digital Twin v1"
═══════════════════════════════════════════════

  [Webhook: POST /twin-chat]
          │
          ▼
  [Toggle Check: GET /health → twin_enabled?]
          │
     ┌────┴────┐
     │ OFF     │ ON
     ▼         ▼
  [423]   [Intent Classifier: POST /orchestration/intent]
                │
         ┌──────┼──────────┬────────────┐
         ▼      ▼          ▼            ▼
     SENSITIVE  PERSONAL   KNOWLEDGE   CASUAL
         │      │          │            │
         ▼      ▼          ▼            ▼
     [Human]  [Identity]  [RAG]     [Generate
      review   Lookup     Search     direct]
                │          │            │
                ▼          ▼            │
            [Generate:  [Generate:     │
             + identity  + RAG ctx]    │
             context]      │           │
                │          │           │
                └──────────┴───────────┘
                           │
                    [Fact Check: POST /orchestration/fact-check]
                           │
                    [Guardrails: POST /orchestration/guardrails]
                           │
                    [Length Check: max 3 sentences]
                           │
                    [Respond: JSON]
```

### 8.2 Granular API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/orchestration/intent` | POST | Classify message → personal/knowledge/casual/sensitive |
| `/orchestration/identity` | POST | Look up static identity (config/identity.yaml) |
| `/orchestration/rag` | POST | Search conversation corpus (BM25/hybrid) |
| `/orchestration/generate` | POST | Generate response via Ollama API |
| `/orchestration/guardrails` | POST | Post-processing (profanity, accents, capitalize) |
| `/orchestration/fact-check` | POST | Validate response against identity data |

### 8.3 Intent classification

Lightweight keyword/pattern classifier (no LLM call needed for routing):

| Intent | Routing | Example |
|---|---|---|
| `personal` | Identity lookup → Generate with identity context | "Πόσο χρονών είσαι;" |
| `knowledge` | RAG search → Generate with RAG context | "Πώς να κάνω deploy;" |
| `casual` | Direct generation (no retrieval) | "Γεια σου!" |
| `sensitive` | Human review — no AI response | "Δώσε μου τον IBAN" |

Priority order: SENSITIVE > PERSONAL > KNOWLEDGE > CASUAL. Default (no match): KNOWLEDGE.

### 8.4 Fact-checking node

Post-generation validation against identity data:
- Detects hallucinated programming languages (GoLang, Ruby, Swift, etc.)
- Validates age claims against `identity.yaml`
- Checks mentioned companies against known career history
- Removes sentences containing hallucinated facts

### 8.5 Deployment stack (Docker Compose)

```
Mac Mini M4 (docker-compose.yml)
├── jarvis-n8n      (port 5678) — workflow orchestration
├── jarvis-ollama   (port 11434) — LLM inference (Krikri-8B)
├── jarvis-webui    (port 3000) — Open WebUI chat interface
├── jarvis-chromadb (port 8100) — vector store for embeddings
└── jarvis-api      (port 8000) — FastAPI granular endpoints
```

### 8.6 Safety boundary

The **master ON/OFF toggle** lives in the FastAPI layer (`/twin/toggle`), NOT in n8n.
This means no n8n workflow, external integration, or API consumer can bypass the kill
switch. The toggle is checked at the workflow entry point (first node after webhook).

## 9. Resolved issues ledger

| Issue | Resolution |
|---|---|
| Mistral Greek token fragmentation | Switched to Krikri-8B (128K Greek vocab) |
| Model output: no accents/capitals | Guardrails pipeline (accent restoration + capitalize) |
| Model output: profanity | Profanity filter with replacement dictionary |
| Model output: hallucinated names | Name hallucination filter |
| Runtime crash at step 83 (save_steps=500) | save_steps=50 everywhere |
| Adapters lost when save cell never ran | save in the same cell as training |
| Part B reloaded fresh model (v1) | Part B structurally continues from Part A adapters |
| ς missing from name regex | fixed + unit tests |
| Random digit runs masked as ΑΦΜ/ΑΜΚΑ | checksum + date validation |
| Config scattered across cells/scripts | single `config/settings.yaml` |
