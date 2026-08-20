# Jarvis George — Architecting Autonomous Digital Twins

Diploma thesis, University of Peloponnese · George Trochidis · Supervisor: Παναγιώτης Ζέρβας

An autonomous digital twin that replicates a specific person's communication style using
**Krikri-8B (Greek-native LLM) + QLoRA fine-tuning + hybrid RAG + n8n orchestration**,
with GDPR-compliant Greek PII sanitization. This is the **v5 codebase** (July 2026) —
n8n orchestration + intent routing + Docker deployment stack.

## Repository map

```
config/           settings.yaml, identity.yaml (ΟΛΕΣ οι ρυθμίσεις εδώ)
src/jarvis/
  extraction/     Phase 1 — Viber & email → normalized JSONL
  sanitization/   PII removal (Greek-specific, GDPR) ← τρέχει ΠΡΙΝ από κάθε training
  rag/            Phase 2 — hybrid search: BM25 + vectors + RRF
  training/       Phase 3 — dataset utilities (Krikri + Mistral formats)
  evaluation/     multi-model comparison (Mistral baseline vs Krikri)
  inference/      Phase 4 — FastAPI endpoint + model loader + guardrails + identity
  orchestration/  Phase 5 — intent classifier + granular API routes for n8n
databricks/       Medallion pipeline 01→05 (bronze → silver → gold → embeddings → sanitized)
notebooks/        Colab notebooks (A100)
n8n/workflows/    n8n workflow JSON files (import στο n8n UI)
scripts/          run_sanitization.py
tests/            pytest suite (81 tests) — `python -m pytest`
docs/             architecture.md (v5)
docker-compose.yml  Mac Mini M4 deployment stack
```

## Κατάσταση project (2026-07)

| Phase | Στάδιο | Κατάσταση |
|---|---|---|
| 1 | Data extraction → Databricks bronze | ✅ |
| 2 | RAG (13,785 embeddings, hybrid search) | ✅ |
| 3 | PII Sanitization (31,715 names + 1,046 regex) | ✅ |
| 3A | Mistral Persona-Chat → Viber sequential QLoRA | ✅ baseline (loss 2.70→0.25) |
| 3B | **Krikri-8B direct QLoRA** (loss 5.52→0.27) | ✅ adapter στο Drive |
| 4 | FastAPI inference + Guardrails + Identity | ✅ pipeline complete |
| 5 | RAG integration in inference | ✅ BM25 done, embeddings pending |
| 6 | **n8n orchestration + intent routing** | ✅ workflow + granular API |
| 7 | Docker deployment stack (Mac Mini M4) | ✅ docker-compose ready |
| 8 | Ray distributed training wrapper | 📋 planned |
| 9 | Evaluation framework (Mistral vs Krikri) | 📋 planned |

## Models

| Model | Role | Details |
|---|---|---|
| **Krikri-8B** (`ilsp/Llama-Krikri-8B-Instruct`) | Primary | Greek-native, 128K vocab, 56.7B Greek tokens pretraining |
| Mistral-7B (`mistralai/Mistral-7B-Instruct-v0.2`) | Baseline | 32K vocab, Greek tokenization issues |

## Development

```bash
pip install -r requirements.txt
python -m pytest
python -m ruff check src tests
```

## ⚠️ Κανόνες ασφαλείας (GDPR)

* Το `jarvis_training_data.json` (raw) **δεν μπαίνει ποτέ** σε training, git, ή chat upload.
* Το `config/contacts.txt` είναι gitignored — προσωπικά δεδομένα.
* Training **μόνο** με `jarvis_training_data_sanitized.json` που έχει περάσει το post-scan.
