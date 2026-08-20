# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Gold: embeddings for RAG
# MAGIC 1024-dim vectors via `ai_query('databricks-gte-large-en')` →
# MAGIC `digital_twin.gold.george_embeddings` (13,785 vectors in production).
# MAGIC
# MAGIC Greek texts are translated to English first (embedding model is
# MAGIC English-optimised) — same decision as the production run.

# COMMAND ----------

CATALOG = "digital_twin"
EMBEDDING_MODEL = "databricks-gte-large-en"
GENERATION_MODEL = "databricks-meta-llama-3-3-70b-instruct"   # used for translation

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# Translate Greek rows to English (batched SQL ai_query), then embed.
spark.sql(f"""
CREATE OR REPLACE TABLE gold.george_embeddings AS
WITH source AS (
    SELECT
        concat_ws(' → ', instruction, response)             AS chunk_text,
        language, chat_id, thread_id, instruction_ts
    FROM gold.viber_training_pairs
),
translated AS (
    SELECT
        chunk_text,
        CASE
            WHEN language = 'el' THEN ai_query(
                '{GENERATION_MODEL}',
                concat('Translate to English, output only the translation: ', chunk_text)
            )
            ELSE chunk_text
        END                                                  AS embed_text,
        language, chat_id, thread_id, instruction_ts
    FROM source
)
SELECT
    chunk_text,
    embed_text,
    ai_query('{EMBEDDING_MODEL}', embed_text)                AS embedding,   -- ARRAY<FLOAT>, 1024-dim
    language, chat_id, thread_id, instruction_ts
FROM translated
""")

print(f"gold.george_embeddings rows: {spark.table('gold.george_embeddings').count()}")
