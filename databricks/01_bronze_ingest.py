# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: ingest raw Viber messages
# MAGIC Medallion layer 1: land the raw export **as-is** (no cleaning here —
# MAGIC bronze is the immutable audit copy).
# MAGIC
# MAGIC ⚠ The existing table `digital_twin.bronze.viber_messages_raw` already
# MAGIC holds the production data. This script is the **reproducible path**
# MAGIC for re-ingesting (new backup, new machine): it appends, never drops.

# COMMAND ----------

CATALOG = "digital_twin"
SOURCE_PATH = "/Volumes/digital_twin/bronze/raw_uploads/viber_messages.jsonl"  # ← adjust upload path

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS bronze.viber_messages_raw (
    timestamp   STRING,
    chat_id     STRING,
    sender      STRING,
    is_me       BOOLEAN,
    text        STRING,
    _ingested_at TIMESTAMP
)
""")

# COMMAND ----------

from pyspark.sql import functions as F

raw = (
    spark.read.json(SOURCE_PATH)              # JSONL from jarvis.extraction.ViberExtractor
    .withColumn("_ingested_at", F.current_timestamp())
)
raw.write.mode("append").saveAsTable("bronze.viber_messages_raw")

print(f"bronze.viber_messages_raw rows: {spark.table('bronze.viber_messages_raw').count()}")
