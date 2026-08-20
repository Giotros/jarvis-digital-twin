# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold: instruction-response training pairs
# MAGIC Pairs each incoming message (them) with George's next reply (me) inside
# MAGIC the same thread → `digital_twin.gold.viber_training_pairs`
# MAGIC (13,289 pairs in the production run).
# MAGIC
# MAGIC ⚠ Output of this table is RAW (contains PII). Training must use
# MAGIC `gold.viber_training_pairs_sanitized` produced by script 05.

# COMMAND ----------

CATALOG = "digital_twin"
MAX_REPLY_GAP_MINUTES = 240      # reply must come within the thread gap window

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

msgs = spark.table("silver.viber_messages_clean")

w = Window.partitionBy("chat_id", "thread_id").orderBy("ts")
paired = (
    msgs
    .withColumn("next_text", F.lead("text").over(w))
    .withColumn("next_is_me", F.lead("is_me").over(w))
    .withColumn("next_ts", F.lead("ts").over(w))
    # pair = (them → me) within the gap window
    .filter(
        (~F.col("is_me")) & F.col("next_is_me") &
        ((F.unix_timestamp("next_ts") - F.unix_timestamp("ts")) <= MAX_REPLY_GAP_MINUTES * 60)
    )
    .select(
        F.col("text").alias("instruction"),
        F.col("next_text").alias("response"),
        F.col("ts").alias("instruction_ts"),
        "chat_id", "thread_id", "language",
    )
    .filter((F.length("instruction") >= 2) & (F.length("response") >= 2))
)

paired.write.mode("overwrite").saveAsTable("gold.viber_training_pairs")
print(f"gold.viber_training_pairs rows: {spark.table('gold.viber_training_pairs').count()}")
