# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: clean & normalize messages
# MAGIC Deduplication, timestamp parsing, language tag, conversation threading.
# MAGIC Produces `digital_twin.silver.viber_messages_clean`.

# COMMAND ----------

CATALOG = "digital_twin"
SESSION_GAP_MINUTES = 240        # >4h of silence starts a new conversation thread

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

bronze = spark.table("bronze.viber_messages_raw")

cleaned = (
    bronze
    .withColumn("ts", F.to_timestamp("timestamp"))
    .filter(F.col("ts").isNotNull() & (F.length(F.trim("text")) > 0))
    .dropDuplicates(["ts", "sender", "text"])
    # crude but effective language tag: any Greek codepoint → 'el'
    .withColumn("language", F.when(F.col("text").rlike("[\\u0370-\\u03FF]"), "el").otherwise("en"))
)

# conversation threading: new thread_id when the gap to the previous
# message in the same chat exceeds SESSION_GAP_MINUTES
w = Window.partitionBy("chat_id").orderBy("ts")
threaded = (
    cleaned
    .withColumn("prev_ts", F.lag("ts").over(w))
    .withColumn(
        "new_thread",
        (F.col("prev_ts").isNull()) |
        ((F.unix_timestamp("ts") - F.unix_timestamp("prev_ts")) > SESSION_GAP_MINUTES * 60),
    )
    .withColumn("thread_id", F.sum(F.col("new_thread").cast("int")).over(w))
    .drop("prev_ts", "new_thread")
)

threaded.select("ts", "chat_id", "thread_id", "sender", "is_me", "language", "text") \
    .write.mode("overwrite").saveAsTable("silver.viber_messages_clean")

print(f"silver.viber_messages_clean rows: {spark.table('silver.viber_messages_clean').count()}")
