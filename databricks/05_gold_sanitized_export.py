# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Gold: PII sanitization + training export  ← THE MISSING STEP
# MAGIC Applies `jarvis.sanitization.PIISanitizer` to every pair →
# MAGIC `gold.viber_training_pairs_sanitized` + downloadable JSON for Colab.
# MAGIC
# MAGIC **This unblocks Part B (Viber) training.** The raw table/JSON must
# MAGIC never reach the trainer (GDPR + model memorisation risk).
# MAGIC
# MAGIC Setup (Free Edition friendly): upload the repo's `src/jarvis` folder
# MAGIC to your Workspace next to this notebook, plus `contacts.txt` to a
# MAGIC Volume, then run all.

# COMMAND ----------

import sys
sys.path.append("../src")                     # repo layout in Workspace
CONTACTS_PATH = "/Volumes/digital_twin/gold/config/contacts.txt"   # ← upload here (NOT in git)
EXPORT_PATH = "/Volumes/digital_twin/gold/exports/jarvis_training_data_sanitized.json"

from jarvis.sanitization import PIISanitizer

# COMMAND ----------

import json

sanitizer = PIISanitizer.from_contacts_file(CONTACTS_PATH)
print(f"contact variants loaded: {sanitizer.n_contacts}")

pdf = spark.table("gold.viber_training_pairs").toPandas()      # 13K rows — fits easily
records = pdf.to_dict("records")

clean, report = sanitizer.sanitize_records(records, fields=("instruction", "response"))
print(report.summary())

# COMMAND ----------

# Write the sanitized gold table
spark.createDataFrame(clean).write.mode("overwrite") \
    .saveAsTable("gold.viber_training_pairs_sanitized")

# Export the Colab training JSON (instruction/response only)
export = [
    {"instruction": r["instruction"], "response": r["response"]} for r in clean
]
with open(EXPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False, indent=1)

print(f"Exported {len(export)} pairs → {EXPORT_PATH}")
print("Download this file and upload it to Google Drive (MyDrive root) as")
print("jarvis_training_data_sanitized.json — then run Part B of the v3 notebook.")

# COMMAND ----------

# Safety gate: fail loudly if anything structural slipped through
from jarvis.sanitization.patterns import PII_PATTERNS

text_blob = json.dumps(export, ensure_ascii=False)
residual = []
for category, pattern, validator in PII_PATTERNS:
    for m in pattern.finditer(text_blob):
        if validator is None or validator("".join(m.group(0).split())):
            residual.append((category, m.group(0)[:6] + "…"))
assert not residual, f"Residual PII found, do NOT train: {residual[:10]}"
print("✓ Post-scan clean — safe for training.")
