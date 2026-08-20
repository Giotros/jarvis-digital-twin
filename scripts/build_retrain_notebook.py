"""Generate the Colab retrain + GGUF-export notebook.

The notebook is generated rather than hand-edited so that the training
configuration stays in one place and cannot drift from the repository. Run:

    python scripts/build_retrain_notebook.py

Output: notebooks/Jarvis_George_Retrain_v4_and_Export.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "Jarvis_George_Retrain_v4_and_Export.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md("""# Jarvis George — Retrain (v4, PII-clean) + GGUF Export

**Διπλωματική:** Architecting Autonomous Digital Twins — Παν. Πελοποννήσου
**Επιβλέπων:** Παναγιώτης Ζέρβας

Αυτό το notebook τρέχει **μία φορά** και βγάζει το τελικό μοντέλο:

| Μέρος | Τι κάνει | Χρόνος |
|---|---|---|
| A | Έλεγχοι + δεδομένα v4 | 5' |
| B | QLoRA fine-tune μέσω **Ray Train** | ~2-3h |
| C | Αξιολόγηση (probes) | 5' |
| D | Merge adapters → GGUF → Drive | ~40' |

**Γιατί retrain:** ο έλεγχος GDPR βρήκε ονόματα τρίτων σε 1.070/13.289
εγγραφές του παλιού corpus. Τα προηγούμενα adapters εκπαιδεύτηκαν πάνω σε
αυτά. Το `v4` είναι καθαρό (0 ονόματα) — βλ. `src/jarvis/sanitization/greek_names.py`.

> **Runtime → Change runtime type → GPU (A100 ή L4 προτιμότερα, T4 δουλεύει).**
> Μην κλείσεις το tab. Τα checkpoints σώζονται κάθε 50 steps στο Drive.
"""),

    md("## Cell 1 — Έλεγχος GPU"),
    code("""import subprocess, torch

if not torch.cuda.is_available():
    raise RuntimeError("Δεν βρέθηκε GPU — Runtime > Change runtime type > GPU")

name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name}  |  VRAM: {vram:.1f} GB  |  GPUs: {torch.cuda.device_count()}")

if vram < 15:
    print("\\n⚠️  <15GB VRAM: κράτα max_seq_length=512 και batch=1.")
"""),

    md("## Cell 2 — Πακέτα\n*(ξανατρέξέ το μετά από κάθε restart)*"),
    code("""%pip install -q "transformers>=4.44,<5" "trl>=0.9,<2" "peft>=0.11" \\
    "bitsandbytes>=0.43" "accelerate>=0.33" "datasets>=2.20" \\
    "ray[train]>=2.9" sentencepiece protobuf
print("OK — αν ζητήσει restart, κάνε restart και ξανατρέξε ΜΟΝΟ αυτό το cell.")
"""),

    md("## Cell 3 — Google Drive"),
    code("""from google.colab import drive
drive.mount("/content/drive")
"""),

    md("""## Cell 4 — Config

Ένα σημείο ρυθμίσεων. Το `base_model` είναι το Krikri-8B: ελληνικό μοντέλο,
σαφώς καλύτερο από Mistral για ελληνικά chat δεδομένα (93.8% του corpus).
"""),
    code('''from pathlib import Path

DRIVE = Path("/content/drive/MyDrive")

CFG = {
    "base_model": "ilsp/Llama-Krikri-8B-Instruct",

    # Δεδομένα — v4 = PII-clean (0 ονόματα τρίτων)
    "data_v4":      DRIVE / "jarvis_training_data_v4.json",
    "data_old":     DRIVE / "jarvis_training_data_sanitized.json",  # μόνο για guard
    "golden":       DRIVE / "golden_examples.yaml",                  # προαιρετικό

    # Έξοδος
    "out_adapters": DRIVE / "jarvis_models/krikri_qlora_v4",
    "out_merged":   Path("/content/krikri_merged"),
    "out_gguf":     DRIVE / "jarvis_models/gguf",
    "eval_out":     DRIVE / "jarvis_models/eval_v4.json",

    # LoRA — 7 modules (τα προηγούμενα runs είχαν 4· περισσότερα = καλύτερη προσαρμογή ύφους)
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],

    # Optimisation
    "learning_rate": 1e-4,
    "num_epochs": 2,
    "batch_size": 1,
    "grad_accum": 16,
    "max_seq_length": 512,
    "save_steps": 50,
    "seed": 42,

    # Ray
    "num_workers": 1,          # Colab = 1 GPU. Άλλαξέ το σε cluster.
    "quant": "Q4_K_M",         # ~4.9GB — χωράει άνετα στα 24GB του Mac Mini
}

CFG["out_adapters"].mkdir(parents=True, exist_ok=True)
CFG["out_gguf"].mkdir(parents=True, exist_ok=True)
print("Config OK")
print(f"  base   : {CFG['base_model']}")
print(f"  data   : {CFG['data_v4']}")
print(f"  output : {CFG['out_adapters']}")
'''),

    md("""---
## Cell 5 — 🛡️ GDPR GUARD

**Το training δεν ξεκινά αν τα δεδομένα δεν είναι καθαρά.**

Ο έλεγχος είναι ανεξάρτητος από το sanitization: ψάχνει ονόματα με τη
μορφολογία που διέφυγε την πρώτη φορά (πεζά, άτονα, κλητική). Αν βρει
έστω ένα, σταματάει.

> Ανέβασε πρώτα το `data/jarvis_training_data_v4.json` στο MyDrive.
"""),
    code('''import json, re, unicodedata

def _norm(t: str) -> str:
    d = unicodedata.normalize("NFD", t.casefold())
    return "".join(c for c in d if not unicodedata.combining(c))

# Θέματα ονομάτων + καταλήξεις — καθρέφτης του src/jarvis/sanitization/greek_names.py
STEMS = ("παναγιωτ δημητρ γιανν ιωανν κωνσταντιν κωστ αποστολ αθανασ βασιλ νικολ "
         "χρηστ μιχαλ αντων στελ πετρ θεοδωρ σπυρ ανδρε αλεξανδρ μανωλ ηλια "
         "λευτερ σωτηρ χαραλαμπ λαμπρ μανθ θωμα στεφαν γρηγορ παυλ μαρκ "
         "μαρι ελεν κατεριν σοφι γιωτ δημητρα βασιλικ αναστασ αννα χριστιν "
         "δεσποιν ευαγγελ αγγελικ ειρην φωτειν χρυσ σταυρ παρασκευ").split()
ENDINGS = ("ιτσα","ουλα","ακης","ακη","ος","ας","ης","ες","ου","ων","α","η","ο","ε","ς","")
EXACT = ("νικος","νικο","ευα","τασος","φωτης","ζωη")

# Λέξεις που ΜΟΙΑΖΟΥΝ με ονόματα αλλά δεν είναι. Πρέπει να μένει
# συγχρονισμένο με το src/jarvis/sanitization/greek_names.py — μια παλιά
# έκδοση αυτού του guard σήμανε το "Παρασκευή" ως διαρροή 122 φορές.
BLOCK = {
    # κοινό λεξιλόγιο
    "νικη","νικης","νικο","χαρα","ελπιδα","φως","φωτα","φωτο","μαρκα","ωρα","ωρες",
    # μέρες & μήνες που είναι ΚΑΙ ονόματα
    "παρασκευη","παρασκευης","παρασκευες","κυριακη","κυριακης","κυριακες",
    "ιουλιος","ιουλιου","ιουλιο","ιουλη","ιουνιος","ιουνιου","ιουνιο",
    "μαρτιος","μαρτιου","μαρτιο","αυγουστος","αυγουστου","αυγουστο",
    # τοπωνύμια χτισμένα σε θέματα ονομάτων
    "γιαννιτσα","γιαννιτσων","γιαννιτσας","μαρκοπουλο","μαρκοπουλου",
    "αγιαννη","αγιαννης",
}

_stem_alt = "|".join(sorted(map(_norm, STEMS), key=len, reverse=True))
_end_alt  = "|".join(sorted({_norm(e) for e in ENDINGS}, key=len, reverse=True))
_exact_alt = "|".join(sorted(map(_norm, EXACT), key=len, reverse=True))
NAME_RE = re.compile(rf"\\b(?:(?:{_stem_alt})(?:{_end_alt})|(?:{_exact_alt}))\\b")
BLOCK_N = {_norm(w) for w in BLOCK}
SELF = tuple(_norm(s) for s in ("γιωργ", "george", "giorgos"))

def audit(path):
    records = json.load(open(path, encoding="utf-8"))
    leaks, n_leak = {}, 0
    for r in records:
        for f in ("instruction_clean", "response_clean", "conversation_with"):
            for m in NAME_RE.finditer(_norm(r.get(f) or "")):
                w = m.group(0)
                if w in BLOCK_N or any(w.startswith(s) for s in SELF):
                    continue
                leaks[w] = leaks.get(w, 0) + 1
                n_leak += 1
    return len(records), n_leak, leaks

if not CFG["data_v4"].exists():
    raise FileNotFoundError(
        f"Λείπει το {CFG['data_v4']}\\n"
        "Ανέβασε το data/jarvis_training_data_v4.json στο MyDrive και ξανατρέξε."
    )

n, leaks, detail = audit(CFG["data_v4"])
print(f"v4: {n:,} εγγραφές  |  διαρροές: {leaks}")

if CFG["data_old"].exists():
    n_o, leaks_o, _ = audit(CFG["data_old"])
    print(f"παλιό (σύγκριση): {n_o:,} εγγραφές  |  διαρροές: {leaks_o}")

if leaks:
    top = sorted(detail.items(), key=lambda x: -x[1])[:10]
    raise SystemExit(f"❌ ΣΤΑΜΑΤΗΜΑ — βρέθηκαν ονόματα: {top}")

print("\\n✅ GUARD OK — καθαρά δεδομένα, το training μπορεί να ξεκινήσει.")
'''),

    md("""## Cell 6 — Φόρτωση δεδομένων (+ golden examples)

Τα golden examples είναι χειροκίνητα «σωστά» παραδείγματα. Επαναλαμβάνονται
x3 ώστε να έχουν βάρος απέναντι στις 13k αυτόματες εγγραφές — 20 παραδείγματα
σε 13.000 αλλιώς χάνονται στον θόρυβο.
"""),
    code('''import json, random

records = json.load(open(CFG["data_v4"], encoding="utf-8"))

def format_pair(instruction: str, response: str) -> str:
    """Krikri/Llama-3 chat template."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
        f"{instruction.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
        f"{response.strip()}<|eot_id|>"
    )

texts = [
    format_pair(r.get("instruction_clean", ""), r.get("response_clean", ""))
    for r in records
    if (r.get("instruction_clean") or "").strip() and (r.get("response_clean") or "").strip()
]
print(f"Από corpus : {len(texts):,}")

# Golden examples — προαιρετικά, με βάρος
GOLDEN_WEIGHT = 3
n_golden = 0
if CFG["golden"].exists():
    import yaml
    g = yaml.safe_load(open(CFG["golden"], encoding="utf-8")) or {}
    for ex in g.get("examples", []):
        resp = (ex.get("response") or "").strip()
        if resp:
            texts.extend([format_pair(ex["prompt"], resp)] * GOLDEN_WEIGHT)
            n_golden += 1
    print(f"Golden     : {n_golden} × {GOLDEN_WEIGHT}")
else:
    print("Golden     : — (δεν βρέθηκε golden_examples.yaml, προαιρετικό)")

random.Random(CFG["seed"]).shuffle(texts)
print(f"ΣΥΝΟΛΟ     : {len(texts):,} δείγματα")
print("\\n--- δείγμα ---")
print(texts[0][:300])
'''),

    md("""---
# Part B — QLoRA fine-tuning μέσω Ray Train

Η εκφώνηση ζητά ρητά «Ενορχήστρωση της εκπαίδευσης μέσω Ray». Το
`ray.train.torch.TorchTrainer` τρέχει το ίδιο training loop σε 1 worker
(Colab) ή N workers (cluster) χωρίς αλλαγή κώδικα — αυτό είναι το ζητούμενο
της κλιμακωσιμότητας.
"""),

    md("## Cell 7 — Ray init"),
    code('''import ray

if ray.is_initialized():
    ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False)

res = ray.cluster_resources()
print(f"Ray {ray.__version__}  |  CPU: {res.get('CPU', 0):.0f}  |  GPU: {res.get('GPU', 0):.0f}")
'''),

    md("""## Cell 8 — Training

Το save γίνεται στο **ίδιο cell** αμέσως μετά — μάθημα από προηγούμενο run
όπου ένα crash μετά το training έχασε τα βάρη.
"""),
    code('''import time, torch
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

def train_loop(config):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_dict({"text": config["texts"]}),
        args=SFTConfig(
            output_dir=config["ckpt_dir"],
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["grad_accum"],
            learning_rate=config["learning_rate"],
            max_seq_length=config["max_seq_length"],
            warmup_ratio=0.03,
            logging_steps=10,
            save_steps=config["save_steps"],
            save_total_limit=2,
            bf16=True,
            report_to=[],
            seed=config["seed"],
            dataset_text_field="text",
        ),
    )
    result = trainer.train()

    model.save_pretrained(config["out_adapters"])
    tokenizer.save_pretrained(config["out_adapters"])
    print(f"✅ Adapters saved → {config['out_adapters']}")
    return {"train_loss": result.training_loss}


train_config = {
    "base_model": CFG["base_model"],
    "texts": texts,
    "ckpt_dir": str(CFG["out_adapters"] / "checkpoints"),
    "out_adapters": str(CFG["out_adapters"]),
    **{k: CFG[k] for k in ("lora_r", "lora_alpha", "lora_dropout", "target_modules",
                           "learning_rate", "num_epochs", "batch_size", "grad_accum",
                           "max_seq_length", "save_steps", "seed")},
}

started = time.perf_counter()
trainer = TorchTrainer(
    train_loop_per_worker=train_loop,
    train_loop_config=train_config,
    scaling_config=ScalingConfig(num_workers=CFG["num_workers"], use_gpu=True),
    run_config=RunConfig(storage_path="/content/ray_results", name="krikri_v4"),
)
result = trainer.fit()
wall = time.perf_counter() - started

print(f"\\n{'='*54}")
print(f"Ολοκληρώθηκε σε {wall/60:.1f} λεπτά  ({CFG['num_workers']} worker)")
print(f"Metrics: {result.metrics}")
print(f"{'='*54}")

# Μετρικές κλιμάκωσης για το κεφάλαιο αποτελεσμάτων
import json as _json
scaling_log = CFG["out_adapters"].parent / "ray_scaling.json"
hist = _json.loads(scaling_log.read_text()) if scaling_log.exists() else []
hist.append({
    "num_workers": CFG["num_workers"],
    "wall_clock_seconds": round(wall, 2),
    "effective_batch_size": CFG["batch_size"] * CFG["grad_accum"] * CFG["num_workers"],
    "n_samples": len(texts),
    "epochs": CFG["num_epochs"],
})
scaling_log.write_text(_json.dumps(hist, indent=2))
print(f"Scaling log → {scaling_log}")
'''),

    md("""---
## Cell 9 — Αξιολόγηση

Τα ίδια probes με το `src/jarvis/evaluation/three_stage.py`, ώστε τα
αποτελέσματα να συγκρίνονται με τα προηγούμενα στάδια.
"""),
    code('''import torch, json, time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROBES = [
    "Τι κάνεις; Όλα καλά;",
    "Θα έρθεις τελικά το Σάββατο;",
    "Μπορείς να μου στείλεις την αναφορά μέχρι αύριο;",
    "Πώς σου φάνηκε η συνάντηση σήμερα;",
    "Έχεις κανένα νέο για το project;",
    "Τι λες να φάμε το βράδυ;",
    "Can you join the call at 3pm tomorrow?",
    "Ευχαριστώ πολύ για τη βοήθεια χθες!",
    "Πότε μπορούμε να τα πούμε από κοντά;",
    "Από πού είσαι;",
]

tok = AutoTokenizer.from_pretrained(str(CFG["out_adapters"]))
base = AutoModelForCausalLM.from_pretrained(
    CFG["base_model"],
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
    device_map={"": 0}, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, str(CFG["out_adapters"]))
model.eval()

def generate(prompt, max_new_tokens=100):
    text = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n")
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=0.7, top_p=0.9, do_sample=True,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

outputs, latencies = [], []
for p in PROBES:
    t0 = time.perf_counter()
    r = generate(p)
    latencies.append(time.perf_counter() - t0)
    outputs.append(r)
    print(f"\\n❓ {p}\\n💬 {r}")

CFG["eval_out"].write_text(json.dumps({
    "stage": "4_krikri_v4_clean",
    "probes": PROBES,
    "outputs": outputs,
    "latencies_s": latencies,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\\n→ {CFG['eval_out']}")
print("\\nΚατέβασε αυτό το αρχείο· τρέχει μέσα από το metrics.py για τους αριθμούς της διπλωματικής.")
'''),

    md("""---
# Part D — Merge + GGUF export

Μετά από αυτό δεν χρειάζεσαι ξανά Colab. Το `.gguf` τρέχει τοπικά στο
Mac Mini μέσω Ollama — καμία εξάρτηση από δίκτυο στην παρουσίαση.
"""),

    md("## Cell 10 — Merge adapters στο base"),
    code('''import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ελευθέρωσε τη μνήμη από το evaluation
for v in ("model", "base"):
    if v in dir():
        del globals()[v]
gc.collect(); torch.cuda.empty_cache()

# Το merge απαιτεί fp16 base (όχι 4-bit) — χρειάζεται RAM, όχι VRAM
base = AutoModelForCausalLM.from_pretrained(
    CFG["base_model"], torch_dtype=torch.float16,
    device_map="cpu", low_cpu_mem_usage=True)
merged = PeftModel.from_pretrained(base, str(CFG["out_adapters"])).merge_and_unload()

CFG["out_merged"].mkdir(parents=True, exist_ok=True)
merged.save_pretrained(CFG["out_merged"], safe_serialization=True)
AutoTokenizer.from_pretrained(str(CFG["out_adapters"])).save_pretrained(CFG["out_merged"])

del base, merged; gc.collect()
print(f"✅ Merged → {CFG['out_merged']}")
!du -sh {CFG["out_merged"]}
'''),

    md("## Cell 11 — Convert σε GGUF + quantize"),
    code('''!git clone --depth 1 https://github.com/ggerganov/llama.cpp /content/llama.cpp 2>/dev/null || echo "υπάρχει ήδη"
%pip install -q -r /content/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

F16 = "/content/krikri-jarvis-f16.gguf"
!python /content/llama.cpp/convert_hf_to_gguf.py {CFG["out_merged"]} --outfile {F16} --outtype f16
'''),

    code('''import os
QUANT = CFG["quant"]
OUT = f"/content/krikri-jarvis-{QUANT}.gguf"

# llama-quantize: δοκίμασε prebuilt, αλλιώς build
!cd /content/llama.cpp && (cmake -B build -DGGML_CUDA=OFF > /dev/null 2>&1 && cmake --build build --config Release -j --target llama-quantize > /dev/null 2>&1) || echo "build issue"

BIN = "/content/llama.cpp/build/bin/llama-quantize"
if not os.path.exists(BIN):
    BIN = "/content/llama.cpp/llama-quantize"

!{BIN} {F16} {OUT} {QUANT}
!ls -lh {OUT}
'''),

    md("## Cell 12 — Αποθήκευση στο Drive"),
    code('''import shutil, os

dest = CFG["out_gguf"] / os.path.basename(OUT)
shutil.copy(OUT, dest)
size_gb = os.path.getsize(dest) / 1e9

print(f"✅ {dest}")
print(f"   {size_gb:.2f} GB")
print(f"""
{'='*58}
ΤΕΛΟΣ. Στο Mac Mini:

  mkdir -p ~/jarvis/models
  # κατέβασε το {os.path.basename(OUT)} από το Drive στο ~/jarvis/models/

Μετά πες στον Claude "το gguf είναι στο ~/jarvis/models" και
στήνει το Ollama + docker-compose.
{'='*58}
""")
'''),

    md("""---
# Επόμενα

1. Κατέβασε το `.gguf` από `MyDrive/jarvis_models/gguf/`
2. Κατέβασε το `eval_v4.json` — δίνει τους αριθμούς για το κεφάλαιο αποτελεσμάτων
3. Κατέβασε το `ray_scaling.json` — πίνακας κλιμάκωσης
4. Βάλε το `.gguf` στο `~/jarvis/models/`

**Δεν χρειάζεσαι ξανά Colab μετά από αυτό.**
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
