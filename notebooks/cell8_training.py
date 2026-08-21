import time, torch
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

    # TRL renamed SFTConfig's sequence-length argument from max_seq_length to
    # max_length, and made dataset_text_field optional, across recent releases.
    # Colab resolves the version at install time, so probe the signature rather
    # than pinning a version that may vanish or conflict with transformers.
    import inspect
    sft_params = inspect.signature(SFTConfig.__init__).parameters

    sft_kwargs = dict(
        output_dir=config["ckpt_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["learning_rate"],
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=config["save_steps"],
        save_total_limit=2,
        bf16=True,
        report_to=[],
        seed=config["seed"],
    )
    if "max_length" in sft_params:
        sft_kwargs["max_length"] = config["max_seq_length"]
    elif "max_seq_length" in sft_params:
        sft_kwargs["max_seq_length"] = config["max_seq_length"]
    if "dataset_text_field" in sft_params:
        sft_kwargs["dataset_text_field"] = "text"

    print(f"TRL SFTConfig: χρησιμοποιώ "
          f"{'max_length' if 'max_length' in sft_params else 'max_seq_length'}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_dict({"text": config["texts"]}),
        args=SFTConfig(**sft_kwargs),
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

print(f"\n{'='*54}")
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
