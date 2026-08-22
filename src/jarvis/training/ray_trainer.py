"""Distributed QLoRA fine-tuning orchestrated with Ray Train (thesis §"Κατανεμημένη Βελτιστοποίηση").

The thesis specification requires training orchestration through Ray for
scalability and efficient use of compute. This module wraps the existing
single-process QLoRA loop in a ``ray.train.torch.TorchTrainer`` so the same
code path runs on 1 worker (Colab T4), N workers on one node, or a multi-node
cluster, with no changes to the training function itself.

Design notes
------------
* **Lazy imports.** ``ray``/``torch``/``transformers`` are imported *inside*
  the functions that need them. The configuration dataclasses and the scaling
  analysis are therefore unit-testable on a machine with none of the heavy
  dependencies installed — which is how the test-suite runs in CI.
* **Data parallelism, not model parallelism.** A 4-bit quantised 8B model fits
  in a single 16GB GPU, so the correct scaling axis is replicating the model
  and sharding the *data*. Each worker sees ``1/N`` of the shards; gradients
  are all-reduced by ``TorchTrainer``'s DDP backend.
* **Reproducibility.** Sharding derives from the same fixed seed used by
  :func:`jarvis.training.dataset.train_val_split`, so a 1-worker run and an
  N-worker run traverse the same examples.

Usage (Colab, 1 GPU)::

    from jarvis.training.ray_trainer import RayTrainingConfig, run_distributed_training
    cfg = RayTrainingConfig(corpus_path="data/train.json", num_workers=1)
    result = run_distributed_training(cfg)

Scaling experiment for the results chapter::

    from jarvis.training.ray_trainer import scaling_efficiency
    report = scaling_efficiency({1: 1840.0, 2: 980.0, 4: 545.0})
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "LoRAConfig",
    "RayTrainingConfig",
    "build_scaling_config",
    "train_loop_per_worker",
    "run_distributed_training",
    "shard_for_worker",
    "scaling_efficiency",
]


# ── Configuration ───────────────────────────────────────────────


@dataclass
class LoRAConfig:
    """QLoRA adapter hyper-parameters.

    Defaults match the Phase-3 notebook so distributed runs are directly
    comparable with the existing single-process baseline.
    """

    # These are the values the shipped adapter was actually trained with,
    # read back from models/adapter/adapter_config.json.
    #
    # They were 16 and 32 here while the trained model used 64 and 128 —
    # library defaults that were overridden in the notebook and never
    # reflected back into the code. Anyone re-running this file would have
    # produced a different model from the one the thesis describes and
    # measures, and nothing would have signalled the difference.
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    def to_peft_kwargs(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": list(self.target_modules),
        }


@dataclass
class RayTrainingConfig:
    """Everything needed to launch a distributed fine-tuning run."""

    corpus_path: str
    base_model: str = "ilsp/Llama-Krikri-8B-Instruct"
    output_dir: str = "outputs/krikri-jarvis"

    # Ray scaling
    num_workers: int = 1
    use_gpu: bool = True
    cpus_per_worker: int = 2

    # Optimisation
    epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 1024
    seed: int = 42

    # Quantisation
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @property
    def effective_batch_size(self) -> int:
        """Global batch size seen by the optimiser across all workers.

        This is the number that must be held constant when comparing a
        1-worker run against an N-worker run — otherwise the comparison
        measures a different optimisation problem, not parallel speed-up.
        """
        return (
            self.per_device_batch_size
            * self.gradient_accumulation_steps
            * self.num_workers
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["effective_batch_size"] = self.effective_batch_size
        return d


# ── Sharding ────────────────────────────────────────────────────


def shard_for_worker(
    records: list[dict], worker_rank: int, num_workers: int
) -> list[dict]:
    """Return the contiguous-strided shard belonging to ``worker_rank``.

    Uses strided (round-robin) assignment rather than contiguous blocks so
    that any ordering left in the corpus — e.g. chronological Viber threads —
    is spread evenly across workers instead of giving one worker all the
    earliest conversations.

    >>> shard_for_worker([{"i": i} for i in range(6)], 0, 2)
    [{'i': 0}, {'i': 2}, {'i': 4}]
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if not 0 <= worker_rank < num_workers:
        raise ValueError(
            f"worker_rank {worker_rank} out of range for {num_workers} workers"
        )
    return records[worker_rank::num_workers]


def build_scaling_config(cfg: RayTrainingConfig):
    """Construct a ``ray.train.ScalingConfig`` from our config object."""
    from ray.train import ScalingConfig

    return ScalingConfig(
        num_workers=cfg.num_workers,
        use_gpu=cfg.use_gpu,
        resources_per_worker={"CPU": cfg.cpus_per_worker, "GPU": 1 if cfg.use_gpu else 0},
    )


# ── The per-worker training function ────────────────────────────


def train_loop_per_worker(train_config: dict) -> None:
    """Body executed independently by every Ray worker.

    Ray passes a plain dict (it must be serialisable), so the first step is
    rehydrating our dataclass. ``prepare_model`` wraps the model in DDP and
    moves it to this worker's assigned GPU.
    """
    import torch
    import ray.train
    from ray.train import Checkpoint
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    from jarvis.training.dataset import load_pairs, format_records, train_val_split

    cfg = RayTrainingConfig(**{
        k: (LoRAConfig(**v) if k == "lora" and isinstance(v, dict) else v)
        for k, v in train_config.items()
        if k in RayTrainingConfig.__dataclass_fields__
    })

    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    # ── Data: every worker loads the corpus, then keeps only its shard.
    records = load_pairs(cfg.corpus_path)
    train_records, _ = train_val_split(records, seed=cfg.seed)
    my_shard = shard_for_worker(train_records, rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = format_records(my_shard)
    dataset = Dataset.from_dict({"text": texts}).map(
        lambda b: tokenizer(
            b["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        ),
        batched=True,
        remove_columns=["text"],
    )

    # ── Model: 4-bit base + LoRA adapters (only adapters are trained).
    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        device_map={"": ray.train.torch.get_device()},
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**cfg.lora.to_peft_kwargs()))

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=cfg.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    train_result = trainer.train()

    # Rank 0 owns checkpoint persistence; other ranks hold identical weights.
    if rank == 0:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out / "adapters")
        tokenizer.save_pretrained(out / "adapters")
        ray.train.report(
            {
                "train_loss": train_result.training_loss,
                "world_size": world_size,
                "shard_size": len(my_shard),
            },
            checkpoint=Checkpoint.from_directory(str(out / "adapters")),
        )
    else:
        ray.train.report(
            {
                "train_loss": train_result.training_loss,
                "world_size": world_size,
                "shard_size": len(my_shard),
            }
        )


# ── Driver ──────────────────────────────────────────────────────


def run_distributed_training(
    cfg: RayTrainingConfig, metrics_path: str | Path | None = None
) -> dict[str, Any]:
    """Launch the distributed run and return a metrics dict for the thesis.

    ``metrics_path`` (if given) receives a JSON record of the run: wall-clock
    time, worker count, effective batch size and final loss. Collecting these
    across ``num_workers ∈ {1, 2, 4}`` produces the scaling table for the
    results chapter.
    """
    import time
    import ray
    from ray.train.torch import TorchTrainer

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=cfg.to_dict(),
        scaling_config=build_scaling_config(cfg),
    )

    started = time.perf_counter()
    result = trainer.fit()
    wall_clock = time.perf_counter() - started

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_workers": cfg.num_workers,
        "use_gpu": cfg.use_gpu,
        "effective_batch_size": cfg.effective_batch_size,
        "epochs": cfg.epochs,
        "base_model": cfg.base_model,
        "wall_clock_seconds": round(wall_clock, 2),
        "final_metrics": dict(result.metrics or {}),
    }

    if metrics_path:
        p = Path(metrics_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        history = json.loads(p.read_text()) if p.exists() else []
        history.append(metrics)
        p.write_text(json.dumps(history, ensure_ascii=False, indent=2))

    return metrics


# ── Scaling analysis (pure, dependency-free, unit-tested) ───────


def scaling_efficiency(timings: dict[int, float]) -> dict[str, Any]:
    """Turn ``{num_workers: wall_clock_seconds}`` into a scaling report.

    Speed-up is measured against the smallest worker count present (the
    baseline), and efficiency is speed-up divided by the worker ratio — the
    standard parallel-efficiency definition. Values near 1.0 mean near-linear
    scaling; the gap from 1.0 is communication and straggler overhead, which
    is exactly what the thesis should report and discuss.

    >>> r = scaling_efficiency({1: 100.0, 2: 55.0})
    >>> round(r["rows"][1]["speedup"], 3)
    1.818
    """
    if not timings:
        raise ValueError("timings must not be empty")
    if any(t <= 0 for t in timings.values()):
        raise ValueError("wall-clock timings must be positive")

    baseline_workers = min(timings)
    baseline_time = timings[baseline_workers]

    rows = []
    for workers in sorted(timings):
        speedup = baseline_time / timings[workers]
        worker_ratio = workers / baseline_workers
        efficiency = speedup / worker_ratio
        rows.append({
            "num_workers": workers,
            "wall_clock_seconds": round(timings[workers], 2),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 3),
            "overhead_pct": round(max(0.0, (1 - efficiency)) * 100, 1),
        })

    return {
        "baseline_workers": baseline_workers,
        "baseline_seconds": round(baseline_time, 2),
        "rows": rows,
    }


def scaling_markdown(report: dict[str, Any]) -> str:
    """Render :func:`scaling_efficiency` output as a thesis-ready table."""
    lines = [
        "| Workers | Wall-clock (s) | Speed-up | Efficiency | Overhead |",
        "|--------:|---------------:|---------:|-----------:|---------:|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['num_workers']} | {r['wall_clock_seconds']:.1f} | "
            f"{r['speedup']:.2f}× | {r['efficiency']:.2f} | {r['overhead_pct']:.1f}% |"
        )
    return "\n".join(lines)
