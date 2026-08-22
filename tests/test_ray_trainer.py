"""Tests for the Ray distributed-training wrapper.

These exercise the parts that must be correct *before* a GPU is involved:
config arithmetic, data sharding, and the scaling analysis that feeds the
thesis results chapter. Ray/torch/transformers are never imported here —
the module is written so the pure logic is testable without them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.training.ray_trainer import (
    LoRAConfig,
    RayTrainingConfig,
    scaling_efficiency,
    scaling_markdown,
    shard_for_worker,
)


# ── Configuration ───────────────────────────────────────────────


def test_effective_batch_size_scales_with_workers():
    """Global batch = per-device × accumulation × workers."""
    cfg = RayTrainingConfig(
        corpus_path="x.json",
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        num_workers=2,
    )
    assert cfg.effective_batch_size == 32


def test_single_worker_matches_notebook_baseline():
    """1 worker must reproduce the Phase-3 single-process effective batch."""
    cfg = RayTrainingConfig(corpus_path="x.json", num_workers=1)
    assert cfg.effective_batch_size == 16


def test_config_serialises_for_ray():
    """Ray requires a plain serialisable dict for train_loop_config."""
    cfg = RayTrainingConfig(corpus_path="x.json")
    d = cfg.to_dict()
    json.dumps(d)  # must not raise
    assert d["lora"]["r"] == 64
    assert d["effective_batch_size"] == 16


def test_defaults_match_the_shipped_adapter():
    """The checked-in configuration must reproduce the measured model.

    The defaults here said r=16 and alpha=32 while the adapter the thesis
    describes was trained with 64 and 128 — values set in the notebook and
    never reflected back into the code. Anyone re-running this file would
    have produced a different model from the one that was evaluated, and
    nothing would have signalled the difference.
    """
    adapter = Path("models/adapter/adapter_config.json")
    if not adapter.exists():
        pytest.skip("adapter not present in this checkout")

    shipped = json.loads(adapter.read_text(encoding="utf-8"))
    lora = LoRAConfig()
    assert lora.r == shipped["r"]
    assert lora.lora_alpha == shipped["lora_alpha"]
    assert set(lora.target_modules) == set(shipped["target_modules"])


def test_lora_peft_kwargs_shape():
    kwargs = LoRAConfig().to_peft_kwargs()
    assert kwargs["task_type"] == "CAUSAL_LM"
    assert "q_proj" in kwargs["target_modules"]
    assert isinstance(kwargs["target_modules"], list)  # peft rejects tuples


# ── Sharding ────────────────────────────────────────────────────


def test_shards_partition_corpus_exactly():
    """Every record lands on exactly one worker — no loss, no duplication."""
    records = [{"i": i} for i in range(100)]
    shards = [shard_for_worker(records, r, 4) for r in range(4)]

    assert sum(len(s) for s in shards) == 100
    seen = [rec["i"] for shard in shards for rec in shard]
    assert sorted(seen) == list(range(100))


def test_shards_are_balanced_within_one():
    """Uneven corpora must not starve or overload a worker."""
    records = [{"i": i} for i in range(101)]
    sizes = [len(shard_for_worker(records, r, 4)) for r in range(4)]
    assert max(sizes) - min(sizes) <= 1


def test_shard_is_strided_not_contiguous():
    """Strided assignment spreads chronological ordering across workers."""
    records = [{"i": i} for i in range(6)]
    assert shard_for_worker(records, 0, 2) == [{"i": 0}, {"i": 2}, {"i": 4}]
    assert shard_for_worker(records, 1, 2) == [{"i": 1}, {"i": 3}, {"i": 5}]


def test_single_worker_gets_everything():
    records = [{"i": i} for i in range(10)]
    assert shard_for_worker(records, 0, 1) == records


def test_more_workers_than_records_is_safe():
    """Extra workers get empty shards rather than crashing."""
    records = [{"i": 0}, {"i": 1}]
    assert shard_for_worker(records, 3, 4) == []


@pytest.mark.parametrize("rank,workers", [(-1, 2), (2, 2), (0, 0)])
def test_invalid_sharding_arguments_rejected(rank, workers):
    with pytest.raises(ValueError):
        shard_for_worker([{"i": 0}], rank, workers)


# ── Scaling analysis ────────────────────────────────────────────


def test_perfect_linear_scaling_has_unit_efficiency():
    report = scaling_efficiency({1: 100.0, 2: 50.0, 4: 25.0})
    assert [r["efficiency"] for r in report["rows"]] == [1.0, 1.0, 1.0]
    assert [r["speedup"] for r in report["rows"]] == [1.0, 2.0, 4.0]


def test_realistic_scaling_reports_overhead():
    """Sub-linear scaling must surface as measurable overhead, not be hidden."""
    report = scaling_efficiency({1: 1840.0, 2: 980.0, 4: 545.0})
    four = report["rows"][-1]
    assert four["speedup"] == pytest.approx(3.376, abs=1e-3)
    assert four["efficiency"] == pytest.approx(0.844, abs=1e-3)
    assert four["overhead_pct"] > 0


def test_baseline_is_smallest_worker_count():
    report = scaling_efficiency({2: 80.0, 8: 25.0})
    assert report["baseline_workers"] == 2
    assert report["rows"][0]["speedup"] == 1.0


def test_superlinear_scaling_does_not_report_negative_overhead():
    """Cache effects can beat linear; overhead must clamp at zero."""
    report = scaling_efficiency({1: 100.0, 2: 40.0})
    assert report["rows"][1]["efficiency"] > 1.0
    assert report["rows"][1]["overhead_pct"] == 0.0


@pytest.mark.parametrize("bad", [{}, {1: 0.0}, {1: -5.0}])
def test_invalid_timings_rejected(bad):
    with pytest.raises(ValueError):
        scaling_efficiency(bad)


def test_markdown_table_is_thesis_ready():
    md = scaling_markdown(scaling_efficiency({1: 100.0, 2: 55.0}))
    lines = md.splitlines()
    assert lines[0].startswith("| Workers")
    assert len(lines) == 4  # header + separator + 2 data rows
    assert "1.82×" in md
