from __future__ import annotations

from coord_harness.benchmarks import build_benchmark_adapter
from coord_harness.config.models import BenchmarkAdapterConfig
from coord_harness.core.enums import BenchmarkFamily, RunStage


def test_repo_local_benchmark_pack_loads_manifest_and_tasks() -> None:
    adapter = build_benchmark_adapter(
        BenchmarkAdapterConfig(family=BenchmarkFamily.CRAFT_MINI, stage=RunStage.CLEAN, task_limit=3)
    )
    dataset = adapter.load_dataset()
    assert dataset.manifest.dataset_revision == "local-mini-v3"
    assert len(dataset.tasks) == 3
    assert dataset.manifest.task_schema_version == "mcq-partial-info-v1"
    assert "agent_0" in dataset.tasks[0].metadata["agent_observations"]
    assert dataset.dataset_digest
