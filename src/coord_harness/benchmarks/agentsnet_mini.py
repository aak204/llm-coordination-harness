from __future__ import annotations

from pathlib import Path

from coord_harness.benchmarks.base import BenchmarkAdapter, BenchmarkDataset, load_jsonl_dataset


class AgentsNetMiniAdapter(BenchmarkAdapter):
    def load_dataset(self) -> BenchmarkDataset:
        dataset_dir = self.config.dataset_dir or Path("data/benchmarks/clean/agentsnet_mini")
        return load_jsonl_dataset(
            family=self.config.family,
            dataset_dir=dataset_dir,
            task_limit=self.config.task_limit,
        )
