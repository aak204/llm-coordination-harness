from __future__ import annotations

from coord_harness.benchmarks.agentsnet_mini import AgentsNetMiniAdapter
from coord_harness.benchmarks.base import BenchmarkAdapter
from coord_harness.benchmarks.craft_mini import CraftMiniAdapter
from coord_harness.config.models import BenchmarkAdapterConfig
from coord_harness.core.enums import BenchmarkFamily


def build_benchmark_adapter(config: BenchmarkAdapterConfig) -> BenchmarkAdapter:
    if config.family is BenchmarkFamily.CRAFT_MINI:
        return CraftMiniAdapter(config)
    if config.family is BenchmarkFamily.AGENTSNET_MINI:
        return AgentsNetMiniAdapter(config)
    raise ValueError(f"Unsupported benchmark family: {config.family.value}")
