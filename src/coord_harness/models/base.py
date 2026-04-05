from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from coord_harness.core.types import BenchmarkTask, GenerationResult


class ModelClient(ABC):
    @abstractmethod
    def generate_decision(
        self,
        *,
        task: BenchmarkTask,
        agent_id: str,
        visible_messages: list[str],
        enable_reasoning: bool,
        seed: int,
    ) -> GenerationResult:
        raise NotImplementedError

    def fetch_model_catalog_snapshot(self) -> dict | None:
        return None

    def describe_runtime_metadata(self) -> dict[str, Any]:
        return {}
