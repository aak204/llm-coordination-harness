from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from coord_harness.config.models import BenchmarkAdapterConfig
from coord_harness.core.enums import BenchmarkFamily, RunStage
from coord_harness.core.types import BenchmarkTask


@dataclass(frozen=True)
class BenchmarkManifest:
    family: BenchmarkFamily
    stage: RunStage
    dataset_revision: str
    task_schema_version: str
    source_kind: str
    description: str


@dataclass(frozen=True)
class BenchmarkDataset:
    manifest: BenchmarkManifest
    tasks: list[BenchmarkTask]
    dataset_path: Path
    manifest_path: Path
    dataset_digest: str


class BenchmarkAdapter(ABC):
    def __init__(self, config: BenchmarkAdapterConfig):
        self.config = config

    @property
    def family(self) -> str:
        return self.config.family.value

    @abstractmethod
    def load_dataset(self) -> BenchmarkDataset:
        raise NotImplementedError

    def score_answer(self, task: BenchmarkTask, answer: str) -> float:
        return 1.0 if answer.strip() == task.gold_answer.strip() else 0.0


def load_jsonl_dataset(*, family: BenchmarkFamily, dataset_dir: Path, task_limit: int | None) -> BenchmarkDataset:
    manifest_path = dataset_dir / "manifest.yaml"
    task_path = dataset_dir / "tasks.jsonl"
    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = BenchmarkManifest(
        family=family,
        stage=RunStage(manifest_payload["stage"]),
        dataset_revision=manifest_payload["dataset_revision"],
        task_schema_version=manifest_payload["task_schema_version"],
        source_kind=manifest_payload["source_kind"],
        description=manifest_payload["description"],
    )
    tasks: list[BenchmarkTask] = []
    for raw_line in task_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload: dict[str, Any] = json.loads(raw_line)
        tasks.append(
            BenchmarkTask(
                family=family,
                task_id=payload["task_id"],
                prompt=payload["prompt"],
                answer_choices=payload["answer_choices"],
                gold_answer=payload["gold_answer"],
                metadata=payload.get("metadata", {}),
            )
        )
    if task_limit is not None:
        tasks = tasks[:task_limit]
    dataset_digest = hashlib.sha256(task_path.read_bytes()).hexdigest()
    return BenchmarkDataset(
        manifest=manifest,
        tasks=tasks,
        dataset_path=task_path,
        manifest_path=manifest_path,
        dataset_digest=dataset_digest,
    )
