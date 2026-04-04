from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from coord_harness.config.models import BatchConfig, ModelSpec
from coord_harness.core.enums import BaselineStrategy, BenchmarkFamily, RunStage, TopologyPreset


@dataclass(frozen=True)
class TrialConfig:
    experiment_id: str
    framework_id: str
    stage: RunStage
    benchmark_family: BenchmarkFamily
    topology_preset: TopologyPreset
    message_token_budget: int
    model_spec: ModelSpec
    baseline: BaselineStrategy
    seed: int
    agent_count: int
    total_billed_token_budget: int
    output_root: Path
    config_digest: str
    config_path: Path
    attack: dict

    @property
    def trial_id(self) -> str:
        return (
            f"{self.experiment_id}"
            f"__{self.benchmark_family.value}"
            f"__{self.topology_preset.value}"
            f"__msg{self.message_token_budget}"
            f"__{self.model_spec.alias}"
            f"__{self.baseline.value}"
            f"__seed{self.seed}"
        )


def load_config(config_path: str | Path) -> tuple[BatchConfig, str]:
    path = Path(config_path)
    raw_text = path.read_text(encoding="utf-8")
    payload: dict[str, Any] = yaml.safe_load(raw_text)
    config = BatchConfig.model_validate(payload)
    config_digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return config, config_digest


def expand_trials(config: BatchConfig, config_digest: str, config_path: str | Path) -> list[TrialConfig]:
    trials: list[TrialConfig] = []
    path = Path(config_path)
    for family in config.sweep.benchmark_families:
        for topology in config.sweep.topology_presets:
            for message_budget in config.sweep.message_token_budgets:
                for alias in config.selected_model_aliases():
                    model_spec = config.model_spec_for(alias)
                    for baseline in config.sweep.baselines:
                        for seed in config.sweep.seeds:
                            trials.append(
                                TrialConfig(
                                    experiment_id=config.run.experiment_id,
                                    framework_id=config.run.framework_id,
                                    stage=config.run.stage,
                                    benchmark_family=family,
                                    topology_preset=topology,
                                    message_token_budget=message_budget,
                                    model_spec=model_spec,
                                    baseline=baseline,
                                    seed=seed,
                                    agent_count=config.run.agent_count,
                                    total_billed_token_budget=config.run.total_billed_token_budget,
                                    output_root=config.run.output_root,
                                    config_digest=config_digest,
                                    config_path=path,
                                    attack=config.run.attack.model_dump(mode="json"),
                                )
                            )
    return trials


def config_to_canonical_json(config: BatchConfig) -> str:
    return json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2)
