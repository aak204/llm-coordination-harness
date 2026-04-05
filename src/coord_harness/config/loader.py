from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from coord_harness.config.models import AttackScenarioConfig, BatchConfig, ModelSpec
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
    enable_reasoning: bool = False
    attack_scenario_name: str | None = None

    @property
    def trial_id(self) -> str:
        attack_suffix = f"__atk{self.attack_scenario_name}" if self.attack_scenario_name else ""
        reasoning_suffix = "__reasoning-on" if self.enable_reasoning else ""
        return (
            f"{self.experiment_id}"
            f"__{self.benchmark_family.value}"
            f"__{self.topology_preset.value}"
            f"__msg{self.message_token_budget}"
            f"__{self.model_spec.alias}"
            f"__{self.baseline.value}"
            f"{reasoning_suffix}"
            f"{attack_suffix}"
            f"__seed{self.seed}"
        )


def load_config(config_path: str | Path) -> tuple[BatchConfig, str]:
    path = Path(config_path)
    raw_text = path.read_text(encoding="utf-8")
    payload: dict[str, Any] = yaml.safe_load(raw_text)
    config = BatchConfig.model_validate(payload)
    config_digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return config, config_digest


def _merged_attack_payload(base_attack: dict[str, Any], scenario: AttackScenarioConfig | None) -> dict[str, Any]:
    payload = deepcopy(base_attack)
    if scenario is None:
        return payload
    scenario_payload = scenario.model_dump(mode="json", exclude_unset=True)
    for key in ("name", "topology_presets", "baselines"):
        scenario_payload.pop(key, None)
    payload.update(scenario_payload)
    return payload


def _scenario_applies(
    *,
    scenario: AttackScenarioConfig,
    topology: TopologyPreset,
    baseline: BaselineStrategy,
) -> bool:
    if scenario.topology_presets and topology not in set(scenario.topology_presets):
        return False
    if scenario.baselines and baseline not in set(scenario.baselines):
        return False
    return True


def expand_trials(config: BatchConfig, config_digest: str, config_path: str | Path) -> list[TrialConfig]:
    trials: list[TrialConfig] = []
    path = Path(config_path)
    scenario_entries = config.sweep.attack_scenarios if config.run.stage is RunStage.STRESS and config.sweep.attack_scenarios else [None]
    base_attack = config.run.attack.model_dump(mode="json")
    reasoning_values = config.sweep.enable_reasoning_values if config.sweep.enable_reasoning_values else [config.run.enable_reasoning]
    for family in config.sweep.benchmark_families:
        for topology in config.sweep.topology_presets:
            for message_budget in config.sweep.message_token_budgets:
                for alias in config.selected_model_aliases():
                    model_spec = config.model_spec_for(alias)
                    for enable_reasoning in reasoning_values:
                        for baseline in config.sweep.baselines:
                            for seed in config.sweep.seeds:
                                for scenario in scenario_entries:
                                    if scenario is not None and not _scenario_applies(
                                        scenario=scenario,
                                        topology=topology,
                                        baseline=baseline,
                                    ):
                                        continue
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
                                            attack=_merged_attack_payload(base_attack, scenario),
                                            enable_reasoning=enable_reasoning,
                                            attack_scenario_name=scenario.name if scenario is not None else None,
                                        )
                                    )
    return trials


def config_to_canonical_json(config: BatchConfig) -> str:
    return json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2)
