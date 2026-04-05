from __future__ import annotations

from pathlib import Path

import pytest

from coord_harness.config.models import BatchConfig
from coord_harness.config.loader import expand_trials, load_config


def test_load_and_expand_smoke_config() -> None:
    config, digest = load_config(Path("archive/configs/p0a_clean_smoke.yaml"))
    trials = expand_trials(config, digest, Path("archive/configs/p0a_clean_smoke.yaml"))
    assert digest
    assert len(trials) == 6
    assert trials[0].benchmark_family.value in {"craft_mini", "agentsnet_mini"}


def test_live_panel_config_requires_and_expands_explicit_provider_pins() -> None:
    config, digest = load_config(Path("archive/configs/p0a_clean_live_panel.yaml"))
    trials = expand_trials(config, digest, Path("archive/configs/p0a_clean_live_panel.yaml"))
    assert len(trials) == 216
    assert all(
        trial.model_spec.provider.value != "openrouter" or trial.model_spec.routing.only
        for trial in trials
    )


def test_tier_selected_config_expands_only_selected_panel_tier() -> None:
    config, digest = load_config(Path("archive/configs/p0a_clean_live_sweep.yaml"))
    trials = expand_trials(config, digest, Path("archive/configs/p0a_clean_live_sweep.yaml"))
    assert len(trials) == 54
    assert {trial.model_spec.alias for trial in trials} == {"qwen35_plus_0215"}


def test_research_strict_rejects_openrouter_auto() -> None:
    payload = {
        "run": {
            "experiment_id": "x",
            "framework_id": "coord_harness_v1",
            "stage": "clean",
            "mode": "research_strict",
            "agent_count": 5,
            "total_billed_token_budget": 1000,
        },
        "benchmarks": [{"family": "craft_mini", "stage": "clean"}],
        "model_panel": [
            {
                "alias": "bad",
                "provider": "openrouter",
                "model_id": "openrouter/auto",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_completion_tokens": 32,
                "allow_auto_routing": False,
                "routing": {"only": ["alibaba"], "allow_fallbacks": False},
            }
        ],
        "sweep": {
            "benchmark_families": ["craft_mini"],
            "topology_presets": ["star"],
            "message_token_budgets": [32],
            "model_aliases": ["bad"],
            "baselines": ["sa_star"],
            "seeds": [7],
        },
    }
    with pytest.raises(ValueError):
        BatchConfig.model_validate(payload)


def test_dev_mode_allows_openrouter_auto() -> None:
    config, _digest = load_config(Path("archive/configs/openrouter_dev_smoke.yaml"))
    assert config.run.mode.value == "dev_convenience"
    assert config.model_panel[0].model_id == "openrouter/auto"


def test_attack_scenarios_expand_only_matching_topologies() -> None:
    payload = {
        "run": {
            "experiment_id": "stress-zoo",
            "framework_id": "coord_harness_v1",
            "stage": "stress",
            "mode": "research_strict",
            "agent_count": 5,
            "total_billed_token_budget": 1000,
            "attack": {
                "enabled": True,
                "injection_depth": "leaf",
                "payload_type": "hard_hallucination",
                "only_baselines": ["ma_ft"],
            },
        },
        "benchmarks": [{"family": "craft_mini", "stage": "clean"}],
        "model_panel": [
            {
                "alias": "mock",
                "provider": "mock",
                "model_id": "mock/unit",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_completion_tokens": 32,
                "allow_auto_routing": False,
            }
        ],
        "sweep": {
            "benchmark_families": ["craft_mini"],
            "topology_presets": ["star", "balanced_tree", "linear_chain", "complete_graph"],
            "message_token_budgets": [96],
            "model_aliases": ["mock"],
            "baselines": ["ma_ft"],
            "seeds": [7],
            "attack_scenarios": [
                {"name": "leaf", "injection_depth": "leaf"},
                {
                    "name": "middle_manager",
                    "injection_depth": "middle_manager",
                    "topology_presets": ["balanced_tree", "linear_chain"],
                },
            ],
        },
    }
    config = BatchConfig.model_validate(payload)
    trials = expand_trials(config, "digest", Path("stress-zoo.yaml"))
    assert len(trials) == 6
    scenario_pairs = {(trial.topology_preset.value, trial.attack_scenario_name) for trial in trials}
    assert ("star", "middle_manager") not in scenario_pairs
    assert ("complete_graph", "middle_manager") not in scenario_pairs
    assert ("linear_chain", "middle_manager") in scenario_pairs
    assert ("balanced_tree", "middle_manager") in scenario_pairs
