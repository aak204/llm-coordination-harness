from __future__ import annotations

import json
from pathlib import Path

from coord_harness.analysis.attack_analysis import write_attack_analysis
from coord_harness.attacks import AttackInjector
from coord_harness.baselines.ma_ft import MAFTExecutor
from coord_harness.baselines.base import StrategyContext
from coord_harness.baselines.vote_local import VoteLocalExecutor
from coord_harness.config.loader import TrialConfig
from coord_harness.config.models import BenchmarkAdapterConfig, ModelSpec
from coord_harness.core.enums import (
    AttackInjectionDepth,
    BaselineStrategy,
    BenchmarkFamily,
    ModelProvider,
    PanelTier,
    RunStage,
    TopologyPreset,
)
from coord_harness.core.topology import build_topology
from coord_harness.core.types import AgentDecision, BenchmarkTask, GenerationResult
from coord_harness.logging.schema import (
    AttackAnalysisSection,
    BenchmarkSection,
    BudgetSection,
    DerivedSection,
    ModelSection,
    OutcomeSection,
    ProvenanceSection,
    RunSection,
    TopologySection,
    TrialSummary,
)
from coord_harness.models.mock import MockModelClient


def _context(*, topology_preset: TopologyPreset, attack: dict, total_billed_token_budget: int = 1000) -> StrategyContext:
    model_spec = ModelSpec(
        alias="mock-local",
        provider=ModelProvider.MOCK,
        model_id="mock/p0a-deterministic-v1",
        panel_tier=PanelTier.DEV,
    )
    trial = TrialConfig(
        experiment_id="unit-exp",
        framework_id="coord_harness_v1",
        stage=RunStage.STRESS,
        benchmark_family=BenchmarkFamily.CRAFT_MINI,
        topology_preset=topology_preset,
        message_token_budget=32,
        model_spec=model_spec,
        baseline=BaselineStrategy.MA_FT,
        seed=7,
        agent_count=9,
        total_billed_token_budget=total_billed_token_budget,
        output_root=Path("outputs"),
        config_digest="digest",
        config_path=Path("configs/unit.yaml"),
        attack=attack,
    )
    benchmark_config = BenchmarkAdapterConfig(family=BenchmarkFamily.CRAFT_MINI, stage=RunStage.CLEAN)
    return StrategyContext(
        trial=trial,
        topology=build_topology(topology_preset, 9, 7),
        benchmark_config=benchmark_config,
        model_client=MockModelClient(model_spec),
    )


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        family=BenchmarkFamily.CRAFT_MINI,
        task_id="t1",
        prompt="Choose one option.",
        answer_choices=["A", "B", "C", "D"],
        gold_answer="A",
        metadata={},
    )


def _decision() -> tuple[AgentDecision, GenerationResult]:
    decision = AgentDecision(agent_id="agent_8", answer="A", confidence=0.8, rationale="clean", correct=True)
    result = GenerationResult(
        answer="A",
        confidence=0.8,
        rationale="clean",
        raw_text="ANSWER: A\nCONFIDENCE: 0.80\nRATIONALE: clean",
        prompt_tokens=10,
        completion_tokens=5,
        route_metadata={},
    )
    return decision, result


def test_attack_injector_selects_deepest_leaf_and_prompt_override_payload() -> None:
    context = _context(
        topology_preset=TopologyPreset.BALANCED_TREE,
        attack={
            "enabled": True,
            "injection_depth": AttackInjectionDepth.LEAF.value,
            "payload_type": "prompt_override",
            "compromise_confidence": 0.99,
            "propagate_instruction": True,
            "only_baselines": [BaselineStrategy.MA_FT.value],
        },
    )
    injector = AttackInjector.from_context(context)
    target = injector.select_target()
    assert target.agent_id == "agent_8"
    assert target.injection_depth is AttackInjectionDepth.LEAF

    decision, result = _decision()
    compromised_decision, compromised_result, payload = injector.maybe_inject(
        task=_task(),
        agent_id="agent_8",
        decision=decision,
        result=result,
    )
    assert compromised_decision.answer == "B"
    assert compromised_result.route_metadata["attack_payload_type"] == "prompt_override"
    assert "SYSTEM OVERRIDE" in compromised_decision.rationale
    assert payload is not None
    assert payload["injection_depth"] == "leaf"


def test_attack_injector_selects_manager_and_rejects_star_without_intermediate_manager() -> None:
    manager_context = _context(
        topology_preset=TopologyPreset.BALANCED_TREE,
        attack={
            "enabled": True,
            "injection_depth": AttackInjectionDepth.MIDDLE_MANAGER.value,
            "payload_type": "hard_hallucination",
            "only_baselines": [BaselineStrategy.MA_FT.value],
        },
    )
    injector = AttackInjector.from_context(manager_context)
    target = injector.select_target()
    assert target.agent_id == "agent_3"
    assert target.child_count > 0

    star_context = _context(
        topology_preset=TopologyPreset.STAR,
        attack={
            "enabled": True,
            "injection_depth": AttackInjectionDepth.MIDDLE_MANAGER.value,
            "payload_type": "hard_hallucination",
            "only_baselines": [BaselineStrategy.MA_FT.value],
        },
    )
    star_injector = AttackInjector.from_context(star_context)
    try:
        star_injector.select_target()
    except ValueError as exc:
        assert "requires an intermediate manager" in str(exc)
    else:
        raise AssertionError("Expected manager injection on star topology to fail.")


def test_attack_injector_selects_middle_manager_in_linear_chain() -> None:
    context = _context(
        topology_preset=TopologyPreset.LINEAR_CHAIN,
        attack={
            "enabled": True,
            "injection_depth": AttackInjectionDepth.MIDDLE_MANAGER.value,
            "payload_type": "hard_hallucination",
            "only_baselines": [BaselineStrategy.MA_FT.value],
        },
    )
    injector = AttackInjector.from_context(context)
    target = injector.select_target()
    assert target.agent_id == "agent_7"
    assert target.path_to_root[0] == "agent_7"
    assert target.path_to_root[-1] == "agent_0"


def test_executors_support_linear_chain_and_complete_graph() -> None:
    task = _task()
    for topology_preset in (TopologyPreset.LINEAR_CHAIN, TopologyPreset.COMPLETE_GRAPH):
        context = _context(topology_preset=topology_preset, attack={"enabled": False}, total_billed_token_budget=5000)
        trace, _events = MAFTExecutor().run_task(task=task, context=context)
        assert trace.selected_answer in task.answer_choices
        assert len(trace.messages) == len(context.topology.agent_ids) - 1

        vote_trace, _vote_events = VoteLocalExecutor().run_task(task=task, context=context)
        assert vote_trace.selected_answer in task.answer_choices
        assert vote_trace.messages == []


def test_attack_analysis_report_enriches_stress_summary_with_f_delta(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    stress_dir = tmp_path / "stress"
    clean_dir.mkdir()
    stress_dir.mkdir()

    def make_summary(*, experiment_id: str, score_mean: float, f_value: float, infection: float | None) -> TrialSummary:
        return TrialSummary(
            run=RunSection(
                experiment_id=experiment_id,
                trial_id=f"{experiment_id}-trial",
                framework_id="coord_harness_v1",
                stage="stress" if infection is not None else "clean",
                mode="research_strict",
                baseline="ma_ft",
                attack_scenario="leaf" if infection is not None else None,
                attack_injection_depth="leaf" if infection is not None else None,
                seed=7,
                status="completed",
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:01Z",
                config_digest="abc",
                decision_prompt_template_version="decision-v2",
                message_serializer_version="message-v2",
            ),
            model=ModelSection(
                alias="qwen35_plus_0215",
                panel_tier="sweep",
                provider="mock",
                model_id="mock",
                requested_model_id="mock",
                allow_auto_routing=False,
                routing={},
                temperature=0.0,
                top_p=1.0,
                max_completion_tokens=64,
                strict_routing_required=True,
                runtime_metadata={},
                route_metadata_examples=[],
            ),
            benchmark=BenchmarkSection(
                family="craft_mini",
                stage="clean",
                split="unit",
                dataset_revision="unit",
                task_schema_version="mcq-partial-info-v1",
                dataset_digest="digest",
                dataset_path="dataset.jsonl",
                manifest_path="manifest.yaml",
                source_kind="repo_local_pack",
                stubbed=False,
                task_count=1,
                task_ids=["t1"],
                scoring="exact_match_binary",
            ),
            topology=TopologySection(
                preset="balanced_tree",
                agent_count=9,
                root_agent="agent_0",
                edge_count=16,
                max_in_degree=2,
                max_out_degree=3,
                edges=[("agent_0", "agent_1"), ("agent_1", "agent_0")],
            ),
            budget=BudgetSection(
                total_billed_token_budget_per_task=1000,
                message_token_budget=32,
                task_prompt_tokens=[10],
                task_completion_tokens=[8],
                task_billed_tokens=[18],
                task_inter_agent_tokens=[6],
                task_model_call_counts=[3],
                mean_billed_tokens=18.0,
                max_billed_tokens=18,
                mean_inter_agent_tokens=6.0,
                max_inter_agent_tokens=6,
            ),
            outcomes=OutcomeSection(
                task_count=1,
                accuracy=score_mean,
                score_mean=score_mean,
                score_std=0.0,
                score_sum=score_mean,
                attacked_tasks=1 if infection is not None else 0,
                attack_success_rate=0.0 if infection is not None else None,
                infection_spread_rate=infection,
                quarantine_strength=0.8 if infection is not None else None,
            ),
            derived=DerivedSection(F=f_value, rho=0.0, B=1.0, C=0.5, total_messages=2, mean_message_tokens=10.0),
            provenance=ProvenanceSection(
                config_path="cfg",
                output_dir="out",
                python_version="3.12",
                platform="test",
                decision_prompt_hash="h1",
                message_serializer_hash="h2",
            ),
            events_path="out/events.jsonl",
            attack_analysis=AttackAnalysisSection() if infection is not None else None,
        )

    clean_summary = make_summary(experiment_id="clean-exp", score_mean=1.0, f_value=0.95, infection=None)
    stress_summary = make_summary(experiment_id="stress-exp", score_mean=0.5, f_value=0.55, infection=0.2)

    clean_summary_path = clean_dir / "summary.json"
    stress_summary_path = stress_dir / "summary.json"
    clean_summary_path.write_text(clean_summary.model_dump_json(indent=2), encoding="utf-8")
    stress_summary_path.write_text(stress_summary.model_dump_json(indent=2), encoding="utf-8")
    (clean_dir / "batch_index.json").write_text(
        json.dumps(
            {
                "experiment_id": "clean-exp",
                "generated_at": "2026-01-01T00:00:00Z",
                "trials": [
                    {
                        "trial_id": "clean-exp-trial",
                        "summary_path": str(clean_summary_path.resolve()),
                        "benchmark_family": "craft_mini",
                        "topology_preset": "balanced_tree",
                        "message_token_budget": 32,
                        "model_alias": "qwen35_plus_0215",
                        "baseline": "ma_ft",
                        "attack_scenario": None,
                        "seed": 7,
                        "score_mean": 1.0,
                        "accuracy": 1.0,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (stress_dir / "batch_index.json").write_text(
        json.dumps(
            {
                "experiment_id": "stress-exp",
                "generated_at": "2026-01-01T00:00:00Z",
                "trials": [
                    {
                        "trial_id": "stress-exp-trial",
                        "summary_path": str(stress_summary_path.resolve()),
                        "benchmark_family": "craft_mini",
                        "topology_preset": "balanced_tree",
                        "message_token_budget": 32,
                        "model_alias": "qwen35_plus_0215",
                        "baseline": "ma_ft",
                        "attack_scenario": "leaf",
                        "seed": 7,
                        "score_mean": 0.5,
                        "accuracy": 0.5,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = write_attack_analysis(stress_dir, clean_dir)
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows"][0]["F_delta_vs_clean"] == 0.4

    enriched_summary = TrialSummary.model_validate_json(stress_summary_path.read_text(encoding="utf-8"))
    assert enriched_summary.attack_analysis is not None
    assert enriched_summary.attack_analysis.F_delta_vs_clean == 0.4
    assert enriched_summary.attack_analysis.score_delta_vs_clean == -0.5
    assert enriched_summary.attack_analysis.attack_scenario == "leaf"
