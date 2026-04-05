from __future__ import annotations

from pathlib import Path

import yaml

from coord_harness.analysis.replay import (
    TrialArtifacts,
    _compute_b_from_artifact,
    _compute_f_from_artifact,
    recompute_experiment_deriveds,
)
from coord_harness.logging.schema import (
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
from coord_harness.runner.executor import run_batch


def test_recompute_deriveds_updates_summary_from_logs(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("archive/configs/p0a_clean_smoke.yaml").read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path)
    payload["benchmarks"][0]["task_limit"] = 1
    payload["benchmarks"][1]["task_limit"] = 1
    temp_config = tmp_path / "smoke.yaml"
    temp_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    batch_index_path = run_batch(temp_config)
    report_path = recompute_experiment_deriveds(batch_index_path)
    assert report_path.exists()

    batch_index = yaml.safe_load(batch_index_path.read_text(encoding="utf-8"))
    ma_ft_entry = next(entry for entry in batch_index["trials"] if entry["baseline"] == "ma_ft")
    summary = yaml.safe_load(Path(ma_ft_entry["summary_path"]).read_text(encoding="utf-8"))
    assert summary["derived"]["recomputed_offline"] is True
    assert summary["derived"]["C_source"] == "offline_replay:max(incoming_peer_tokens_raw / peer_context_quota)"


def test_fact_survival_extractors_do_not_copy_final_score() -> None:
    summary = TrialSummary(
        run=RunSection(
            experiment_id="exp",
            trial_id="trial",
            framework_id="coord_harness_v1",
            stage="clean",
            mode="research_strict",
            baseline="ma_ft",
            seed=7,
            status="completed",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:01Z",
            config_digest="abc",
            decision_prompt_template_version="decision-v2",
            message_serializer_version="message-v2",
        ),
        model=ModelSection(
            alias="mock-local",
            panel_tier="dev",
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
            family="agentsnet_mini",
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
            edge_count=6,
            max_in_degree=2,
            max_out_degree=2,
            edges=[
                ("agent_0", "agent_1"),
                ("agent_1", "agent_0"),
                ("agent_1", "agent_3"),
                ("agent_3", "agent_1"),
                ("agent_3", "agent_7"),
                ("agent_7", "agent_3"),
            ],
        ),
        budget=BudgetSection(
            total_billed_token_budget_per_task=100,
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
        outcomes=OutcomeSection(task_count=1, accuracy=0.0, score_mean=0.0, score_std=0.0, score_sum=0.0),
        derived=DerivedSection(total_messages=3, mean_message_tokens=6.0),
        provenance=ProvenanceSection(
            config_path="cfg",
            output_dir="out",
            python_version="3.12",
            platform="test",
            decision_prompt_hash="h1",
            message_serializer_hash="h2",
        ),
        events_path="out/events.jsonl",
    )
    tasks = {
        "t1": {
            "task_id": "t1",
            "metadata": {
                "agent_fact_payloads": {
                    "agent_7": [{"fact_id": "critical_leaf_fact", "critical": True, "supports": ["C"]}]
                }
            },
        }
    }
    events = [
        {
            "task_id": "t1",
            "event_type": "message_sent",
            "agent_id": "agent_7",
            "target_agent_id": "agent_3",
            "payload": {
                "sender_local_state_snapshot": {"answer": "C", "correct": True},
                "receiver_local_state_after": {"answer": "C", "correct": True},
            },
        },
        {
            "task_id": "t1",
            "event_type": "message_sent",
            "agent_id": "agent_3",
            "target_agent_id": "agent_1",
            "payload": {
                "sender_local_state_snapshot": {"answer": "C", "correct": True},
                "receiver_local_state_after": {"answer": "C", "correct": True},
            },
        },
        {
            "task_id": "t1",
            "event_type": "message_sent",
            "agent_id": "agent_1",
            "target_agent_id": "agent_0",
            "payload": {
                "sender_local_state_snapshot": {"answer": "C", "correct": True},
                "receiver_local_state_after": {"answer": "A", "correct": False},
            },
        },
        {
            "task_id": "t1",
            "event_type": "final_aggregate",
            "payload": {"selected_answer": "A"},
        },
    ]
    artifact = TrialArtifacts(summary_path=Path("summary.json"), summary=summary, events=events, tasks=tasks)
    f_metric, _ = _compute_f_from_artifact(artifact)
    b_metric, _ = _compute_b_from_artifact(artifact)
    assert f_metric == 0.666667
    assert b_metric < 1.0
    assert f_metric != summary.outcomes.score_mean
