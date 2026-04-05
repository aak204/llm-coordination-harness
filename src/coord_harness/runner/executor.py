from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from coord_harness.analysis.attack_analysis import write_attack_analysis
from coord_harness.analysis.replay import recompute_experiment_deriveds
from coord_harness.baselines import build_baseline_executor
from coord_harness.baselines.base import StrategyContext
from coord_harness.benchmarks import build_benchmark_adapter
from coord_harness.benchmarks.base import BenchmarkDataset
from coord_harness.config.loader import TrialConfig, config_to_canonical_json, expand_trials, load_config
from coord_harness.config.models import BatchConfig
from coord_harness.core.enums import BaselineStrategy, RegimeLabel, RunMode
from coord_harness.core.protocols import (
    DECISION_PROMPT_TEMPLATE_VERSION,
    MESSAGE_SERIALIZER_VERSION,
    decision_prompt_hash,
    message_serializer_hash,
)
from coord_harness.core.topology import build_topology
from coord_harness.core.types import TaskRunTrace
from coord_harness.logging.schema import (
    BatchIndex,
    BatchIndexEntry,
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
from coord_harness.logging.writer import ArtifactWriter
from coord_harness.models import build_model_client


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_git_value(command: list[str]) -> str | None:
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output or None


def _git_provenance() -> tuple[str | None, bool | None]:
    commit = _safe_git_value(["git", "rev-parse", "HEAD"])
    status = _safe_git_value(["git", "status", "--porcelain"])
    dirty = bool(status) if status is not None else None
    return commit, dirty


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return round(statistics.pstdev(values), 6)


def _pearson_binary(xs: list[int], ys: list[int]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_values = sorted(values)
    weighted = sum((index + 1) * value for index, value in enumerate(sorted_values))
    n = len(values)
    return (2 * weighted) / (n * total) - (n + 1) / n


def derive_core_metrics(traces: list[TaskRunTrace], agent_count: int) -> DerivedSection:
    total_messages = sum(len(trace.messages) for trace in traces)
    message_tokens = [message.token_count for trace in traces for message in trace.messages]

    fidelity_samples = [
        1.0 if message.sender_correct == message.receiver_correct_after else 0.0
        for trace in traces
        for message in trace.messages
    ]
    f_metric = round(sum(fidelity_samples) / len(fidelity_samples), 6) if fidelity_samples else None

    sent_counts = {f"agent_{idx}": 0 for idx in range(agent_count)}
    recv_counts = {f"agent_{idx}": 0 for idx in range(agent_count)}
    for trace in traces:
        for message in trace.messages:
            sent_counts[message.sender] += 1
            recv_counts[message.receiver] += 1

    initial_error_by_agent: dict[str, list[int]] = defaultdict(list)
    for trace in traces:
        for state in trace.initial_states:
            initial_error_by_agent[state.agent_id].append(0 if state.correct else 1)
    rho_samples: list[float] = []
    agent_ids = sorted(initial_error_by_agent)
    for idx, left_agent in enumerate(agent_ids):
        for right_agent in agent_ids[idx + 1 :]:
            rho_samples.append(_pearson_binary(initial_error_by_agent[left_agent], initial_error_by_agent[right_agent]))
    rho_metric = round(sum(rho_samples) / len(rho_samples), 6) if rho_samples else None

    balance = round(1 - _gini(list(sent_counts.values())), 6) if sent_counts else None
    fan_in = round(max(recv_counts.values(), default=0) / total_messages, 6) if total_messages else 0.0

    notes = []
    if total_messages == 0:
        notes.append("No inter-agent messages observed; F is undefined and C collapses to 0.")
    if len(agent_ids) <= 1:
        notes.append("rho is undefined or degenerate for single-agent traces.")

    return DerivedSection(
        F=f_metric,
        rho=rho_metric,
        B=balance,
        C=fan_in,
        B_status="placeholder:in_memory_message_flow_balance",
        recomputed_offline=False,
        F_source="in_memory_prewrite",
        rho_source="in_memory_prewrite",
        C_source="in_memory_prewrite",
        total_messages=total_messages,
        mean_message_tokens=round(sum(message_tokens) / len(message_tokens), 4) if message_tokens else 0.0,
        total_dropped_peer_tokens_raw=0,
        max_incoming_peer_tokens_raw=0,
        max_peer_context_quota=0,
        notes=notes,
    )


def _build_summary(
    *,
    batch_config: BatchConfig,
    benchmark_dataset: BenchmarkDataset,
    trial: TrialConfig,
    traces: list[TaskRunTrace],
    started_at: str,
    completed_at: str,
    output_dir: Path,
    route_metadata_examples: list[dict],
    runtime_metadata: dict,
    topology_edges: list[tuple[str, str]],
    topology_summary: dict[str, int | str],
    model_catalog_snapshot_path: Path | None,
) -> TrialSummary:
    scores = [trace.score for trace in traces]
    billed_tokens = [trace.budget_usage["billed_tokens"] for trace in traces]
    prompt_tokens = [trace.budget_usage["prompt_tokens"] for trace in traces]
    completion_tokens = [trace.budget_usage["completion_tokens"] for trace in traces]
    inter_agent_tokens = [trace.budget_usage.get("inter_agent_tokens", 0) for trace in traces]
    model_call_counts = [trace.budget_usage.get("model_call_count", 0) for trace in traces]
    accuracy = round(sum(1 for trace in traces if trace.selected_correct) / len(traces), 6) if traces else 0.0
    attack_success_values = [float(trace.metadata["attack_success"]) for trace in traces if "attack_success" in trace.metadata]
    infection_spread_values = [float(trace.metadata["infection_spread"]) for trace in traces if "infection_spread" in trace.metadata]
    quarantine_strength_values = [float(trace.metadata["quarantine_strength"]) for trace in traces if "quarantine_strength" in trace.metadata]
    git_commit, git_dirty = _git_provenance()
    return TrialSummary(
        run=RunSection(
            experiment_id=trial.experiment_id,
            trial_id=trial.trial_id,
            framework_id=trial.framework_id,
            stage=trial.stage.value,
            mode=batch_config.run.mode.value,
            baseline=trial.baseline.value,
            attack_scenario=trial.attack_scenario_name,
            attack_injection_depth=trial.attack.get("injection_depth"),
            seed=trial.seed,
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            config_digest=trial.config_digest,
            decision_prompt_template_version=DECISION_PROMPT_TEMPLATE_VERSION,
            message_serializer_version=MESSAGE_SERIALIZER_VERSION,
        ),
        model=ModelSection(
            alias=trial.model_spec.alias,
            panel_tier=trial.model_spec.panel_tier.value,
            provider=trial.model_spec.provider.value,
            model_id=trial.model_spec.model_id,
            requested_model_id=runtime_metadata.get("requested_model_id", trial.model_spec.model_id),
            allow_auto_routing=trial.model_spec.allow_auto_routing,
            routing=trial.model_spec.routing.model_dump(mode="json"),
            temperature=trial.model_spec.temperature,
            top_p=trial.model_spec.top_p,
            max_completion_tokens=trial.model_spec.max_completion_tokens,
            strict_routing_required=batch_config.run.mode is RunMode.RESEARCH_STRICT,
            runtime_metadata=runtime_metadata,
            route_metadata_examples=route_metadata_examples,
        ),
        benchmark=BenchmarkSection(
            family=trial.benchmark_family.value,
            stage=benchmark_dataset.manifest.stage.value,
            split=batch_config.benchmark_config_for(trial.benchmark_family).split,
            dataset_revision=benchmark_dataset.manifest.dataset_revision,
            task_schema_version=benchmark_dataset.manifest.task_schema_version,
            dataset_digest=benchmark_dataset.dataset_digest,
            dataset_path=str(benchmark_dataset.dataset_path.resolve()),
            manifest_path=str(benchmark_dataset.manifest_path.resolve()),
            source_kind=benchmark_dataset.manifest.source_kind,
            stubbed=batch_config.benchmark_config_for(trial.benchmark_family).stubbed,
            task_count=len(benchmark_dataset.tasks),
            task_ids=[task.task_id for task in benchmark_dataset.tasks],
            scoring="exact_match_binary",
        ),
        topology=TopologySection(
            preset=trial.topology_preset.value,
            agent_count=trial.agent_count,
            root_agent=str(topology_summary["root_agent"]),
            edge_count=int(topology_summary["edge_count"]),
            max_in_degree=int(topology_summary["max_in_degree"]),
            max_out_degree=int(topology_summary["max_out_degree"]),
            edges=topology_edges,
        ),
        budget=BudgetSection(
            total_billed_token_budget_per_task=trial.total_billed_token_budget,
            message_token_budget=trial.message_token_budget,
            task_prompt_tokens=prompt_tokens,
            task_completion_tokens=completion_tokens,
            task_billed_tokens=billed_tokens,
            task_inter_agent_tokens=inter_agent_tokens,
            task_model_call_counts=model_call_counts,
            mean_billed_tokens=_mean(billed_tokens),
            max_billed_tokens=max(billed_tokens, default=0),
            mean_inter_agent_tokens=_mean(inter_agent_tokens),
            max_inter_agent_tokens=max(inter_agent_tokens, default=0),
        ),
        outcomes=OutcomeSection(
            task_count=len(traces),
            accuracy=accuracy,
            score_mean=round(sum(scores) / len(scores), 6) if scores else 0.0,
            score_std=_stdev(scores),
            score_sum=round(sum(scores), 6),
            attacked_tasks=len(attack_success_values),
            attack_success_rate=round(sum(attack_success_values) / len(attack_success_values), 6) if attack_success_values else None,
            infection_spread_rate=round(sum(infection_spread_values) / len(infection_spread_values), 6) if infection_spread_values else None,
            quarantine_strength=(
                round(sum(quarantine_strength_values) / len(quarantine_strength_values), 6)
                if quarantine_strength_values
                else None
            ),
        ),
        derived=derive_core_metrics(traces, trial.agent_count),
        provenance=ProvenanceSection(
            config_path=str(trial.config_path.resolve()),
            output_dir=str(output_dir.resolve()),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            git_commit=git_commit,
            git_dirty=git_dirty,
            model_catalog_snapshot_path=str(model_catalog_snapshot_path.resolve()) if model_catalog_snapshot_path else None,
            decision_prompt_hash=decision_prompt_hash(),
            message_serializer_hash=message_serializer_hash(),
            model_catalog_snapshot_hash=runtime_metadata.get("catalog_snapshot_hash"),
            model_catalog_snapshot_fetched_at=runtime_metadata.get("catalog_snapshot_fetched_at"),
        ),
        events_path=str((output_dir / "events.jsonl").resolve()),
    )


def run_trial(batch_config: BatchConfig, trial: TrialConfig) -> tuple[TrialSummary, Path]:
    output_dir = trial.output_root / trial.experiment_id / trial.trial_id
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        existing_summary = TrialSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        return existing_summary, summary_path

    started_at = utc_now_iso()
    topology = build_topology(trial.topology_preset, trial.agent_count, trial.seed)
    benchmark_adapter = build_benchmark_adapter(batch_config.benchmark_config_for(trial.benchmark_family))
    benchmark_dataset = benchmark_adapter.load_dataset()
    model_client = build_model_client(trial.model_spec)
    strategy = build_baseline_executor(trial.baseline)
    context = StrategyContext(
        trial=trial,
        topology=topology,
        benchmark_config=batch_config.benchmark_config_for(trial.benchmark_family),
        model_client=model_client,
    )

    writer = ArtifactWriter(output_dir)
    model_catalog_snapshot_path: Path | None = None
    if batch_config.run.capture_model_catalog_snapshot:
        snapshot = model_client.fetch_model_catalog_snapshot()
        if snapshot:
            model_catalog_snapshot_path = writer.write_json("model_catalog_snapshot.json", snapshot)

    traces: list[TaskRunTrace] = []
    events: list[dict] = []
    route_metadata_examples: list[dict] = []
    runtime_metadata = {}
    try:
        runtime_metadata = model_client.describe_runtime_metadata()
        for task in benchmark_dataset.tasks:
            trace, task_events = strategy.run_task(task=task, context=context)
            traces.append(trace)
            events.extend(task_events)
            route_metadata_examples.extend(
                event["payload"]["route_metadata"]
                for event in task_events
                if event["event_type"] == "agent_decision" and "route_metadata" in event["payload"]
            )
    finally:
        close_method = getattr(model_client, "close", None)
        if callable(close_method):
            close_method()

    completed_at = utc_now_iso()
    summary = _build_summary(
        batch_config=batch_config,
        benchmark_dataset=benchmark_dataset,
        trial=trial,
        traces=traces,
        started_at=started_at,
        completed_at=completed_at,
        output_dir=output_dir,
        route_metadata_examples=route_metadata_examples[:3],
        runtime_metadata=runtime_metadata,
        topology_edges=topology.edges,
        topology_summary=topology.summary(),
        model_catalog_snapshot_path=model_catalog_snapshot_path,
    )
    writer.write_events(events)
    writer.write_summary(summary)
    return summary, writer.summary_path


def _comparison_key(summary: TrialSummary) -> tuple[str, str, int, str, str | None, int]:
    return (
        summary.benchmark.family,
        summary.topology.preset,
        summary.budget.message_token_budget,
        summary.model.alias,
        summary.run.attack_scenario,
        summary.run.seed,
    )


def _assign_regime(lift: float, thresholds) -> RegimeLabel:
    if lift >= thresholds.help_min_lift:
        return RegimeLabel.HELP
    if lift <= thresholds.collapse_max_lift:
        return RegimeLabel.COLLAPSE
    return RegimeLabel.SATURATION


def postprocess_summaries(*, summaries: list[tuple[TrialSummary, Path]], batch_config: BatchConfig) -> BatchIndex:
    sa_scores = {
        _comparison_key(summary): summary.outcomes.score_mean
        for summary, _path in summaries
        if summary.run.baseline == BaselineStrategy.SA_STAR.value
    }
    entries: list[BatchIndexEntry] = []
    for summary, path in summaries:
        lift = None
        regime = None
        if summary.run.baseline != BaselineStrategy.SA_STAR.value:
            baseline_score = sa_scores.get(_comparison_key(summary))
            if baseline_score is not None:
                lift = round(summary.outcomes.score_mean - baseline_score, 6)
                regime = _assign_regime(lift, batch_config.run.regime_thresholds)
                summary.outcomes.lift_vs_sa_star = lift
                summary.outcomes.regime = regime
                path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        entries.append(
            BatchIndexEntry(
                trial_id=summary.run.trial_id,
                summary_path=str(path.resolve()),
                benchmark_family=summary.benchmark.family,
                topology_preset=summary.topology.preset,
                message_token_budget=summary.budget.message_token_budget,
                model_alias=summary.model.alias,
                baseline=summary.run.baseline,
                attack_scenario=summary.run.attack_scenario,
                seed=summary.run.seed,
                score_mean=summary.outcomes.score_mean,
                accuracy=summary.outcomes.accuracy,
                lift_vs_sa_star=lift,
                regime=regime,
            )
        )
    return BatchIndex(
        experiment_id=batch_config.run.experiment_id,
        generated_at=utc_now_iso(),
        trials=entries,
    )


def run_batch(config_path: str | Path) -> Path:
    batch_config, config_digest = load_config(config_path)
    trials = expand_trials(batch_config, config_digest, config_path)
    output_root = batch_config.run.output_root / batch_config.run.experiment_id
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config_snapshot.json").write_text(config_to_canonical_json(batch_config), encoding="utf-8")
    summaries: list[tuple[TrialSummary, Path]] = []
    sorted_trials = sorted(trials, key=lambda trial: trial.trial_id)
    with ThreadPoolExecutor(max_workers=batch_config.run.max_concurrency) as executor:
        future_map = {
            executor.submit(run_trial, batch_config, trial): trial.trial_id
            for trial in sorted_trials
        }
        for future in as_completed(future_map):
            summaries.append(future.result())
    summaries.sort(key=lambda item: item[0].run.trial_id)
    batch_index = postprocess_summaries(summaries=summaries, batch_config=batch_config)
    batch_index_path = output_root / "batch_index.json"
    batch_index_path.write_text(batch_index.model_dump_json(indent=2), encoding="utf-8")
    recompute_experiment_deriveds(batch_index_path)
    if (
        batch_config.run.stage.value == "stress"
        and batch_config.run.attack.enabled
        and batch_config.run.attack.clean_reference_experiment_dir is not None
    ):
        write_attack_analysis(output_root, batch_config.run.attack.clean_reference_experiment_dir)
    return batch_index_path


def validate_only(config_path: str | Path) -> str:
    batch_config, config_digest = load_config(config_path)
    expanded = expand_trials(batch_config, config_digest, config_path)
    payload = {
        "experiment_id": batch_config.run.experiment_id,
        "stage": batch_config.run.stage.value,
        "mode": batch_config.run.mode.value,
        "trial_count": len(expanded),
        "benchmarks": [family.value for family in batch_config.sweep.benchmark_families],
        "topologies": [preset.value for preset in batch_config.sweep.topology_presets],
        "message_token_budgets": batch_config.sweep.message_token_budgets,
        "model_aliases": batch_config.selected_model_aliases(),
        "model_tiers": [tier.value for tier in batch_config.sweep.model_tiers],
        "baselines": [strategy.value for strategy in batch_config.sweep.baselines],
        "attack_scenarios": [scenario.name for scenario in batch_config.sweep.attack_scenarios],
        "seeds": batch_config.sweep.seeds,
    }
    return json.dumps(payload, indent=2, sort_keys=True)
