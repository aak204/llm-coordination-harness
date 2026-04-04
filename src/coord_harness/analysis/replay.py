from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coord_harness.logging.schema import TrialSummary


@dataclass(frozen=True)
class TrialArtifacts:
    summary_path: Path
    summary: TrialSummary
    events: list[dict[str, Any]]
    tasks: dict[str, dict[str, Any]]


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


def _gini(values: list[float]) -> float:
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_values = sorted(values)
    weighted = sum((index + 1) * value for index, value in enumerate(sorted_values))
    n = len(values)
    return (2 * weighted) / (n * total) - (n + 1) / n


def _load_task_payloads(dataset_path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        tasks[payload["task_id"]] = payload
    return tasks


def load_trial_artifacts(experiment_dir_or_batch_index: str | Path) -> list[TrialArtifacts]:
    path = Path(experiment_dir_or_batch_index)
    batch_index_path = path if path.name == "batch_index.json" else path / "batch_index.json"
    batch_index = json.loads(batch_index_path.read_text(encoding="utf-8"))
    dataset_cache: dict[str, dict[str, dict[str, Any]]] = {}
    artifacts: list[TrialArtifacts] = []
    for entry in batch_index["trials"]:
        summary_path = Path(entry["summary_path"])
        summary = TrialSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in Path(summary.events_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dataset_path = summary.benchmark.dataset_path
        if dataset_path not in dataset_cache:
            dataset_cache[dataset_path] = _load_task_payloads(Path(dataset_path))
        artifacts.append(
            TrialArtifacts(
                summary_path=summary_path,
                summary=summary,
                events=events,
                tasks=dataset_cache[dataset_path],
            )
        )
    return artifacts


def _trial_key(summary: TrialSummary) -> tuple[str, str, int, str, int]:
    return (
        summary.benchmark.family,
        summary.topology.preset,
        summary.budget.message_token_budget,
        summary.model.alias,
        summary.run.seed,
    )


def _build_parent_children(summary: TrialSummary) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    neighbors: dict[str, list[str]] = defaultdict(list)
    for left, right in summary.topology.edges:
        neighbors[left].append(right)
    root = summary.topology.root_agent
    parent: dict[str, str | None] = {root: None}
    children: dict[str, list[str]] = defaultdict(list)
    queue: deque[str] = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in neighbors[node]:
            if neighbor in parent:
                continue
            parent[neighbor] = node
            children[node].append(neighbor)
            queue.append(neighbor)
    return parent, children


def _subtree_agents(children: dict[str, list[str]], root: str) -> list[str]:
    agents = [root]
    stack = list(children.get(root, []))
    while stack:
        node = stack.pop()
        agents.append(node)
        stack.extend(children.get(node, []))
    return sorted(agents)


def _critical_fact_payloads(task_payload: dict[str, Any], agent_ids: list[str]) -> list[dict[str, Any]]:
    fact_payloads = task_payload.get("metadata", {}).get("agent_fact_payloads", {})
    results: list[dict[str, Any]] = []
    for agent_id in agent_ids:
        for payload in fact_payloads.get(agent_id, []):
            if payload.get("critical") is True:
                item = dict(payload)
                item["agent_id"] = agent_id
                results.append(item)
    return results


def _answer_supports_fact(answer: str, fact_payload: dict[str, Any]) -> bool:
    supports = fact_payload.get("supports")
    if supports:
        return answer in supports
    contradicts = fact_payload.get("contradicts")
    if contradicts:
        return answer not in contradicts
    return True


def _find_root_answer(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event["event_type"] == "final_aggregate":
            return event["payload"].get("selected_answer")
    return None


def _message_event_index(events: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        if event["event_type"] != "message_sent":
            continue
        key = (event["task_id"], event["agent_id"], event["target_agent_id"])
        index[key] = event
    return index


def _path_to_root(parent: dict[str, str | None], agent_id: str) -> list[tuple[str, str]]:
    hops: list[tuple[str, str]] = []
    current = agent_id
    while parent.get(current) is not None:
        parent_node = parent[current]
        hops.append((current, parent_node))
        current = parent_node
    return hops


def _compute_f_from_artifact(artifact: TrialArtifacts) -> tuple[float | None, str | None]:
    if artifact.summary.run.baseline != "ma_ft":
        return None, None
    parent, _children = _build_parent_children(artifact.summary)
    event_index = _message_event_index(artifact.events)
    survival_scores: list[float] = []
    for task_id, task_payload in artifact.tasks.items():
        fact_payloads = _critical_fact_payloads(task_payload, agent_ids=sorted(parent))
        for fact_payload in fact_payloads:
            origin = fact_payload["agent_id"]
            for sender, receiver in _path_to_root(parent, origin):
                event = event_index.get((task_id, sender, receiver))
                if event is None:
                    continue
                sender_state = event["payload"].get("sender_local_state_snapshot", {})
                receiver_after = event["payload"].get("receiver_local_state_after", {})
                sender_answer = sender_state.get("answer")
                receiver_answer = receiver_after.get("answer")
                if sender_answer is None or receiver_answer is None:
                    continue
                sender_supports = _answer_supports_fact(sender_answer, fact_payload)
                receiver_supports = _answer_supports_fact(receiver_answer, fact_payload)
                survival_scores.append(1.0 if sender_supports and receiver_supports else 0.0)
    if not survival_scores:
        return None, None
    return round(sum(survival_scores) / len(survival_scores), 6), "offline_replay:critical_fact_survival_per_hop"


def _compute_c_from_events(events: list[dict[str, Any]]) -> tuple[float | None, dict[str, int], str]:
    pressures: list[float] = []
    total_dropped = 0
    max_incoming = 0
    max_quota = 0
    for event in events:
        if event["event_type"] not in {"agent_decision", "fusion_skipped"}:
            continue
        payload = event["payload"]
        if event["event_type"] == "agent_decision" and payload.get("phase") != "fusion":
            continue
        incoming_raw = int(payload.get("incoming_peer_tokens_raw", 0))
        peer_context_quota = int(payload.get("peer_context_quota", 0))
        dropped_raw = int(payload.get("dropped_peer_tokens_raw", 0))
        total_dropped += dropped_raw
        max_incoming = max(max_incoming, incoming_raw)
        max_quota = max(max_quota, peer_context_quota)
        if peer_context_quota > 0:
            pressures.append(incoming_raw / peer_context_quota)
        elif incoming_raw > 0:
            pressures.append(1.0)
    metric = round(max(pressures), 6) if pressures else 0.0
    return metric, {
        "total_dropped_peer_tokens_raw": total_dropped,
        "max_incoming_peer_tokens_raw": max_incoming,
        "max_peer_context_quota": max_quota,
    }, "offline_replay:max(incoming_peer_tokens_raw / peer_context_quota)"


def _compute_b_from_artifact(artifact: TrialArtifacts) -> tuple[float | None, str]:
    if artifact.summary.run.baseline != "ma_ft":
        return 1.0, "computed:no_message_flow_baseline"
    parent, children = _build_parent_children(artifact.summary)
    edge_available_mass: dict[str, float] = defaultdict(float)
    edge_surviving_mass: dict[str, float] = defaultdict(float)
    event_index = _message_event_index(artifact.events)

    for task_id, task_payload in artifact.tasks.items():
        fact_payloads = _critical_fact_payloads(task_payload, agent_ids=sorted(parent))
        for fact_payload in fact_payloads:
            origin = fact_payload["agent_id"]
            for sender, receiver in _path_to_root(parent, origin):
                event = event_index.get((task_id, sender, receiver))
                if event is None:
                    continue
                sender_state = event["payload"].get("sender_local_state_snapshot", {})
                receiver_after = event["payload"].get("receiver_local_state_after", {})
                sender_answer = sender_state.get("answer")
                receiver_answer = receiver_after.get("answer")
                if sender_answer is None or receiver_answer is None:
                    continue
                sender_supports = _answer_supports_fact(sender_answer, fact_payload)
                edge_key = f"{sender}->{receiver}"
                if sender_supports:
                    edge_available_mass[edge_key] += 1.0
                    if _answer_supports_fact(receiver_answer, fact_payload):
                        edge_surviving_mass[edge_key] += 1.0

    ratios = []
    for edge_key, available_mass in edge_available_mass.items():
        if available_mass > 0:
            ratios.append(edge_surviving_mass[edge_key] / available_mass)

    if not ratios:
        return 1.0, "computed:no_critical_fact_flow"
    metric = round(1 - _gini(ratios), 6)
    return metric, "offline_replay:1_minus_gini(edge_fact_survival_ratio)"


def _extract_vote_local_error_sequences(artifacts: TrialArtifacts) -> dict[str, list[int]]:
    by_agent: dict[str, list[int]] = defaultdict(list)
    task_agent_state: dict[tuple[str, str], int] = {}
    for event in artifacts.events:
        if event["event_type"] != "agent_decision":
            continue
        if event["baseline"] != "vote_local":
            continue
        payload = event["payload"]
        if payload.get("phase") != "local":
            continue
        snapshot = payload.get("local_state_snapshot", {})
        correct = snapshot.get("correct")
        if correct is None:
            continue
        task_agent_state[(event["task_id"], event["agent_id"])] = 0 if correct else 1
    for task_id, agent_id in sorted(task_agent_state):
        by_agent[agent_id].append(task_agent_state[(task_id, agent_id)])
    return by_agent


def _compute_rho_from_vote_local(vote_local_artifacts: TrialArtifacts) -> tuple[float | None, str | None]:
    by_agent = _extract_vote_local_error_sequences(vote_local_artifacts)
    agent_ids = sorted(by_agent)
    rho_samples = []
    for idx, left_agent in enumerate(agent_ids):
        for right_agent in agent_ids[idx + 1 :]:
            rho_samples.append(_pearson_binary(by_agent[left_agent], by_agent[right_agent]))
    if not rho_samples:
        return None, None
    return round(sum(rho_samples) / len(rho_samples), 6), "offline_replay:paired_vote_local_no_communication_baseline"


def recompute_experiment_deriveds(experiment_dir_or_batch_index: str | Path) -> Path:
    path = Path(experiment_dir_or_batch_index)
    experiment_dir = path.parent if path.name == "batch_index.json" else path
    artifacts = load_trial_artifacts(experiment_dir_or_batch_index)
    by_key = {_trial_key(artifact.summary): artifact for artifact in artifacts}
    report: dict[str, Any] = {"updated_trials": []}

    for artifact in artifacts:
        summary = artifact.summary
        f_metric, f_source = _compute_f_from_artifact(artifact)
        c_metric, c_aux, c_source = _compute_c_from_events(artifact.events)
        b_metric, b_source = _compute_b_from_artifact(artifact)

        rho_metric = None
        rho_source = None
        for candidate in artifacts:
            if (
                candidate.summary.run.baseline == "vote_local"
                and _trial_key(candidate.summary) == _trial_key(summary)
            ):
                rho_metric, rho_source = _compute_rho_from_vote_local(candidate)
                break

        summary.derived.F = f_metric
        summary.derived.rho = rho_metric
        summary.derived.B = b_metric
        summary.derived.C = c_metric
        summary.derived.B_status = "computed"
        summary.derived.recomputed_offline = True
        summary.derived.F_source = f_source
        summary.derived.rho_source = rho_source
        summary.derived.B_source = b_source
        summary.derived.C_source = c_source
        summary.derived.total_dropped_peer_tokens_raw = c_aux["total_dropped_peer_tokens_raw"]
        summary.derived.max_incoming_peer_tokens_raw = c_aux["max_incoming_peer_tokens_raw"]
        summary.derived.max_peer_context_quota = c_aux["max_peer_context_quota"]

        notes = []
        if rho_metric is None:
            notes.append("rho unavailable from paired vote_local baseline.")
        if f_metric is None:
            notes.append("F unavailable because no critical fact flow was logged.")
        notes.append(f"B source: {b_source}")
        summary.derived.notes = notes

        artifact.summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        report["updated_trials"].append(
            {
                "trial_id": summary.run.trial_id,
                "F": f_metric,
                "rho": rho_metric,
                "B": b_metric,
                "C": c_metric,
            }
        )

    output_path = experiment_dir / "derived_replay_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
