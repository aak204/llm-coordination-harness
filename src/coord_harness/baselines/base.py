from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from coord_harness.attacks import AttackInjector
from coord_harness.config.loader import TrialConfig
from coord_harness.config.models import BenchmarkAdapterConfig
from coord_harness.core.budget import BudgetLedger
from coord_harness.core.protocols import SerializedPeerMessage, serialize_peer_message, visible_message_token_count
from coord_harness.core.topology import Topology
from coord_harness.core.types import AgentDecision, BenchmarkTask, GenerationResult, MessageTrace, TaskRunTrace
from coord_harness.models.base import ModelClient


@dataclass(frozen=True)
class StrategyContext:
    trial: TrialConfig
    topology: Topology
    benchmark_config: BenchmarkAdapterConfig
    model_client: ModelClient


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_message_content(result: GenerationResult, message_token_budget: int) -> SerializedPeerMessage:
    return serialize_peer_message(
        answer=result.answer,
        confidence=result.confidence,
        rationale=result.rationale,
        message_token_budget=message_token_budget,
    )


def vote_from_decisions(decisions: list[AgentDecision]) -> tuple[str, list[dict[str, Any]]]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for decision in decisions:
        totals[decision.answer] = totals.get(decision.answer, 0.0) + decision.confidence
        counts[decision.answer] = counts.get(decision.answer, 0) + 1
    ordered = sorted(
        totals,
        key=lambda answer: (
            counts[answer],
            totals[answer],
            answer,
        ),
        reverse=True,
    )
    breakdown = [
        {"answer": answer, "vote_count": counts[answer], "confidence_sum": round(totals[answer], 4)}
        for answer in ordered
    ]
    return ordered[0], breakdown


class BaselineExecutor(ABC):
    name: str

    @abstractmethod
    def run_task(self, *, task: BenchmarkTask, context: StrategyContext) -> tuple[TaskRunTrace, list[dict[str, Any]]]:
        raise NotImplementedError

    @staticmethod
    def invoke_agent(
        *,
        task: BenchmarkTask,
        agent_id: str,
        visible_messages: list[str],
        context: StrategyContext,
        ledger: BudgetLedger,
    ) -> tuple[AgentDecision, GenerationResult]:
        result = context.model_client.generate_decision(
            task=task,
            agent_id=agent_id,
            visible_messages=visible_messages,
            seed=context.trial.seed,
        )
        ledger.consume(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            label=f"{task.task_id}:{agent_id}",
        )
        decision = AgentDecision(
            agent_id=agent_id,
            answer=result.answer,
            confidence=result.confidence,
            rationale=result.rationale,
            correct=result.answer == task.gold_answer,
        )
        return decision, result

    @staticmethod
    def build_attack_injector(context: StrategyContext) -> AttackInjector | None:
        injector = AttackInjector.from_context(context)
        if not injector.is_enabled_for_trial():
            return None
        return injector

    @staticmethod
    def maybe_compromise_agent(
        *,
        task: BenchmarkTask,
        agent_id: str,
        decision: AgentDecision,
        result: GenerationResult,
        attack_injector: AttackInjector | None,
    ) -> tuple[AgentDecision, GenerationResult, dict[str, Any] | None]:
        if attack_injector is None:
            return decision, result, None
        return attack_injector.maybe_inject(task=task, agent_id=agent_id, decision=decision, result=result)

    @staticmethod
    def finalize_trace_budget(
        *,
        trace: TaskRunTrace,
        model_call_count: int,
    ) -> None:
        inter_agent_tokens = sum(message.token_count for message in trace.messages)
        trace.budget_usage["inter_agent_tokens"] = inter_agent_tokens
        trace.budget_usage["model_call_count"] = model_call_count

    @staticmethod
    def event(
        *,
        task_id: str,
        baseline: str,
        event_type: str,
        payload: dict[str, Any],
        agent_id: str | None = None,
        target_agent_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "timestamp": utc_now_iso(),
            "task_id": task_id,
            "baseline": baseline,
            "event_type": event_type,
            "agent_id": agent_id,
            "target_agent_id": target_agent_id,
            "payload": payload,
        }

    @staticmethod
    def decision_payload(
        *,
        decision: AgentDecision,
        result: GenerationResult,
        visible_messages: list[str],
        phase: str,
        local_state_snapshot: dict[str, Any] | None = None,
        peer_context_quota: int | None = None,
        incoming_peer_tokens_raw: int | None = None,
        incoming_peer_tokens_used: int | None = None,
        dropped_peer_tokens_raw: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "answer": decision.answer,
            "confidence": decision.confidence,
            "route_metadata": result.route_metadata,
            "phase": phase,
            "visible_message_count": len(visible_messages),
            "visible_message_token_count": visible_message_token_count(visible_messages),
        }
        if local_state_snapshot is not None:
            payload["local_state_snapshot"] = local_state_snapshot
        if peer_context_quota is not None:
            payload["peer_context_quota"] = peer_context_quota
        if incoming_peer_tokens_raw is not None:
            payload["incoming_peer_tokens_raw"] = incoming_peer_tokens_raw
        if incoming_peer_tokens_used is not None:
            payload["incoming_peer_tokens_used"] = incoming_peer_tokens_used
        if dropped_peer_tokens_raw is not None:
            payload["dropped_peer_tokens_raw"] = dropped_peer_tokens_raw
        return payload

    @staticmethod
    def local_state_snapshot(decision: AgentDecision) -> dict[str, Any]:
        return {
            "answer": decision.answer,
            "confidence": decision.confidence,
            "correct": decision.correct,
        }

    @staticmethod
    def message_trace(
        *,
        message_id: str,
        task_id: str,
        sender: str,
        receiver: str,
        serialized_message: SerializedPeerMessage,
        peer_context_quota: int,
        sender_state: AgentDecision,
        receiver_before: AgentDecision,
        receiver_after: AgentDecision,
    ) -> MessageTrace:
        return MessageTrace(
            message_id=message_id,
            task_id=task_id,
            sender=sender,
            receiver=receiver,
            content=serialized_message.used_content,
            token_count=serialized_message.used_token_count,
            raw_content=serialized_message.raw_content,
            raw_token_count=serialized_message.raw_token_count,
            dropped_token_count=serialized_message.dropped_token_count,
            peer_context_quota=peer_context_quota,
            sender_answer=sender_state.answer,
            sender_correct=sender_state.correct,
            receiver_answer_before=receiver_before.answer,
            receiver_correct_before=receiver_before.correct,
            receiver_answer_after=receiver_after.answer,
            receiver_correct_after=receiver_after.correct,
        )

    @staticmethod
    def attack_metrics(
        *,
        task: BenchmarkTask,
        final_states: list[AgentDecision],
        selected_answer: str,
        attack_injector: AttackInjector | None,
    ) -> dict[str, Any]:
        if attack_injector is None:
            return {}
        return attack_injector.attack_metrics(task=task, final_states=final_states, selected_answer=selected_answer)
