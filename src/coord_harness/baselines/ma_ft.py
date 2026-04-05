from __future__ import annotations

from collections import defaultdict

from coord_harness.baselines.base import BaselineExecutor, StrategyContext, make_message_content
from coord_harness.core.budget import BudgetLedger
from coord_harness.core.types import AgentDecision, BenchmarkTask, GenerationResult, TaskRunTrace


def _decision_to_generation(decision: AgentDecision) -> GenerationResult:
    return GenerationResult(
        answer=decision.answer,
        confidence=decision.confidence,
        rationale=decision.rationale,
        raw_text="",
        prompt_tokens=0,
        completion_tokens=0,
        route_metadata={},
    )


class MAFTExecutor(BaselineExecutor):
    name = "ma_ft"

    def run_task(self, *, task: BenchmarkTask, context: StrategyContext) -> tuple[TaskRunTrace, list[dict]]:
        ledger = BudgetLedger(context.trial.total_billed_token_budget)
        attack_injector = self.build_attack_injector(context)
        initial_states: dict[str, AgentDecision] = {}
        final_states: dict[str, AgentDecision] = {}
        events: list[dict] = []
        messages = []
        model_call_count = 0

        for agent_id in context.topology.agent_ids:
            decision, result = self.invoke_agent(
                task=task,
                agent_id=agent_id,
                visible_messages=[],
                context=context,
                ledger=ledger,
            )
            decision, result, attack_payload = self.maybe_compromise_agent(
                task=task,
                agent_id=agent_id,
                decision=decision,
                result=result,
                attack_injector=attack_injector,
            )
            model_call_count += 1
            initial_states[agent_id] = decision
            final_states[agent_id] = decision
            if attack_payload is not None:
                events.append(
                    self.event(
                        task_id=task.task_id,
                        baseline=self.name,
                        event_type="attack_applied",
                        agent_id=agent_id,
                        payload=attack_payload,
                    )
                )
            events.append(
                self.event(
                    task_id=task.task_id,
                    baseline=self.name,
                    event_type="agent_decision",
                    agent_id=agent_id,
                    payload=self.decision_payload(
                        decision=decision,
                        result=result,
                        visible_messages=[],
                        phase="local",
                        local_state_snapshot=self.local_state_snapshot(decision),
                    ),
                )
            )

        parent_map = context.topology.shortest_path_parents_to_root()
        children: dict[str, list[str]] = defaultdict(list)
        for child, parent in parent_map.items():
            if parent is not None:
                children[parent].append(child)

        depths = _compute_depths(parent_map, context.topology.root_agent)
        order = sorted(context.topology.agent_ids, key=lambda agent_id: depths.get(agent_id, 0), reverse=True)
        for receiver in order:
            child_nodes = sorted(children.get(receiver, []))
            if not child_nodes:
                continue

            receiver_before = final_states[receiver]
            visible_messages = []
            child_message_payloads: list[tuple[str, object, AgentDecision]] = []
            incoming_peer_tokens_raw = 0
            incoming_peer_tokens_used = 0
            dropped_peer_tokens_raw = 0
            peer_context_quota = len(child_nodes) * context.trial.message_token_budget
            for sender in child_nodes:
                sender_state = final_states[sender]
                serialized = make_message_content(_decision_to_generation(sender_state), context.trial.message_token_budget)
                incoming_peer_tokens_raw += serialized.raw_token_count
                incoming_peer_tokens_used += serialized.used_token_count
                dropped_peer_tokens_raw += serialized.dropped_token_count
                child_message_payloads.append((sender, serialized, sender_state))
                if not serialized.used_content:
                    continue
                visible_messages.append(serialized.used_content)

            if not visible_messages:
                events.append(
                    self.event(
                        task_id=task.task_id,
                        baseline=self.name,
                        event_type="fusion_skipped",
                        agent_id=receiver,
                        payload={
                            "reason": "no_visible_peer_messages_after_quota",
                            "local_state_snapshot": self.local_state_snapshot(receiver_before),
                            "peer_context_quota": peer_context_quota,
                            "incoming_peer_tokens_raw": incoming_peer_tokens_raw,
                            "incoming_peer_tokens_used": incoming_peer_tokens_used,
                            "dropped_peer_tokens_raw": dropped_peer_tokens_raw,
                        },
                    )
                )
                continue

            fused_decision, fused_result = self.invoke_agent(
                task=task,
                agent_id=receiver,
                visible_messages=visible_messages,
                context=context,
                ledger=ledger,
            )
            model_call_count += 1
            final_states[receiver] = fused_decision
            events.append(
                self.event(
                    task_id=task.task_id,
                    baseline=self.name,
                    event_type="agent_decision",
                    agent_id=receiver,
                    payload=self.decision_payload(
                        decision=fused_decision,
                        result=fused_result,
                        visible_messages=visible_messages,
                        phase="fusion",
                        local_state_snapshot=self.local_state_snapshot(fused_decision),
                        peer_context_quota=peer_context_quota,
                        incoming_peer_tokens_raw=incoming_peer_tokens_raw,
                        incoming_peer_tokens_used=incoming_peer_tokens_used,
                        dropped_peer_tokens_raw=dropped_peer_tokens_raw,
                    ),
                )
            )

            for sender, serialized, sender_state in child_message_payloads:
                if not serialized.used_content:
                    continue
                message_id = f"{task.task_id}:{sender}->{receiver}"
                message_trace = self.message_trace(
                    message_id=message_id,
                    task_id=task.task_id,
                    sender=sender,
                    receiver=receiver,
                    serialized_message=serialized,
                    peer_context_quota=peer_context_quota,
                    sender_state=sender_state,
                    receiver_before=receiver_before,
                    receiver_after=fused_decision,
                )
                messages.append(message_trace)
                events.append(
                    self.event(
                        task_id=task.task_id,
                        baseline=self.name,
                        event_type="message_sent",
                        agent_id=sender,
                        target_agent_id=receiver,
                        payload={
                            "message_id": message_id,
                            "message_snapshot": {
                                "used_content": serialized.used_content,
                                "raw_content": serialized.raw_content,
                                "used_token_count": serialized.used_token_count,
                                "raw_token_count": serialized.raw_token_count,
                                "dropped_token_count": serialized.dropped_token_count,
                                "answer_snapshot": serialized.answer_snapshot,
                                "includes_confidence": serialized.includes_confidence,
                                "includes_rationale": serialized.includes_rationale,
                            },
                            "sender_local_state_snapshot": self.local_state_snapshot(sender_state),
                            "receiver_local_state_before": self.local_state_snapshot(receiver_before),
                            "receiver_local_state_after": self.local_state_snapshot(fused_decision),
                            "peer_context_quota": peer_context_quota,
                            "incoming_peer_tokens_raw": incoming_peer_tokens_raw,
                            "incoming_peer_tokens_used": incoming_peer_tokens_used,
                            "dropped_peer_tokens_raw": dropped_peer_tokens_raw,
                            "token_count": message_trace.token_count,
                        },
                    )
                )

        root_state = final_states[context.topology.root_agent]
        trace = TaskRunTrace(
            task_id=task.task_id,
            selected_answer=root_state.answer,
            selected_correct=root_state.correct,
            score=1.0 if root_state.correct else 0.0,
            budget_usage=ledger.snapshot(),
            initial_states=[initial_states[agent_id] for agent_id in context.topology.agent_ids],
            final_states=[final_states[agent_id] for agent_id in context.topology.agent_ids],
            messages=messages,
            metadata={
                "root_answer": root_state.answer,
                **self.attack_metrics(
                    task=task,
                    final_states=[final_states[agent_id] for agent_id in context.topology.agent_ids],
                    selected_answer=root_state.answer,
                    attack_injector=attack_injector,
                ),
            },
        )
        self.finalize_trace_budget(trace=trace, model_call_count=model_call_count)
        events.append(
            self.event(
                task_id=task.task_id,
                baseline=self.name,
                event_type="final_aggregate",
                agent_id=context.topology.root_agent,
                payload={"selected_answer": root_state.answer, "confidence": root_state.confidence},
            )
        )
        return trace, events


def _compute_depths(parent_map: dict[str, str | None], root_agent: str) -> dict[str, int]:
    depths = {root_agent: 0}
    pending = True
    while pending:
        pending = False
        for node, parent in parent_map.items():
            if node in depths or parent is None:
                continue
            if parent in depths:
                depths[node] = depths[parent] + 1
                pending = True
    return depths
