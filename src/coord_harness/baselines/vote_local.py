from __future__ import annotations

from coord_harness.baselines.base import BaselineExecutor, StrategyContext, vote_from_decisions
from coord_harness.core.budget import BudgetLedger
from coord_harness.core.types import BenchmarkTask, TaskRunTrace


class VoteLocalExecutor(BaselineExecutor):
    name = "vote_local"

    def run_task(self, *, task: BenchmarkTask, context: StrategyContext) -> tuple[TaskRunTrace, list[dict]]:
        ledger = BudgetLedger(context.trial.total_billed_token_budget)
        initial_states = []
        events: list[dict] = []
        route_metadata = []
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
                context=context,
            )
            initial_states.append(decision)
            route_metadata.append(result.route_metadata)
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
        selected_answer, vote_breakdown = vote_from_decisions(initial_states)
        trace = TaskRunTrace(
            task_id=task.task_id,
            selected_answer=selected_answer,
            selected_correct=selected_answer == task.gold_answer,
            score=1.0 if selected_answer == task.gold_answer else 0.0,
            budget_usage=ledger.snapshot(),
            initial_states=initial_states,
            final_states=list(initial_states),
            messages=[],
            metadata={
                "vote_breakdown": vote_breakdown,
                "route_metadata": route_metadata,
                **self.attack_metrics(
                    task=task,
                    final_states=list(initial_states),
                    selected_answer=selected_answer,
                    context=context,
                ),
            },
        )
        self.finalize_trace_budget(trace=trace, model_call_count=len(context.topology.agent_ids))
        events.append(
            self.event(
                task_id=task.task_id,
                baseline=self.name,
                event_type="vote_aggregate",
                payload={"selected_answer": selected_answer, "vote_breakdown": vote_breakdown},
            )
        )
        return trace, events
