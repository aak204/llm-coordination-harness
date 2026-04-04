from __future__ import annotations

from dataclasses import replace

from coord_harness.baselines.base import BaselineExecutor, StrategyContext
from coord_harness.core.protocols import build_sa_star_fusion_prompt
from coord_harness.core.budget import BudgetLedger
from coord_harness.core.types import BenchmarkTask, TaskRunTrace


class SAStarExecutor(BaselineExecutor):
    name = "sa_star"

    def run_task(self, *, task: BenchmarkTask, context: StrategyContext) -> tuple[TaskRunTrace, list[dict]]:
        ledger = BudgetLedger(context.trial.total_billed_token_budget)
        root_agent = context.topology.root_agent
        fused_task = replace(
            task,
            prompt=build_sa_star_fusion_prompt(task=task, agent_ids=context.topology.agent_ids),
            metadata={**task.metadata, "agent_observations": {}},
        )
        decision, result = self.invoke_agent(
            task=fused_task,
            agent_id=root_agent,
            visible_messages=[],
            context=context,
            ledger=ledger,
        )
        events = [
            self.event(
                task_id=task.task_id,
                baseline=self.name,
                event_type="agent_decision",
                agent_id=root_agent,
                payload=self.decision_payload(
                    decision=decision,
                    result=result,
                    visible_messages=[],
                    phase="single",
                    local_state_snapshot=self.local_state_snapshot(decision),
                ),
            ),
        ]
        trace = TaskRunTrace(
            task_id=task.task_id,
            selected_answer=decision.answer,
            selected_correct=decision.correct,
            score=1.0 if decision.correct else 0.0,
            budget_usage=ledger.snapshot(),
            initial_states=[decision],
            final_states=[decision],
            messages=[],
            metadata={"vote_breakdown": [{"answer": decision.answer, "vote_count": 1, "confidence_sum": decision.confidence}]},
        )
        self.finalize_trace_budget(trace=trace, model_call_count=1)
        return trace, events
