from __future__ import annotations

import hashlib
import random
from typing import Any

from coord_harness.config.models import ModelSpec
from coord_harness.core.budget import estimate_text_tokens
from coord_harness.core.protocols import render_decision_prompt
from coord_harness.core.types import BenchmarkTask, GenerationResult
from coord_harness.models.base import ModelClient


class MockModelClient(ModelClient):
    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    def generate_decision(
        self,
        *,
        task: BenchmarkTask,
        agent_id: str,
        visible_messages: list[str],
        enable_reasoning: bool,
        seed: int,
    ) -> GenerationResult:
        agent_index = int(agent_id.split("_")[-1])
        digest_source = "|".join(
            [
                task.task_id,
                agent_id,
                str(seed),
                task.gold_answer,
                ",".join(sorted(visible_messages)),
            ]
        )
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:12], 16))

        difficulty_score = float(task.metadata.get("difficulty_score", 0.5))
        family_bias = {"craft_mini": 0.05, "agentsnet_mini": -0.04}.get(task.family.value, 0.0)
        model_quality = _model_quality(self.model_spec.model_id)
        agent_bias = (agent_index % 3) * 0.035 - 0.02

        correct_support = sum(1 for message in visible_messages if f"ANSWER={task.gold_answer}" in message)
        incorrect_support = max(
            (
                sum(1 for message in visible_messages if f"ANSWER={candidate}" in message)
                for candidate in task.answer_choices
                if candidate != task.gold_answer
            ),
            default=0,
        )
        rich_messages = sum(1 for message in visible_messages if "RATIONALE=" in message)
        base_accuracy = model_quality + family_bias + agent_bias - difficulty_score * 0.28
        support_bonus = min(correct_support * 0.12 + rich_messages * 0.05, 0.35)
        support_penalty = min(incorrect_support * 0.11, 0.33)
        reasoning_bonus = 0.05 if enable_reasoning else 0.0
        effective_accuracy = max(0.05, min(0.95, base_accuracy + support_bonus + reasoning_bonus - support_penalty))

        if rng.random() < effective_accuracy:
            answer = task.gold_answer
            confidence = min(0.95, 0.53 + support_bonus + model_quality / 3 + (agent_index % 2) * 0.05)
        else:
            distractors = [choice for choice in task.answer_choices if choice != task.gold_answer]
            answer = distractors[rng.randrange(len(distractors))]
            confidence = max(0.25, 0.52 - support_bonus + support_penalty / 2)

        rationale = (
            f"{task.metadata.get('domain', 'general')} signal favors option {answer} "
            f"after {len(visible_messages)} visible messages."
        )
        raw_text = (
            f"ANSWER: {answer}\n"
            f"CONFIDENCE: {confidence:.2f}\n"
            f"RATIONALE: {rationale}"
        )
        if enable_reasoning:
            raw_text = (
                "<think>\n"
                f"Compare local evidence with {len(visible_messages)} peer messages, then keep only evidence-consistent options.\n"
                "</think>\n"
                f"{raw_text}"
            )
        prompt_tokens = estimate_text_tokens(
            render_decision_prompt(
                task=task,
                agent_id=agent_id,
                visible_messages=visible_messages,
                enable_reasoning=enable_reasoning,
            )
        )
        completion_tokens = estimate_text_tokens(raw_text)
        return GenerationResult(
            answer=answer,
            confidence=round(confidence, 2),
            rationale=rationale,
            raw_text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            route_metadata={
                "requested_model_id": self.model_spec.model_id,
                "resolved_model_id": self.model_spec.model_id,
                "resolved_provider_name": "mock",
                "resolved_provider_tag": "mock",
                "requested_provider_policy": self.model_spec.routing.model_dump(mode="json"),
                "fallback_used": False,
                "fallback_inference_basis": "single_mock_provider",
                "pricing_snapshot": None,
            },
        )

    def describe_runtime_metadata(self) -> dict[str, Any]:
        return {
            "requested_model_id": self.model_spec.model_id,
            "resolved_model_id": self.model_spec.model_id,
            "resolved_provider_name": "mock",
            "resolved_provider_tag": "mock",
            "requested_provider_policy": self.model_spec.routing.model_dump(mode="json"),
            "fallback_used": False,
            "fallback_inference_basis": "single_mock_provider",
            "pricing_snapshot": None,
            "catalog_snapshot_hash": None,
            "catalog_snapshot_fetched_at": None,
            "endpoints_snapshot_hash": None,
            "endpoints_snapshot_fetched_at": None,
        }


def _model_quality(model_id: str) -> float:
    normalized = model_id.lower()
    if "qwen3.5-plus" in normalized:
        return 0.67
    if "minimax-m2.7" in normalized:
        return 0.64
    if "gemma-4-31b-it" in normalized:
        return 0.66
    if "gemini-3.1-flash-lite" in normalized:
        return 0.6
    if "glm-5v-turbo" in normalized:
        return 0.68
    if "glm-5" in normalized:
        return 0.57
    return 0.54
