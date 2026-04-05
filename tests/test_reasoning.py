from __future__ import annotations

from coord_harness.core.enums import BenchmarkFamily
from coord_harness.core.protocols import render_decision_prompt, serialize_peer_message
from coord_harness.core.types import BenchmarkTask


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        family=BenchmarkFamily.CRAFT_MINI,
        task_id="t1",
        prompt="Choose the best option.",
        answer_choices=["A", "B", "C", "D"],
        gold_answer="A",
        metadata={"agent_observations": {"agent_0": "Local clue."}},
    )


def test_reasoning_prompt_includes_think_block_instructions() -> None:
    prompt = render_decision_prompt(task=_task(), agent_id="agent_0", visible_messages=[], enable_reasoning=True)
    assert "<think>" in prompt
    assert "Do not use the tokens ANSWER:" in prompt


def test_non_reasoning_prompt_uses_plain_answer_format() -> None:
    prompt = render_decision_prompt(task=_task(), agent_id="agent_0", visible_messages=[], enable_reasoning=False)
    assert "<think>" not in prompt
    assert "ANSWER: <option>" in prompt


def test_peer_message_serializer_strips_think_blocks_from_public_payload() -> None:
    serialized = serialize_peer_message(
        answer="B",
        confidence=0.91,
        rationale="<think>\nprivate chain of thought\n</think>\nPublic one-line explanation.",
        message_token_budget=96,
    )
    assert "<think>" not in serialized.raw_content
    assert "private chain of thought" not in serialized.raw_content
    assert "Public one-line explanation." in serialized.raw_content
