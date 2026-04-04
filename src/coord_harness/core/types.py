from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coord_harness.core.enums import BenchmarkFamily


@dataclass(frozen=True)
class BenchmarkTask:
    family: BenchmarkFamily
    task_id: str
    prompt: str
    answer_choices: list[str]
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    agent_id: str
    answer: str
    confidence: float
    rationale: str
    correct: bool


@dataclass
class MessageTrace:
    message_id: str
    task_id: str
    sender: str
    receiver: str
    content: str
    token_count: int
    raw_content: str
    raw_token_count: int
    dropped_token_count: int
    peer_context_quota: int
    sender_answer: str
    sender_correct: bool
    receiver_answer_before: str
    receiver_correct_before: bool
    receiver_answer_after: str
    receiver_correct_after: bool


@dataclass
class TaskRunTrace:
    task_id: str
    selected_answer: str
    selected_correct: bool
    score: float
    budget_usage: dict[str, int]
    initial_states: list[AgentDecision]
    final_states: list[AgentDecision]
    messages: list[MessageTrace]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    answer: str
    confidence: float
    rationale: str
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    route_metadata: dict[str, Any]
