from __future__ import annotations

import hashlib
from dataclasses import dataclass

from coord_harness.core.budget import estimate_text_tokens, truncate_to_token_limit
from coord_harness.core.types import BenchmarkTask


DECISION_PROMPT_TEMPLATE_VERSION = "decision-v2"
MESSAGE_SERIALIZER_VERSION = "message-v2"
SA_STAR_FUSION_SERIALIZER_VERSION = "sa-star-fusion-v1"

DECISION_PROMPT_TEMPLATE = """You are {agent_id} in a frozen research harness.
You must choose the single best answer option for the task.
Assume your local evidence may be incomplete and that the correct option must satisfy all compatible evidence, not just your own clue.

Local evidence visible only to you:
{local_evidence_block}

Visible peer messages:
{visible_messages_block}

Task:
{task_prompt}

Allowed answers: {choices}

Reply in exactly this format:
ANSWER: <option>
CONFIDENCE: <0.00-1.00>
RATIONALE: <one short sentence grounded in the task or peer evidence>
"""


def decision_prompt_hash() -> str:
    return hashlib.sha256(DECISION_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def message_serializer_hash() -> str:
    signature = f"{MESSAGE_SERIALIZER_VERSION}|serializer"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def sa_star_fusion_serializer_hash() -> str:
    signature = f"{SA_STAR_FUSION_SERIALIZER_VERSION}|serializer"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SerializedPeerMessage:
    raw_content: str
    raw_token_count: int
    used_content: str
    used_token_count: int
    dropped_token_count: int
    answer_snapshot: str
    includes_confidence: bool
    includes_rationale: bool


def render_decision_prompt(*, task: BenchmarkTask, agent_id: str, visible_messages: list[str]) -> str:
    visible_messages_block = "\n".join(f"- {message}" for message in visible_messages) if visible_messages else "- none"
    local_evidence = get_agent_local_evidence(task=task, agent_id=agent_id)
    local_evidence_block = local_evidence or "- none"
    return DECISION_PROMPT_TEMPLATE.format(
        agent_id=agent_id,
        local_evidence_block=local_evidence_block,
        visible_messages_block=visible_messages_block,
        task_prompt=task.prompt,
        choices=", ".join(task.answer_choices),
    )


def get_agent_local_evidence(*, task: BenchmarkTask, agent_id: str) -> str:
    agent_observations = task.metadata.get("agent_observations") or {}
    return str(agent_observations.get(agent_id, "")).strip()


def get_agent_fact_payloads(*, task: BenchmarkTask, agent_id: str) -> list[dict]:
    agent_fact_payloads = task.metadata.get("agent_fact_payloads") or {}
    payloads = agent_fact_payloads.get(agent_id, [])
    return list(payloads)


def get_critical_fact_payloads(*, task: BenchmarkTask, agent_ids: list[str]) -> list[dict]:
    payloads: list[dict] = []
    for agent_id in agent_ids:
        for payload in get_agent_fact_payloads(task=task, agent_id=agent_id):
            if payload.get("critical") is True:
                payloads.append(dict(payload))
    return payloads


def build_sa_star_fusion_prompt(*, task: BenchmarkTask, agent_ids: list[str], token_budget: int = 96) -> str:
    fused_lines = []
    for agent_id in agent_ids:
        observation = get_agent_local_evidence(task=task, agent_id=agent_id)
        if observation:
            fused_lines.append(f"{agent_id}: {observation}")
    if not fused_lines:
        return task.prompt
    fusion_block = "\n".join(fused_lines)
    fusion_block = truncate_to_token_limit(fusion_block, token_budget)
    return (
        f"{task.prompt}\n\n"
        "Fused cross-agent evidence snapshot:\n"
        f"{fusion_block}\n"
        "Use the fused evidence conservatively because it is compressed.\n"
    )


def serialize_peer_message(
    *,
    answer: str,
    confidence: float,
    rationale: str,
    message_token_budget: int,
) -> SerializedPeerMessage:
    if message_token_budget <= 32:
        raw_payload = f"ANSWER={answer};CONFIDENCE={confidence:.2f}"
        includes_rationale = False
    else:
        raw_payload = f"ANSWER={answer};CONFIDENCE={confidence:.2f};RATIONALE={rationale}"
        includes_rationale = True
    raw_token_count = estimate_text_tokens(raw_payload)
    used_content = truncate_to_token_limit(raw_payload, message_token_budget)
    used_token_count = estimate_text_tokens(used_content)
    dropped_token_count = max(raw_token_count - used_token_count, 0)
    return SerializedPeerMessage(
        raw_content=raw_payload,
        raw_token_count=raw_token_count,
        used_content=used_content,
        used_token_count=used_token_count,
        dropped_token_count=dropped_token_count,
        answer_snapshot=answer,
        includes_confidence=True,
        includes_rationale=includes_rationale and bool(used_content),
    )


def visible_message_token_count(messages: list[str]) -> int:
    return sum(estimate_text_tokens(message) for message in messages)
