from __future__ import annotations

from dataclasses import dataclass

from coord_harness.core.enums import AttackInjectionDepth, AttackPayloadType


@dataclass(frozen=True)
class ResolvedAttackSpec:
    enabled: bool
    injection_depth: AttackInjectionDepth
    payload_type: AttackPayloadType
    target_answer_policy: str
    compromise_confidence: float
    propagate_instruction: bool
    only_baselines: tuple[str, ...]
    clean_reference_experiment_dir: str | None = None
    legacy_attack_mode: str | None = None
    legacy_attacker_policy: str | None = None


@dataclass(frozen=True)
class AttackTarget:
    agent_id: str
    injection_depth: AttackInjectionDepth
    topology_depth: int
    path_to_root: tuple[str, ...]
    child_count: int

