from __future__ import annotations

from enum import Enum


class BaselineStrategy(str, Enum):
    SA_STAR = "sa_star"
    VOTE_LOCAL = "vote_local"
    MA_FT = "ma_ft"


class TopologyPreset(str, Enum):
    STAR = "star"
    BALANCED_TREE = "balanced_tree"
    LINEAR_CHAIN = "linear_chain"
    COMPLETE_GRAPH = "complete_graph"
    SPARSE_GRAPH = "sparse_graph"


class BenchmarkFamily(str, Enum):
    CRAFT_MINI = "craft_mini"
    AGENTSNET_MINI = "agentsnet_mini"


class ModelProvider(str, Enum):
    MOCK = "mock"
    OPENROUTER = "openrouter"


class RunMode(str, Enum):
    RESEARCH_STRICT = "research_strict"
    DEV_CONVENIENCE = "dev_convenience"


class PanelTier(str, Enum):
    SWEEP = "sweep"
    REFEREE = "referee"
    CONFIRMATORY = "confirmatory"
    DEV = "dev"


class AttackMode(str, Enum):
    COMPROMISED_LEAF = "compromised_leaf"


class AttackInjectionDepth(str, Enum):
    LEAF = "leaf"
    MIDDLE_MANAGER = "middle_manager"
    MANAGER = "manager"


class AttackPayloadType(str, Enum):
    HARD_HALLUCINATION = "hard_hallucination"
    PROMPT_OVERRIDE = "prompt_override"


class RunStage(str, Enum):
    CLEAN = "clean"
    STRESS = "stress"


class RegimeLabel(str, Enum):
    HELP = "help"
    SATURATION = "saturation"
    COLLAPSE = "collapse"
