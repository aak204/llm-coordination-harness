from __future__ import annotations

from coord_harness.baselines.base import BaselineExecutor
from coord_harness.baselines.ma_ft import MAFTExecutor
from coord_harness.baselines.sa_star import SAStarExecutor
from coord_harness.baselines.vote_local import VoteLocalExecutor
from coord_harness.core.enums import BaselineStrategy


def build_baseline_executor(strategy: BaselineStrategy) -> BaselineExecutor:
    if strategy is BaselineStrategy.SA_STAR:
        return SAStarExecutor()
    if strategy is BaselineStrategy.VOTE_LOCAL:
        return VoteLocalExecutor()
    if strategy is BaselineStrategy.MA_FT:
        return MAFTExecutor()
    raise ValueError(f"Unsupported baseline strategy: {strategy.value}")
