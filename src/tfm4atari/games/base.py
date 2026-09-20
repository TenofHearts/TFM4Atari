"""Shared contracts for game-specific behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from tfm4atari.judges import TrajectoryJudge


class RelevanceFilter(Protocol):
    name: str
    version: int

    def filter(self, executed_actions: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class GameModule:
    judge: TrajectoryJudge
    relevance_filter: RelevanceFilter


@dataclass(frozen=True)
class AllExecutedActionsFilter:
    """Generic fallback that keeps every actually executed action."""

    name: str = "all_executed_actions"
    version: int = 1

    def filter(self, executed_actions: pd.DataFrame) -> pd.DataFrame:
        return executed_actions.copy()
