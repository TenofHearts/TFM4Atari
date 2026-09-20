"""Resolve configured game-specific modules."""

from __future__ import annotations

from tfm4atari.config import ProjectConfig
from tfm4atari.games.base import AllExecutedActionsFilter, GameModule
from tfm4atari.games.beamrider import (
    BeamRiderRelevanceFilter,
    BeamRiderSymbolicJudge,
)
from tfm4atari.judges import ReturnQuantileJudge


def game_module(config: ProjectConfig, game: str) -> GameModule:
    judge_name = config.trajectory_judges.games.get(
        game, config.trajectory_judges.default
    )
    if judge_name == "beamrider_symbolic":
        if game != "BeamRider":
            raise ValueError("beamrider_symbolic can only be used for BeamRider")
        judge = BeamRiderSymbolicJudge(config.trajectory_judges.beamrider_symbolic)
    elif judge_name == "return_quantile":
        judge = ReturnQuantileJudge(config.trajectory_judges.return_quantile)
    else:  # pragma: no cover - Pydantic rejects unknown names
        raise ValueError(f"Unknown trajectory judge: {judge_name}")

    relevance_filter = (
        BeamRiderRelevanceFilter(config.context_cache.beamrider)
        if game == "BeamRider"
        else AllExecutedActionsFilter()
    )
    return GameModule(judge=judge, relevance_filter=relevance_filter)
