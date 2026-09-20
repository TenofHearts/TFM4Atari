"""Symbolic trajectory and context rules for BeamRider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tfm4atari.config import (
    BeamRiderCacheConfig,
    BeamRiderSymbolicJudgeConfig,
)
from tfm4atari.judges import Judgment, TrajectoryOutcome


@dataclass(frozen=True)
class BeamRiderSymbolicJudge:
    """Require combat progress, survival, and preservation of lives."""

    config: BeamRiderSymbolicJudgeConfig
    name: str = "beamrider_symbolic"
    version: int = 1

    def calibrate(self, outcomes: list[TrajectoryOutcome]) -> dict[str, Any]:
        del outcomes
        return {"rules": self.config.model_dump(mode="json")}

    def judge(self, outcome: TrajectoryOutcome, state: Mapping[str, Any]) -> Judgment:
        del state
        if outcome.game != "BeamRider":
            raise ValueError("BeamRiderSymbolicJudge only accepts BeamRider")
        checks = {
            "minimum_score": outcome.episode_return >= self.config.minimum_score,
            "positive_reward_event": (
                outcome.positive_reward_events
                >= self.config.minimum_positive_reward_events
            ),
            "survival": (
                outcome.survival_fraction >= self.config.minimum_survival_fraction
            ),
            "lives_preserved": (outcome.lives_lost <= self.config.maximum_lives_lost),
        }
        success = all(checks.values())
        score = (
            outcome.episode_return
            + 44.0 * outcome.positive_reward_events
            + 100.0 * outcome.survival_fraction
            - 100.0 * outcome.lives_lost
        )
        return Judgment(
            judge=self.name,
            version=self.version,
            score=score,
            success=success,
            reasons=tuple(
                f"{name}={'pass' if passed else 'fail'}"
                for name, passed in checks.items()
            ),
            metrics={
                **checks,
                "episode_return": outcome.episode_return,
                "positive_reward_events": outcome.positive_reward_events,
                "survival_fraction": outcome.survival_fraction,
                "lives_lost": outcome.lives_lost,
            },
        )


@dataclass(frozen=True)
class BeamRiderRelevanceFilter:
    """Keep only executed actions with direct BeamRider relevance."""

    config: BeamRiderCacheConfig
    name: str = "beamrider_combat"
    version: int = 1

    def filter(self, executed_actions: pd.DataFrame) -> pd.DataFrame:
        if executed_actions.empty:
            return executed_actions.copy()
        relevant = executed_actions["candidate_action"].isin(self.config.combat_actions)
        if self.config.keep_positive_reward_actions:
            relevant |= executed_actions["observed_reward"] > 0
        if self.config.keep_life_change_actions:
            relevant |= (
                executed_actions["lives_before"] != executed_actions["lives_after"]
            )
        return executed_actions.loc[relevant].copy()
