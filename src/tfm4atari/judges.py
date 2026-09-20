"""Pluggable and auditable trajectory-success judges."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from tfm4atari.config import ReturnQuantileJudgeConfig


@dataclass(frozen=True)
class TrajectoryOutcome:
    game: str
    episode_return: float
    decisions: int
    decision_limit: int
    initial_lives: int
    final_lives: int
    positive_reward_events: int
    terminated: bool
    truncated: bool

    @property
    def lives_lost(self) -> int:
        return max(self.initial_lives - self.final_lives, 0)

    @property
    def survival_fraction(self) -> float:
        return min(self.decisions / max(self.decision_limit, 1), 1.0)

    def as_trial_columns(self) -> dict[str, Any]:
        return {
            "played_episode_return": self.episode_return,
            "played_episode_length": self.decisions,
            "played_decision_limit": self.decision_limit,
            "played_initial_lives": self.initial_lives,
            "played_final_lives": self.final_lives,
            "played_positive_reward_events": self.positive_reward_events,
            "played_terminated": self.terminated,
            "played_truncated": self.truncated,
        }

    @classmethod
    def from_trial(cls, game: str, row: Mapping[str, Any]) -> TrajectoryOutcome:
        return cls(
            game=game,
            episode_return=float(row["played_episode_return"]),
            decisions=int(row["played_episode_length"]),
            decision_limit=int(row["played_decision_limit"]),
            initial_lives=int(row["played_initial_lives"]),
            final_lives=int(row["played_final_lives"]),
            positive_reward_events=int(row["played_positive_reward_events"]),
            terminated=bool(row["played_terminated"]),
            truncated=bool(row["played_truncated"]),
        )


@dataclass(frozen=True)
class Judgment:
    judge: str
    version: int
    score: float
    success: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | bool]

    def as_trial_columns(self) -> dict[str, Any]:
        return {
            "judge_name": self.judge,
            "judge_version": self.version,
            "trajectory_score": self.score,
            "trajectory_success": self.success,
            "judge_reasons": json.dumps(self.reasons),
            "judge_metrics": json.dumps(self.metrics, sort_keys=True),
        }


class TrajectoryJudge(Protocol):
    name: str
    version: int

    def calibrate(self, outcomes: list[TrajectoryOutcome]) -> dict[str, Any]: ...

    def judge(
        self, outcome: TrajectoryOutcome, state: Mapping[str, Any]
    ) -> Judgment: ...


@dataclass(frozen=True)
class ReturnQuantileJudge:
    config: ReturnQuantileJudgeConfig
    name: str = "return_quantile"
    version: int = 1

    def calibrate(self, outcomes: list[TrajectoryOutcome]) -> dict[str, Any]:
        if not outcomes:
            raise ValueError("ReturnQuantileJudge requires calibration outcomes")
        threshold = float(
            np.quantile(
                [outcome.episode_return for outcome in outcomes],
                self.config.success_quantile,
            )
        )
        return {"threshold": threshold, "quantile": self.config.success_quantile}

    def judge(self, outcome: TrajectoryOutcome, state: Mapping[str, Any]) -> Judgment:
        threshold = float(state["threshold"])
        success = outcome.episode_return >= threshold
        comparison = ">=" if success else "<"
        reason = (
            f"return {outcome.episode_return:g} {comparison} threshold {threshold:g}"
        )
        return Judgment(
            judge=self.name,
            version=self.version,
            score=outcome.episode_return - threshold,
            success=success,
            reasons=(reason,),
            metrics={
                "episode_return": outcome.episode_return,
                "threshold": threshold,
            },
        )


def label_trials(
    trials: pd.DataFrame,
    *,
    game: str,
    judge: TrajectoryJudge,
    state: Mapping[str, Any],
) -> pd.DataFrame:
    labeled = trials.copy()
    judgments = [
        judge.judge(TrajectoryOutcome.from_trial(game, row), state)
        for row in labeled.to_dict(orient="records")
    ]
    columns = [judgment.as_trial_columns() for judgment in judgments]
    for column in (
        "judge_name",
        "judge_version",
        "trajectory_score",
        "trajectory_success",
        "judge_reasons",
        "judge_metrics",
    ):
        labeled[column] = [values[column] for values in columns]
    return labeled


def judge_state_document(
    judge: TrajectoryJudge, state: Mapping[str, Any]
) -> dict[str, Any]:
    return {"judge": judge.name, "version": judge.version, "state": dict(state)}
