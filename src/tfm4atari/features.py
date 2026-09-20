"""RAM features and teacher-Q success labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

RAM_SIZE = 128
MAX_ALE_ACTIONS = 18
RAM_COLUMNS = tuple(f"ram_{index:03d}" for index in range(RAM_SIZE))
DELTA_COLUMNS = tuple(f"delta_{index:03d}" for index in range(RAM_SIZE))
ACTION_META_COLUMNS = (
    "previous_action",
    "candidate_action",
    "previous_reward",
    "lives",
    "episode_progress",
    "first_step",
)
ACTION_FEATURE_COLUMNS = RAM_COLUMNS + DELTA_COLUMNS + ACTION_META_COLUMNS
ACTION_CATEGORICAL_COLUMNS = ("previous_action", "candidate_action", "first_step")


class RamFeatureExtractor(Protocol):
    name: str
    version: int

    def action_features(
        self,
        ram: np.ndarray,
        previous_ram: np.ndarray | None,
        *,
        previous_action: int,
        candidate_action: int,
        previous_reward: float,
        lives: int,
        episode_progress: float,
    ) -> dict[str, float | int]: ...


@dataclass(frozen=True)
class DefaultRamFeatureExtractor:
    name: str = "raw_ram_delta"
    version: int = 1

    def action_features(
        self,
        ram: np.ndarray,
        previous_ram: np.ndarray | None,
        *,
        previous_action: int,
        candidate_action: int,
        previous_reward: float,
        lives: int,
        episode_progress: float,
    ) -> dict[str, float | int]:
        current = np.asarray(ram, dtype=np.uint8)
        if current.shape != (RAM_SIZE,):
            raise ValueError(f"Expected {RAM_SIZE} RAM bytes, got {current.shape}")
        first_step = previous_ram is None
        delta = (
            np.zeros(RAM_SIZE, dtype=np.int16)
            if first_step
            else current.astype(np.int16) - np.asarray(previous_ram, dtype=np.int16)
        )
        values: dict[str, float | int] = {
            **dict(zip(RAM_COLUMNS, current.tolist(), strict=True)),
            **dict(zip(DELTA_COLUMNS, delta.tolist(), strict=True)),
            "previous_action": int(previous_action),
            "candidate_action": int(candidate_action),
            "previous_reward": float(previous_reward),
            "lives": int(lives),
            "episode_progress": float(episode_progress),
            "first_step": int(first_step),
        }
        return values


def q_success_labels(
    q_values: np.ndarray, *, quantile: float, epsilon: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return robust normalized advantages and tied upper-quantile labels."""
    values = np.asarray(q_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("q_values must be a finite one-dimensional action vector")
    if float(np.ptp(values)) <= epsilon:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(mad, epsilon)
    advantages = (values - median) / scale
    cutoff = float(np.quantile(values, quantile))
    labels = values >= cutoff
    if labels.all():
        labels = values == values.max()
    return advantages, labels.astype(np.int8)


def make_action_rows(
    extractor: RamFeatureExtractor,
    ram: np.ndarray,
    previous_ram: np.ndarray | None,
    q_values: np.ndarray,
    *,
    previous_action: int,
    previous_reward: float,
    lives: int,
    episode_progress: float,
    success_quantile: float,
    q_epsilon: float,
) -> pd.DataFrame:
    labeled = q_success_labels(q_values, quantile=success_quantile, epsilon=q_epsilon)
    if labeled is None:
        return pd.DataFrame()
    advantages, labels = labeled
    rows: list[dict[str, float | int | str | bool]] = []
    for action, (q_value, advantage, label) in enumerate(
        zip(q_values, advantages, labels, strict=True)
    ):
        row = extractor.action_features(
            ram,
            previous_ram,
            previous_action=previous_action,
            candidate_action=action,
            previous_reward=previous_reward,
            lives=lives,
            episode_progress=episode_progress,
        )
        row.update(
            q_value=float(q_value),
            q_advantage=float(advantage),
            action_success=bool(label),
            label_source="teacher_q",
        )
        rows.append(row)
    return pd.DataFrame(rows)
