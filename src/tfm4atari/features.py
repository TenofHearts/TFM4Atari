"""State features for symbolic-outcome-conditioned action imitation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

RAM_SIZE = 128
RAM_COLUMNS = tuple(f"ram_{index:03d}" for index in range(RAM_SIZE))
DELTA_COLUMNS = tuple(f"delta_{index:03d}" for index in range(RAM_SIZE))
STATE_META_COLUMNS = (
    "previous_action",
    "previous_reward",
    "lives",
    "episode_progress",
    "first_step",
)
DESIRED_SYMBOLIC_LABEL = "desired_symbolic_label"
ACTION_FEATURE_COLUMNS = (
    RAM_COLUMNS + DELTA_COLUMNS + STATE_META_COLUMNS + (DESIRED_SYMBOLIC_LABEL,)
)
ACTION_CATEGORICAL_COLUMNS = (
    "previous_action",
    "first_step",
    DESIRED_SYMBOLIC_LABEL,
)
ACTION_TARGET_COLUMN = "executed_action"
SYMBOLIC_LABEL_SCHEMA = "symbolic_teacher_window16_online_window8_v5"


class RamFeatureExtractor(Protocol):
    name: str
    version: int

    def state_features(
        self,
        ram: np.ndarray,
        previous_ram: np.ndarray | None,
        *,
        previous_action: int,
        previous_reward: float,
        lives: int,
        episode_progress: float,
        desired_symbolic_label: int,
    ) -> dict[str, float | int]: ...


@dataclass(frozen=True)
class DefaultRamFeatureExtractor:
    name: str = "raw_ram_delta_symbolic"
    version: int = 2

    def state_features(
        self,
        ram: np.ndarray,
        previous_ram: np.ndarray | None,
        *,
        previous_action: int,
        previous_reward: float,
        lives: int,
        episode_progress: float,
        desired_symbolic_label: int,
    ) -> dict[str, float | int]:
        if desired_symbolic_label not in (-1, 0, 1):
            raise ValueError("desired_symbolic_label must be -1, 0, or 1")
        current = np.asarray(ram, dtype=np.uint8)
        if current.shape != (RAM_SIZE,):
            raise ValueError(f"Expected {RAM_SIZE} RAM bytes, got {current.shape}")
        first_step = previous_ram is None
        delta = (
            np.zeros(RAM_SIZE, dtype=np.int16)
            if first_step
            else current.astype(np.int16) - np.asarray(previous_ram, dtype=np.int16)
        )
        return {
            **dict(zip(RAM_COLUMNS, current.tolist(), strict=True)),
            **dict(zip(DELTA_COLUMNS, delta.tolist(), strict=True)),
            "previous_action": int(previous_action),
            "previous_reward": float(previous_reward),
            "lives": int(lives),
            "episode_progress": float(episode_progress),
            "first_step": int(first_step),
            DESIRED_SYMBOLIC_LABEL: desired_symbolic_label,
        }
