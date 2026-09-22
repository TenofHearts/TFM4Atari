"""Strict project configuration loaded exclusively from ``config.toml``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the project currently uses Python 3.10
    import tomli as tomllib


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictModel):
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    video_dir: Path = Path("videos")


class RuntimeConfig(StrictModel):
    device: str = "auto"
    precision: str = "auto"
    seed: int = 42
    render_mode: Literal["none", "rgb_array", "human"] = "none"
    record_video: bool = False
    cpu_context_rows: int = Field(4000, ge=2)
    cuda_context_rows: int = Field(10000, ge=2)
    n_preprocessing_jobs: int = Field(1, ge=1)


class AtariConfig(StrictModel):
    frameskip: int = Field(1, ge=1)
    repeat_action_probability: float = Field(0.0, ge=0.0, le=1.0)
    wrapper_frame_skip: int = Field(4, ge=1)
    noop_max: int = Field(30, ge=0)
    screen_size: int = Field(84, ge=32)
    terminal_on_life_loss: bool = False
    full_action_space: bool = False


class FeaturesConfig(StrictModel):
    # This is a feature scale, not the current collection or playback limit.
    episode_progress_reference_decisions: int = Field(27000, ge=1)


class CollectionConfig(StrictModel):
    episodes_per_game: int = Field(2, ge=2)
    sample_stride: int = Field(1, ge=1)
    max_decisions_per_episode: int = Field(27000, ge=1)
    maximum_attempts_per_game: int = Field(6, ge=2)


class LearningConfig(StrictModel):
    episodes: int = Field(100, ge=1)
    policy_mode: Literal[
        "behavior_cloning",
        "outcome_conditioned",
        "outcome_prediction_categorical",
        "outcome_prediction_regression",
    ] = (
        "outcome_prediction_categorical"
    )
    desired_symbolic_label: Literal[-1, 0, 1] = 1
    outcome_score_discount: float = Field(0.90, gt=0.0, le=1.0)
    outcome_sampling_temperature: float = Field(0.25, gt=0.0)
    action_selection: Literal[
        "greedy", "probability_sample", "epsilon_sample"
    ] = "probability_sample"
    epsilon_sample_probability: float = Field(0.30, ge=0.0, le=1.0)


class ReturnQuantileJudgeConfig(StrictModel):
    success_quantile: float = Field(0.50, gt=0.0, lt=1.0)


class BeamRiderSymbolicJudgeConfig(StrictModel):
    minimum_score: float = Field(44.0, ge=0.0)
    minimum_positive_reward_events: int = Field(1, ge=1)
    minimum_survival_fraction: float = Field(0.80, gt=0.0, le=1.0)
    maximum_lives_lost: int = Field(1, ge=0)


class TrajectoryJudgesConfig(StrictModel):
    default: Literal["return_quantile", "beamrider_symbolic"] = "return_quantile"
    games: dict[str, Literal["return_quantile", "beamrider_symbolic"]] = Field(
        default_factory=dict
    )
    return_quantile: ReturnQuantileJudgeConfig
    beamrider_symbolic: BeamRiderSymbolicJudgeConfig


class BeamRiderCacheConfig(StrictModel):
    """BeamRider keeps every action; reserved for future game-specific settings."""


class ContextCacheConfig(StrictModel):
    enabled: bool = True
    teacher_judgment_capacity: int = Field(8, ge=1)
    judgment_capacity: int = Field(64, ge=1)
    refit_every_judged_batches: int = Field(4, ge=1)
    maximum_rows: int = Field(4000, ge=2)
    minimum_cold_start_rows: int = Field(2000, ge=2)
    beamrider: BeamRiderCacheConfig


class EvaluationConfig(StrictModel):
    random_episodes: int = Field(10, ge=1)
    teacher_episodes: int = Field(10, ge=1)
    final_episodes: int = Field(30, ge=1)
    early_window: int = Field(20, ge=1)
    late_window: int = Field(20, ge=1)


class VideoConfig(StrictModel):
    seed: int = 9001
    max_decisions: int = Field(256, ge=1)
    fps: int = Field(30, ge=1)
    name_prefix: str = "beamrider-tabpfn"
    persist_context_additions: bool = True
    learning_episodes: int = Field(10, ge=1)
    retain_learning_episode_videos: bool = True
    teacher_max_decisions: int = Field(27000, ge=1)
    teacher_name_prefix: str = "beamrider-teacher"


class TabPFNConfig(StrictModel):
    model_path: str = "auto"
    cache_dir: Path = Path("artifacts/tabpfn")
    n_estimators: int = Field(4, ge=1)
    fit_mode: Literal[
        "low_memory", "fit_preprocessors", "fit_with_cache", "batched"
    ] = "fit_with_cache"
    balance_probabilities: bool = True


class GameConfig(StrictModel):
    name: str
    env_id: str
    teacher_backend: Literal["dqn", "qrdqn", "ppo"]
    teacher_repo: str
    teacher_file: str
    teacher_revision: str


class ProjectConfig(StrictModel):
    schema_version: int = Field(1, ge=1)
    enabled_games: tuple[str, ...]
    paths: PathsConfig
    runtime: RuntimeConfig
    atari: AtariConfig
    features: FeaturesConfig
    collection: CollectionConfig
    learning: LearningConfig
    trajectory_judges: TrajectoryJudgesConfig
    context_cache: ContextCacheConfig
    evaluation: EvaluationConfig
    video: VideoConfig
    tabpfn: TabPFNConfig
    games: tuple[GameConfig, ...]
    root: Path = Field(default=Path.cwd(), exclude=True)

    @model_validator(mode="after")
    def validate_games(self) -> ProjectConfig:
        names = [game.name for game in self.games]
        if not names or len(names) != len(set(names)):
            raise ValueError("games must be non-empty and have unique names")
        unknown = set(self.enabled_games) - set(names)
        if unknown:
            raise ValueError(f"enabled_games contains unknown games: {sorted(unknown)}")
        if not self.enabled_games:
            raise ValueError("enabled_games must not be empty")
        smallest_budget = min(
            self.runtime.cpu_context_rows, self.runtime.cuda_context_rows
        )
        if self.context_cache.minimum_cold_start_rows >= smallest_budget:
            raise ValueError(
                "context_cache.minimum_cold_start_rows must be smaller than "
                "every hardware context budget"
            )
        return self

    @property
    def active_games(self) -> tuple[GameConfig, ...]:
        enabled = set(self.enabled_games)
        return tuple(game for game in self.games if game.name in enabled)

    def path(self, value: Path) -> Path:
        return value if value.is_absolute() else self.root / value


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load the fixed project configuration, rejecting unknown settings."""
    config_path = (path or Path.cwd() / "config.toml").resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    raw["root"] = config_path.parent
    return ProjectConfig.model_validate(raw)
