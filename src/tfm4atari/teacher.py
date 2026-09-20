"""Pluggable RL Zoo teacher backends."""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from contextlib import redirect_stderr
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from huggingface_hub import hf_hub_download, model_info

from tfm4atari.config import GameConfig, ProjectConfig


class TeacherPolicy(Protocol):
    def predict(self, observation: np.ndarray) -> int: ...

    def action_scores(self, observation: np.ndarray) -> np.ndarray: ...

    @property
    def identity(self) -> dict[str, Any]: ...


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fetch_teacher(config: ProjectConfig, game: GameConfig) -> Path:
    """Download a configured checkpoint and record its resolved Hub revision."""
    destination = config.path(config.paths.artifact_dir) / "teachers" / game.name
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        hf_hub_download(
            repo_id=game.teacher_repo,
            filename=game.teacher_file,
            revision=game.teacher_revision,
            local_dir=destination,
        )
    )
    info = model_info(game.teacher_repo, revision=game.teacher_revision)
    _atomic_json(
        destination / "artifact.json",
        {
            "backend": game.teacher_backend,
            "repo_id": game.teacher_repo,
            "filename": game.teacher_file,
            "requested_revision": game.teacher_revision,
            "resolved_revision": info.sha,
        },
    )
    return checkpoint


def teacher_checkpoint(config: ProjectConfig, game: GameConfig) -> Path:
    path = (
        config.path(config.paths.artifact_dir)
        / "teachers"
        / game.name
        / game.teacher_file
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint is missing for {game.name}; run fetch-teachers"
        )
    return path


def _legacy_custom_objects() -> dict[str, Any]:
    return {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "exploration_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        "optimize_memory_usage": False,
    }


@dataclass
class RLZooTeacher:
    model: Any
    checkpoint: Path
    backend: str

    def predict(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())

    @property
    def identity(self) -> dict[str, Any]:
        result = {
            "type": f"rl_zoo_{self.backend}",
            "backend": self.backend,
            "checkpoint": str(self.checkpoint),
            "sb3_version": importlib.metadata.version("stable-baselines3"),
            "action_score_kind": {
                "dqn": "q_value",
                "qrdqn": "mean_quantile_value",
                "ppo": "policy_probability",
            }[self.backend],
        }
        if self.backend == "qrdqn":
            result["sb3_contrib_version"] = importlib.metadata.version("sb3-contrib")
        return result


@dataclass
class RLZooDQNTeacher(RLZooTeacher):
    @classmethod
    def load(cls, checkpoint: Path, *, device: str = "auto") -> RLZooDQNTeacher:
        with redirect_stderr(StringIO()):
            from stable_baselines3 import DQN

        model = DQN.load(
            checkpoint, device=device, custom_objects=_legacy_custom_objects()
        )
        return cls(model=model, checkpoint=checkpoint, backend="dqn")

    def action_scores(self, observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            scores = self.model.q_net(tensor)
        return scores.detach().cpu().numpy()[0].astype(np.float64, copy=False)


@dataclass
class RLZooQRDQNTeacher(RLZooTeacher):
    @classmethod
    def load(cls, checkpoint: Path, *, device: str = "auto") -> RLZooQRDQNTeacher:
        with redirect_stderr(StringIO()):
            from sb3_contrib import QRDQN

        model = QRDQN.load(
            checkpoint, device=device, custom_objects=_legacy_custom_objects()
        )
        return cls(model=model, checkpoint=checkpoint, backend="qrdqn")

    def action_scores(self, observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            quantiles = self.model.quantile_net(tensor)
            scores = quantiles.mean(dim=1)
        return scores.detach().cpu().numpy()[0].astype(np.float64, copy=False)


@dataclass
class RLZooPPOTeacher(RLZooTeacher):
    @classmethod
    def load(cls, checkpoint: Path, *, device: str = "auto") -> RLZooPPOTeacher:
        with redirect_stderr(StringIO()):
            from stable_baselines3 import PPO

        model = PPO.load(
            checkpoint, device=device, custom_objects=_legacy_custom_objects()
        )
        return cls(model=model, checkpoint=checkpoint, backend="ppo")

    def action_scores(self, observation: np.ndarray) -> np.ndarray:
        """Return policy probabilities as action preferences, not Q-values."""
        with torch.inference_mode():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            distribution = self.model.policy.get_distribution(tensor)
            scores = distribution.distribution.probs
        return scores.detach().cpu().numpy()[0].astype(np.float64, copy=False)


TEACHER_BACKENDS = {
    "dqn": RLZooDQNTeacher,
    "qrdqn": RLZooQRDQNTeacher,
    "ppo": RLZooPPOTeacher,
}


def load_teacher(
    config: ProjectConfig, game: GameConfig, *, device: str
) -> TeacherPolicy:
    backend = TEACHER_BACKENDS[game.teacher_backend]
    return backend.load(teacher_checkpoint(config, game), device=device)
