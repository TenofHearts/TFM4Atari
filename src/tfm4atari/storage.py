"""Immutable, atomic Parquet storage for teacher episodes and learning trials."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from tfm4atari.config import ProjectConfig
from tfm4atari.features import SYMBOLIC_LABEL_SCHEMA


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".parquet")
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class DataStore:
    def __init__(self, config: ProjectConfig) -> None:
        self.root = config.path(config.paths.data_dir)
        self.schema_version = config.schema_version

    def _game(self, game: str) -> Path:
        return self.root / game

    def episode_path(self, game: str, trajectory_id: str) -> Path:
        return self._game(game) / "episodes" / f"{trajectory_id}.parquet"

    def write_episode(
        self,
        game: str,
        trajectory_id: str,
        rows: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> None:
        target = self.episode_path(game, trajectory_id)
        if target.exists():
            return
        atomic_parquet(target, rows)
        atomic_json(
            target.with_suffix(".json"),
            {"schema_version": self.schema_version, **metadata},
        )

    def episode_ids(self, game: str) -> tuple[str, ...]:
        directory = self._game(game) / "episodes"
        if not directory.exists():
            return ()
        return tuple(
            sorted(
                path.stem
                for path in directory.glob("*.parquet")
                if path.with_suffix(".json").is_file()
            )
        )

    def cold_start_episode_ids(
        self, game: str, teacher_backend: str
    ) -> tuple[str, ...]:
        """Return only naturally completed, explicitly cold-start episodes."""
        selected: list[str] = []
        for trajectory_id in self.episode_ids(game):
            metadata = self.episode_metadata(game, trajectory_id)
            if (
                metadata.get("collection_role") == "cold_start_teacher"
                and bool(metadata.get("complete_episode"))
                and metadata.get("teacher", {}).get("backend") == teacher_backend
                and metadata.get("context_label_schema") == SYMBOLIC_LABEL_SCHEMA
            ):
                selected.append(trajectory_id)
        return tuple(selected)

    def read_episode(self, game: str, trajectory_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.episode_path(game, trajectory_id))

    def episode_metadata(self, game: str, trajectory_id: str) -> dict[str, Any]:
        with (
            self.episode_path(game, trajectory_id)
            .with_suffix(".json")
            .open(encoding="utf-8") as handle
        ):
            return json.load(handle)

    def trial_dir(
        self, game: str, teacher_backend: str, phase: str = "learning"
    ) -> Path:
        return (
            self._game(game)
            / "learning_trials"
            / teacher_backend
            / SYMBOLIC_LABEL_SCHEMA
            / phase
        )

    def write_trial(
        self,
        game: str,
        teacher_backend: str,
        trial_id: str,
        row: dict[str, Any],
        *,
        phase: str,
    ) -> None:
        atomic_parquet(
            self.trial_dir(game, teacher_backend, phase) / f"{trial_id}.parquet",
            pd.DataFrame([row]),
        )

    def read_trials(
        self,
        game: str,
        teacher_backend: str,
        phases: Iterable[str] = ("learning",),
    ) -> pd.DataFrame:
        paths = [
            path
            for phase in phases
            for path in self.trial_dir(game, teacher_backend, phase).glob("*.parquet")
        ]
        return (
            pd.concat(
                (pd.read_parquet(path) for path in sorted(paths)), ignore_index=True
            )
            if paths
            else pd.DataFrame()
        )

    def judge_state_path(self, game: str, teacher_backend: str) -> Path:
        return self._game(game) / (
            f"trajectory_judge_{teacher_backend}_{SYMBOLIC_LABEL_SCHEMA}.json"
        )

    def write_judge_state(
        self, game: str, teacher_backend: str, document: dict[str, Any]
    ) -> None:
        atomic_json(
            self.judge_state_path(game, teacher_backend),
            {"schema_version": self.schema_version, **document},
        )

    def read_judge_state(self, game: str, teacher_backend: str) -> dict[str, Any]:
        with self.judge_state_path(game, teacher_backend).open(
            encoding="utf-8"
        ) as handle:
            return json.load(handle)

    def context_cache_path(self, game: str, teacher_backend: str) -> Path:
        return self._game(game) / (
            f"context_cache_{teacher_backend}_{SYMBOLIC_LABEL_SCHEMA}.parquet"
        )

    def read_context_cache(self, game: str, teacher_backend: str) -> pd.DataFrame:
        path = self.context_cache_path(game, teacher_backend)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def write_context_cache(
        self, game: str, teacher_backend: str, rows: pd.DataFrame
    ) -> None:
        atomic_parquet(self.context_cache_path(game, teacher_backend), rows)
