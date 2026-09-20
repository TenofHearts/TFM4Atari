"""TabPFN action policy and hardware-aware context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from tfm4atari.config import ProjectConfig
from tfm4atari.features import (
    ACTION_CATEGORICAL_COLUMNS,
    ACTION_FEATURE_COLUMNS,
    DefaultRamFeatureExtractor,
)


class Classifier(Protocol):
    classes_: np.ndarray

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


def resolved_device(config: ProjectConfig) -> str:
    if config.runtime.device != "auto":
        return config.runtime.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def context_budget(config: ProjectConfig) -> int:
    return (
        config.runtime.cuda_context_rows
        if resolved_device(config).startswith("cuda")
        else config.runtime.cpu_context_rows
    )


@dataclass(frozen=True)
class TabPFNFactory:
    config: ProjectConfig

    def classifier(self, categorical_columns: tuple[str, ...] = ()) -> Classifier:
        from tabpfn import TabPFNClassifier
        from tabpfn.settings import settings

        cache_dir = self.config.path(self.config.tabpfn.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings.tabpfn.model_cache_dir = cache_dir
        categorical_indices = [
            ACTION_FEATURE_COLUMNS.index(name) for name in categorical_columns
        ]
        precision: Any = self.config.runtime.precision
        if precision not in ("auto", "autocast"):
            try:
                precision = getattr(torch, precision)
            except AttributeError as error:
                raise ValueError(f"Unknown torch precision: {precision}") from error
        return TabPFNClassifier(
            model_path=self.config.tabpfn.model_path,
            device=resolved_device(self.config),
            n_estimators=self.config.tabpfn.n_estimators,
            fit_mode=self.config.tabpfn.fit_mode,
            inference_precision=precision,
            balance_probabilities=self.config.tabpfn.balance_probabilities,
            categorical_features_indices=categorical_indices or None,
            n_preprocessing_jobs=self.config.runtime.n_preprocessing_jobs,
            random_state=self.config.runtime.seed,
        )


def stratified_context(rows: pd.DataFrame, budget: int, *, seed: int) -> pd.DataFrame:
    """Deterministically preserve action/label coverage under the context cap."""
    ordered = rows.sort_values(["step", "candidate_action"]).reset_index(drop=True)
    if len(ordered) <= budget:
        return ordered
    group_columns = ["candidate_action", "action_success"]
    if "trajectory_id" in ordered:
        group_columns.insert(0, "trajectory_id")
    groups = list(ordered.groupby(group_columns, sort=True))
    allocation = max(1, budget // len(groups))
    selected = [
        group.sample(min(len(group), allocation), random_state=seed)
        for _, group in groups
    ]
    context = pd.concat(selected).drop_duplicates()
    remaining = budget - len(context)
    if remaining > 0:
        pool = ordered.drop(index=context.index, errors="ignore")
        if not pool.empty:
            context = pd.concat(
                [context, pool.sample(min(remaining, len(pool)), random_state=seed)]
            )
    return context.head(budget).sort_values(["step", "candidate_action"])


@dataclass
class Actor:
    classifier: Classifier
    extractor: DefaultRamFeatureExtractor

    @classmethod
    def fit(
        cls,
        factory: TabPFNFactory,
        rows: pd.DataFrame,
        *,
        budget: int,
        seed: int,
        additions: pd.DataFrame | None = None,
        minimum_base_rows: int = 0,
    ) -> Actor:
        additions = pd.DataFrame() if additions is None else additions
        if additions.empty:
            context = stratified_context(rows, budget, seed=seed)
        else:
            addition_budget = max(budget - minimum_base_rows, 0)
            retained_additions = additions.tail(addition_budget)
            base_budget = budget - len(retained_additions)
            if base_budget < 2:
                raise ValueError("Context budget leaves no room for cold start")
            retained_base = stratified_context(rows, base_budget, seed=seed)
            context = pd.concat(
                [retained_base, retained_additions], ignore_index=True, sort=False
            )
        if context["action_success"].nunique() < 2:
            raise ValueError("Actor context must contain successful and failed actions")
        classifier = factory.classifier(ACTION_CATEGORICAL_COLUMNS)
        classifier.fit(context[list(ACTION_FEATURE_COLUMNS)], context["action_success"])
        return cls(classifier=classifier, extractor=DefaultRamFeatureExtractor())

    def choose_action(
        self,
        ram: np.ndarray,
        previous_ram: np.ndarray | None,
        *,
        action_count: int,
        previous_action: int,
        previous_reward: float,
        lives: int,
        episode_progress: float,
    ) -> int:
        rows = [
            self.extractor.action_features(
                ram,
                previous_ram,
                previous_action=previous_action,
                candidate_action=action,
                previous_reward=previous_reward,
                lives=lives,
                episode_progress=episode_progress,
            )
            for action in range(action_count)
        ]
        probabilities = self.classifier.predict_proba(
            pd.DataFrame(rows)[list(ACTION_FEATURE_COLUMNS)]
        )
        classes = list(self.classifier.classes_)
        positive_index = classes.index(True) if True in classes else classes.index(1)
        return int(np.argmax(probabilities[:, positive_index]))
