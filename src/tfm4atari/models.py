"""TabPFN action policy and hardware-aware context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from tfm4atari.config import ProjectConfig
from tfm4atari.features import (
    ACTION_TARGET_COLUMN,
    DESIRED_SYMBOLIC_LABEL,
    DefaultRamFeatureExtractor,
    actor_categorical_columns,
    actor_feature_columns,
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

    def classifier(
        self,
        feature_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...] = (),
    ) -> Classifier:
        from tabpfn import TabPFNClassifier
        from tabpfn.settings import settings

        cache_dir = self.config.path(self.config.tabpfn.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings.tabpfn.model_cache_dir = cache_dir
        categorical_indices = [
            feature_columns.index(name) for name in categorical_columns
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


def stratified_context(
    rows: pd.DataFrame,
    budget: int,
    *,
    seed: int,
    stratify_symbolic: bool = True,
) -> pd.DataFrame:
    """Sample context proportionally while preserving every observed stratum."""
    ordered = rows.sort_values(["step", ACTION_TARGET_COLUMN]).reset_index(drop=True)
    if len(ordered) <= budget:
        return ordered
    trajectory_columns = ["trajectory_id"] if "trajectory_id" in ordered else []
    candidates = []
    if stratify_symbolic:
        candidates.extend(
            (
                trajectory_columns
                + [DESIRED_SYMBOLIC_LABEL, ACTION_TARGET_COLUMN],
                trajectory_columns + [DESIRED_SYMBOLIC_LABEL],
            )
        )
    candidates.extend(
        (trajectory_columns + [ACTION_TARGET_COLUMN], trajectory_columns)
    )
    groups: list[tuple[Any, pd.DataFrame]] = [("all", ordered)]
    for group_columns in candidates:
        if not group_columns:
            continue
        candidate_groups = list(ordered.groupby(group_columns, sort=True))
        if len(candidate_groups) <= budget:
            groups = candidate_groups
            break

    sizes = np.asarray([len(group) for _, group in groups], dtype=np.int64)
    quotas = budget * sizes / sizes.sum()
    allocations = np.minimum(np.floor(quotas).astype(np.int64), sizes)
    allocations[allocations == 0] = 1
    difference = budget - int(allocations.sum())
    fractions = quotas - np.floor(quotas)
    if difference > 0:
        order = np.argsort(-fractions, kind="stable")
        while difference:
            for index in order:
                if allocations[index] < sizes[index]:
                    allocations[index] += 1
                    difference -= 1
                    if difference == 0:
                        break
    elif difference < 0:
        excess = allocations - quotas
        order = np.argsort(-excess, kind="stable")
        while difference:
            for index in order:
                if allocations[index] > 1:
                    allocations[index] -= 1
                    difference += 1
                    if difference == 0:
                        break

    selected = [
        group.sample(int(count), random_state=seed)
        for (_, group), count in zip(groups, allocations, strict=True)
    ]
    return pd.concat(selected).sort_values(["step", ACTION_TARGET_COLUMN])


@dataclass
class Actor:
    classifier: Classifier
    extractor: DefaultRamFeatureExtractor
    feature_columns: tuple[str, ...]
    desired_symbolic_label: int
    action_selection: str
    epsilon_sample_probability: float
    rng: np.random.Generator

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
        policy_mode = factory.config.learning.policy_mode
        stratify_symbolic = policy_mode == "outcome_conditioned"
        if additions.empty:
            context = stratified_context(
                rows,
                budget,
                seed=seed,
                stratify_symbolic=stratify_symbolic,
            )
        else:
            addition_budget = max(budget - minimum_base_rows, 0)
            retained_additions = additions.tail(addition_budget)
            base_budget = budget - len(retained_additions)
            if base_budget < 2:
                raise ValueError("Context budget leaves no room for cold start")
            retained_base = stratified_context(
                rows,
                base_budget,
                seed=seed,
                stratify_symbolic=stratify_symbolic,
            )
            context = pd.concat(
                [retained_base, retained_additions], ignore_index=True, sort=False
            )
        if context[ACTION_TARGET_COLUMN].nunique() < 2:
            raise ValueError("Actor context must contain at least two actions")
        feature_columns = actor_feature_columns(policy_mode)
        classifier = factory.classifier(
            feature_columns, actor_categorical_columns(policy_mode)
        )
        classifier.fit(context[list(feature_columns)], context[ACTION_TARGET_COLUMN])
        return cls(
            classifier=classifier,
            extractor=DefaultRamFeatureExtractor(),
            feature_columns=feature_columns,
            desired_symbolic_label=factory.config.learning.desired_symbolic_label,
            action_selection=factory.config.learning.action_selection,
            epsilon_sample_probability=(
                factory.config.learning.epsilon_sample_probability
            ),
            rng=np.random.default_rng(seed),
        )

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
        row = self.extractor.state_features(
            ram,
            previous_ram,
            previous_action=previous_action,
            previous_reward=previous_reward,
            lives=lives,
            episode_progress=episode_progress,
            desired_symbolic_label=self.desired_symbolic_label,
        )
        probabilities = self.classifier.predict_proba(
            pd.DataFrame([row])[list(self.feature_columns)]
        )
        classes = np.asarray(self.classifier.classes_, dtype=np.int64)
        valid = (classes >= 0) & (classes < action_count)
        if not valid.any():
            raise ValueError("Actor classifier has no valid Atari actions")
        valid_classes = classes[valid]
        valid_probabilities = np.asarray(probabilities[0, valid], dtype=np.float64)
        probability_sum = float(valid_probabilities.sum())
        if not np.isfinite(probability_sum) or probability_sum <= 0:
            raise ValueError("Actor classifier returned invalid action probabilities")
        valid_probabilities /= probability_sum
        should_sample = self.action_selection == "probability_sample" or (
            self.action_selection == "epsilon_sample"
            and self.rng.random() < self.epsilon_sample_probability
        )
        if should_sample:
            return int(self.rng.choice(valid_classes, p=valid_probabilities))
        if self.action_selection not in ("greedy", "epsilon_sample"):
            raise ValueError(f"Unknown action selection: {self.action_selection}")
        return int(valid_classes[np.argmax(valid_probabilities)])
