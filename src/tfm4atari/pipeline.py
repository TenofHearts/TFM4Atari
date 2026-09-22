"""End-to-end collection, selection, play, and evaluation workflows."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gymnasium.wrappers import RecordVideo

from tfm4atari.config import GameConfig, ProjectConfig
from tfm4atari.context_cache import ContextQueue
from tfm4atari.environment import (
    action_meanings,
    build_env,
    current_lives,
    current_ram,
)
from tfm4atari.features import (
    ACTION_TARGET_COLUMN,
    DESIRED_SYMBOLIC_LABEL,
    OUTCOME_CATEGORY_TARGET,
    OUTCOME_SCORE_TARGET,
    SYMBOLIC_LABEL_SCHEMA,
    DefaultRamFeatureExtractor,
    actor_feature_columns,
)
from tfm4atari.games.beamrider import label_beamrider_interval
from tfm4atari.games.registry import game_module
from tfm4atari.judges import (
    Judgment,
    TrajectoryOutcome,
    judge_state_document,
    label_trials,
)
from tfm4atari.models import (
    Actor,
    TabPFNFactory,
    context_budget,
    resolved_device,
)
from tfm4atari.storage import DataStore, atomic_json
from tfm4atari.teacher import (
    TeacherPolicy,
    fetch_teacher,
    load_teacher,
)


def _versions() -> dict[str, str]:
    names = (
        "tfm4atari",
        "gymnasium",
        "ale-py",
        "stable-baselines3",
        "sb3-contrib",
        "tabpfn",
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
    )
    values: dict[str, str] = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "missing"
    return values


def preflight(config: ProjectConfig) -> dict[str, Any]:
    """Probe hardware, ROMs, action spaces, and local artifacts without downloads."""
    report: dict[str, Any] = {
        "ok": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": resolved_device(config),
        "cuda_available": torch.cuda.is_available(),
        "context_budget": context_budget(config),
        "tabpfn_cache": str(config.path(config.tabpfn.cache_dir)),
        "policy_mode": config.learning.policy_mode,
        "action_selection": config.learning.action_selection,
        "action_feature_count": len(
            actor_feature_columns(config.learning.policy_mode)
        ),
        "versions": _versions(),
        "games": {},
    }
    if len(actor_feature_columns(config.learning.policy_mode)) > 500:
        report["ok"] = False
        report["feature_error"] = "Conservative 500-feature limit exceeded"
    for game in config.active_games:
        game_report: dict[str, Any] = {}
        try:
            env = build_env(game.env_id, config.atari, config.runtime)
            observation, _ = env.reset(seed=config.runtime.seed)
            game_report.update(
                observation_shape=list(observation.shape),
                action_count=int(env.action_space.n),  # type: ignore[attr-defined]
                action_meanings=list(action_meanings(env)),
                ram_size=int(current_ram(env).size),
            )
            env.close()
        except Exception as error:  # noqa: BLE001  # pragma: no cover
            game_report["error"] = f"{type(error).__name__}: {error}"
            report["ok"] = False
        checkpoint = (
            config.path(config.paths.artifact_dir)
            / "teachers"
            / game.name
            / game.teacher_file
        )
        game_report["teacher_checkpoint"] = str(checkpoint)
        game_report["teacher_present"] = checkpoint.is_file()
        game_report["teacher_backend"] = game.teacher_backend
        report["games"][game.name] = game_report
    return report


def fetch_teachers(config: ProjectConfig) -> dict[str, str]:
    return {game.name: str(fetch_teacher(config, game)) for game in config.active_games}


def prepare_tabpfn(config: ProjectConfig) -> dict[str, Any]:
    """Download/license-check TabPFN 3.5 and execute a tiny real inference."""
    feature_columns = ("feature_a", "feature_b")
    x = pd.DataFrame({"feature_a": [0.0, 0.2, 0.8, 1.0], "feature_b": [1, 1, 0, 0]})
    query = pd.DataFrame({"feature_a": [0.1, 0.9], "feature_b": [1, 0]})
    factory = TabPFNFactory(config)
    report: dict[str, Any] = {
        "device": resolved_device(config),
        "cache_dir": str(config.path(config.tabpfn.cache_dir)),
        "policy_mode": config.learning.policy_mode,
    }
    if config.learning.policy_mode == "outcome_prediction_regression":
        regressor = factory.regressor(feature_columns)
        regressor.fit(x, pd.Series([-1.0, -0.5, 0.5, 1.0], name="outcome"))
        report["task"] = "regression"
        report["predictions"] = np.asarray(regressor.predict(query)).tolist()
    else:
        classifier = factory.classifier(feature_columns)
        classifier.fit(x, pd.Series([False, False, True, True], name="success"))
        report["task"] = "classification"
        report["classes"] = [bool(value) for value in classifier.classes_]
        report["probabilities"] = classifier.predict_proba(query).tolist()
    return report


def _load_teacher(config: ProjectConfig, game: GameConfig) -> TeacherPolicy:
    return load_teacher(config, game, device=resolved_device(config))


def _cold_start_context(
    config: ProjectConfig, store: DataStore, game: GameConfig
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    ids = store.cold_start_episode_ids(game.name, game.teacher_backend)
    required = config.collection.episodes_per_game
    if len(ids) != required:
        raise RuntimeError(
            f"{game.name} requires exactly {required} complete cold-start "
            f"teacher trajectories, found {len(ids)}; run collect"
        )
    frames = [store.read_episode(game.name, trajectory_id) for trajectory_id in ids]
    return pd.concat(frames, ignore_index=True), ids


def _episode_progress_feature(config: ProjectConfig, decisions: int) -> float:
    """Use the same time scale for teacher, learning, and capped playback."""
    return decisions / config.features.episode_progress_reference_decisions


def _label_symbolic_interval(
    config: ProjectConfig, game: GameConfig, actions: pd.DataFrame
) -> pd.DataFrame:
    """Label one completed relevant-action interval without teacher values."""
    if game.name == "BeamRider":
        labeled = label_beamrider_interval(actions)
    else:
        labeled = actions.copy()
        label = (
            -1
            if (labeled["lives_after"] < labeled["lives_before"]).any()
            else (1 if (labeled["observed_reward"] > 0).any() else 0)
        )
        labeled["symbolic_label"] = np.int8(label)
    labeled[DESIRED_SYMBOLIC_LABEL] = labeled["symbolic_label"].astype("int8")
    categories, scores, _ = _rolling_outcome_targets(
        actions,
        horizon=len(actions),
        discount=config.learning.outcome_score_discount,
    )
    labeled[OUTCOME_CATEGORY_TARGET] = categories
    labeled[OUTCOME_SCORE_TARGET] = scores
    labeled["label_source"] = "symbolic_action_judge"
    return labeled


def _rolling_outcome_targets(
    actions: pd.DataFrame,
    *,
    horizon: int,
    discount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-action categorical and continuous forward outcomes."""
    if horizon < 1:
        raise ValueError("Outcome horizon must be positive")
    if not 0.0 < discount <= 1.0:
        raise ValueError("Outcome discount must be in (0, 1]")
    if actions.empty:
        return (
            np.asarray([], dtype=np.int8),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int64),
        )
    rewards = actions["observed_reward"].to_numpy(dtype=np.float64) > 0
    deaths = (
        actions["lives_after"].to_numpy(dtype=np.int64)
        < actions["lives_before"].to_numpy(dtype=np.int64)
    )
    signals = np.where(deaths, -1.0, np.where(rewards, 1.0, 0.0))
    sizes = np.minimum(horizon, len(actions) - np.arange(len(actions)))
    categories = np.zeros(len(actions), dtype=np.int8)
    scores = np.zeros(len(actions), dtype=np.float32)
    for start, size in enumerate(sizes):
        stop = start + int(size)
        categories[start] = (
            -1 if deaths[start:stop].any() else int(rewards[start:stop].any())
        )
        weights = np.power(discount, np.arange(size, dtype=np.float64))
        scores[start] = np.float32(
            np.dot(weights, signals[start:stop]) / weights.sum()
        )
    return categories, scores, sizes


def _label_teacher_symbolic_rolling(
    config: ProjectConfig, game: GameConfig, actions: pd.DataFrame
) -> pd.DataFrame:
    """Label each teacher action from its overlapping forward outcome window."""
    if actions.empty:
        return actions.copy()
    capacity = config.context_cache.teacher_judgment_capacity
    labels, scores, sizes = _rolling_outcome_targets(
        actions,
        horizon=capacity,
        discount=config.learning.outcome_score_discount,
    )
    labeled = actions.copy().reset_index(drop=True)
    labeled["symbolic_label"] = labels
    labeled[DESIRED_SYMBOLIC_LABEL] = labels
    labeled[OUTCOME_CATEGORY_TARGET] = labels
    labeled[OUTCOME_SCORE_TARGET] = scores
    labeled["label_source"] = "symbolic_action_judge_rolling"
    labeled["rolling_window_end_step"] = (
        labeled["step"].to_numpy()[np.arange(len(labeled)) + sizes - 1]
    )
    labeled["rolling_window_size"] = sizes
    return labeled


def collect(
    config: ProjectConfig,
    *,
    collection_role: str = "cold_start_teacher",
    require_complete: bool = True,
    target_episodes: int | None = None,
) -> dict[str, int]:
    store = DataStore(config)
    counts: dict[str, int] = {}
    extractor = DefaultRamFeatureExtractor()
    target = target_episodes or config.collection.episodes_per_game
    for game in config.active_games:
        teacher = _load_teacher(config, game)
        matching = [
            trajectory_id
            for trajectory_id in store.episode_ids(game.name)
            if store.episode_metadata(game.name, trajectory_id).get("collection_role")
            == collection_role
            and store.episode_metadata(game.name, trajectory_id)
            .get("teacher", {})
            .get("backend")
            == game.teacher_backend
            and store.episode_metadata(game.name, trajectory_id).get(
                "context_label_schema"
            )
            == SYMBOLIC_LABEL_SCHEMA
        ]
        completed = [
            trajectory_id
            for trajectory_id in matching
            if bool(
                store.episode_metadata(game.name, trajectory_id).get("complete_episode")
            )
        ]
        accepted = completed if require_complete else matching
        attempts = len(matching)
        while (
            len(accepted) < target
            and attempts < config.collection.maximum_attempts_per_game
        ):
            trajectory_id = str(uuid.uuid4())
            seed = config.runtime.seed + attempts
            env = build_env(game.env_id, config.atari, config.runtime)
            observation, _ = env.reset(seed=seed)
            meanings = action_meanings(env)
            action_count = int(env.action_space.n)  # type: ignore[attr-defined]
            previous_ram: np.ndarray | None = None
            previous_action = -1
            previous_reward = 0.0
            episode_return = 0.0
            positive_reward_events = 0
            frames: list[pd.DataFrame] = []
            decisions = 0
            terminated = truncated = False
            initial_lives = current_lives(env)
            while (
                not (terminated or truncated)
                and decisions < config.collection.max_decisions_per_episode
            ):
                ram = current_ram(env)
                teacher_action = teacher.predict(observation)
                if not 0 <= teacher_action < action_count:
                    env.close()
                    raise ValueError(
                        f"{game.name} teacher emitted invalid action {teacher_action}"
                    )
                lives_before = current_lives(env)
                row = extractor.state_features(
                    ram,
                    previous_ram,
                    previous_action=previous_action,
                    previous_reward=previous_reward,
                    lives=lives_before,
                    episode_progress=_episode_progress_feature(config, decisions),
                    desired_symbolic_label=0,
                )
                observation, reward, terminated, truncated, _ = env.step(teacher_action)
                frames.append(
                    {
                        **row,
                        "game": game.name,
                        "trajectory_id": trajectory_id,
                        "step": decisions,
                        ACTION_TARGET_COLUMN: teacher_action,
                        "observed_reward": float(reward),
                        "lives_before": lives_before,
                        "lives_after": current_lives(env),
                    }
                )
                episode_return += float(reward)
                positive_reward_events += int(reward > 0)
                previous_ram = ram
                previous_action = teacher_action
                previous_reward = float(reward)
                decisions += 1
            final_lives = current_lives(env)
            env.close()
            if not frames:
                raise RuntimeError(f"No relevant symbolic rows for {game.name}")
            episode = _label_teacher_symbolic_rolling(
                config, game, pd.DataFrame(frames)
            )
            complete_episode = bool(terminated or truncated)
            store.write_episode(
                game.name,
                trajectory_id,
                episode,
                {
                    "game": game.name,
                    "env_id": game.env_id,
                    "trajectory_id": trajectory_id,
                    "seed": seed,
                    "collection_role": collection_role,
                    "complete_episode": complete_episode,
                    "decision_limit": config.collection.max_decisions_per_episode,
                    "sample_stride": config.collection.sample_stride,
                    "context_label_schema": SYMBOLIC_LABEL_SCHEMA,
                    "episode_return": episode_return,
                    "episode_length": decisions,
                    "initial_lives": initial_lives,
                    "final_lives": final_lives,
                    "positive_reward_events": positive_reward_events,
                    "terminated": terminated,
                    "truncated": truncated,
                    "action_meanings": meanings,
                    "feature_extractor": {
                        "name": extractor.name,
                        "version": extractor.version,
                    },
                    "teacher": teacher.identity,
                    "versions": _versions(),
                    "config": config.model_dump(mode="json", exclude={"root"}),
                },
            )
            matching.append(trajectory_id)
            if complete_episode:
                completed.append(trajectory_id)
            accepted = completed if require_complete else matching
            attempts += 1
        if len(accepted) < target:
            raise RuntimeError(
                f"{game.name} produced {len(accepted)} accepted "
                f"{collection_role} trajectories after {attempts} attempts; "
                "increase collection.maximum_attempts_per_game"
            )
        counts[game.name] = len(accepted)
    return counts


@dataclass(frozen=True)
class EpisodeResult:
    initial_ram: np.ndarray
    episode_return: float
    episode_length: int
    decision_limit: int
    initial_lives: int
    final_lives: int
    positive_reward_events: int
    terminated: bool
    truncated: bool

    def outcome(self, game: str) -> TrajectoryOutcome:
        return TrajectoryOutcome(
            game=game,
            episode_return=self.episode_return,
            decisions=self.episode_length,
            decision_limit=self.decision_limit,
            initial_lives=self.initial_lives,
            final_lives=self.final_lives,
            positive_reward_events=self.positive_reward_events,
            terminated=self.terminated,
            truncated=self.truncated,
        )


def _run_actor(
    config: ProjectConfig,
    game: GameConfig,
    actor: Actor,
    *,
    seed: int,
    initial_callback: Callable[[np.ndarray], None] | None = None,
    factory: TabPFNFactory | None = None,
    base_context: pd.DataFrame | None = None,
    queue: ContextQueue | None = None,
    judge: Any | None = None,
    judge_state: dict[str, Any] | None = None,
    relevance_filter: Any | None = None,
    persist_queue: Callable[[pd.DataFrame], None] | None = None,
    environment: Any | None = None,
    decision_limit: int | None = None,
) -> EpisodeResult:
    env = environment or build_env(game.env_id, config.atari, config.runtime)
    limit = decision_limit or config.collection.max_decisions_per_episode
    actor.rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    del observation
    initial = current_ram(env)
    initial_lives = current_lives(env)
    if initial_callback is not None:
        initial_callback(initial)
    previous_ram: np.ndarray | None = None
    previous_action = -1
    previous_reward = 0.0
    total = 0.0
    decisions = 0
    positive_reward_events = 0
    terminated = truncated = False
    action_count = int(env.action_space.n)  # type: ignore[attr-defined]
    online_trajectory_id = str(uuid.uuid4())
    pending_rows: list[dict[str, Any]] = []
    judged_batches = 0

    adaptive = (
        all(
            value is not None
            for value in (
                factory,
                base_context,
                queue,
                judge,
                judge_state,
                relevance_filter,
            )
        )
        and config.context_cache.enabled
    )

    def flush_context(*, refit_actor: bool) -> None:
        nonlocal actor, judged_batches
        if not adaptive or not pending_rows:
            return
        labeled = _label_symbolic_interval(config, game, pd.DataFrame(pending_rows))
        if not labeled.empty:
            labeled = labeled.assign(
                label_source="symbolic_action_judge",
                judged_window_id=(
                    f"{online_trajectory_id}:{pending_rows[0]['step']}-"
                    f"{pending_rows[-1]['step']}"
                ),
                relevance_filter=relevance_filter.name,
                relevance_filter_version=relevance_filter.version,
            )
            queue.add(labeled)
            judged_batches += 1
            if persist_queue is not None:
                persist_queue(queue.rows)
            refit_due = (
                judged_batches % config.context_cache.refit_every_judged_batches == 0
            )
            if refit_actor and refit_due:
                actor = Actor.fit(
                    factory,
                    base_context,
                    budget=context_budget(config),
                    seed=config.runtime.seed,
                    additions=queue.rows,
                    minimum_base_rows=config.context_cache.minimum_cold_start_rows,
                )
        pending_rows.clear()

    while not (terminated or truncated) and decisions < limit:
        ram = current_ram(env)
        lives_before = current_lives(env)
        action = actor.choose_action(
            ram,
            previous_ram,
            action_count=action_count,
            previous_action=previous_action,
            previous_reward=previous_reward,
            lives=current_lives(env),
            episode_progress=_episode_progress_feature(config, decisions),
        )
        executed_features = actor.extractor.state_features(
            ram,
            previous_ram,
            previous_action=previous_action,
            previous_reward=previous_reward,
            lives=lives_before,
            episode_progress=_episode_progress_feature(config, decisions),
            desired_symbolic_label=0,
        )
        _, reward, terminated, truncated, _ = env.step(action)
        lives_after = current_lives(env)
        total += float(reward)
        positive_reward_events += int(reward > 0)
        executed = {
            **executed_features,
            "game": game.name,
            "online_trajectory_id": online_trajectory_id,
            "step": decisions,
            ACTION_TARGET_COLUMN: action,
            "observed_reward": float(reward),
            "lives_before": lives_before,
            "lives_after": lives_after,
        }
        if adaptive and not relevance_filter.filter(pd.DataFrame([executed])).empty:
            pending_rows.append(executed)
        previous_ram = ram
        previous_action = action
        previous_reward = float(reward)
        decisions += 1
        capacity_full = len(pending_rows) >= config.context_cache.judgment_capacity
        if adaptive and (capacity_full or terminated or truncated):
            has_future_action = not (terminated or truncated) and decisions < limit
            flush_context(refit_actor=has_future_action)
    if adaptive and pending_rows:
        flush_context(refit_actor=False)
    final_lives = current_lives(env)
    env.close()
    return EpisodeResult(
        initial_ram=initial,
        episode_return=total,
        episode_length=decisions,
        decision_limit=limit,
        initial_lives=initial_lives,
        final_lives=final_lives,
        positive_reward_events=positive_reward_events,
        terminated=terminated,
        truncated=truncated,
    )


def bootstrap_judges(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    """Calibrate trajectory judges from the complete teacher cold start."""
    store = DataStore(config)
    judge_documents: dict[str, dict[str, Any]] = {}
    for game in config.active_games:
        judge = game_module(config, game.name).judge
        outcomes = [
            TrajectoryOutcome(
                game=game.name,
                episode_return=float(metadata["episode_return"]),
                decisions=int(metadata["episode_length"]),
                decision_limit=int(metadata["decision_limit"]),
                initial_lives=int(metadata["initial_lives"]),
                final_lives=int(metadata["final_lives"]),
                positive_reward_events=int(metadata["positive_reward_events"]),
                terminated=bool(metadata["terminated"]),
                truncated=bool(metadata["truncated"]),
            )
            for trajectory_id in store.cold_start_episode_ids(
                game.name, game.teacher_backend
            )
            for metadata in [store.episode_metadata(game.name, trajectory_id)]
        ]
        if len(outcomes) != config.collection.episodes_per_game:
            raise RuntimeError(f"Collect the complete cold start for {game.name}")
        state = judge.calibrate(outcomes)
        document = judge_state_document(judge, state)
        store.write_judge_state(game.name, game.teacher_backend, document)
        judge_documents[game.name] = {
            **document,
            "teacher_trajectories": len(outcomes),
        }
    return judge_documents


def _judge_for_game(
    config: ProjectConfig, store: DataStore, game: GameConfig
) -> tuple[Any, dict[str, Any]]:
    judge = game_module(config, game.name).judge
    try:
        document = store.read_judge_state(game.name, game.teacher_backend)
    except FileNotFoundError:
        if judge.name == "return_quantile":
            raise RuntimeError(
                f"No calibrated judge exists for {game.name}; run bootstrap-judges"
            ) from None
        document = judge_state_document(judge, judge.calibrate([]))
        store.write_judge_state(game.name, game.teacher_backend, document)
    if document["judge"] != judge.name or int(document["version"]) != judge.version:
        raise RuntimeError(
            f"Stored judge {document['judge']} v{document['version']} does not "
            f"match configured judge {judge.name} v{judge.version}; rerun "
            "bootstrap-judges"
        )
    return judge, dict(document["state"])


def play(config: ProjectConfig) -> dict[str, int]:
    """Run teacher-free selection and append filtered judged actions to FIFO."""
    store = DataStore(config)
    factory = TabPFNFactory(config)
    counts: dict[str, int] = {}
    for game in config.active_games:
        base_context, cold_start_ids = _cold_start_context(config, store, game)
        existing = len(
            list(
                store.trial_dir(game.name, game.teacher_backend, "learning").glob(
                    "*.parquet"
                )
            )
        )
        judge, judge_state = _judge_for_game(config, store, game)
        module = game_module(config, game.name)
        stored_context = (
            store.read_context_cache(game.name, game.teacher_backend).tail(
                config.context_cache.maximum_rows
            )
            if config.context_cache.enabled
            else pd.DataFrame()
        )
        queue = ContextQueue(
            config.context_cache.maximum_rows,
            stored_context,
        )
        frozen_actor = (
            Actor.fit(
                factory,
                base_context,
                budget=context_budget(config),
                seed=config.runtime.seed,
            )
            if not config.context_cache.enabled
            else None
        )
        for episode_index in range(existing, config.learning.episodes):
            seed = config.runtime.seed + 200_000 + episode_index
            actor = frozen_actor or Actor.fit(
                factory,
                base_context,
                budget=context_budget(config),
                seed=config.runtime.seed,
                additions=queue.rows,
                minimum_base_rows=config.context_cache.minimum_cold_start_rows,
            )
            result = _run_actor(
                config,
                game,
                actor,
                seed=seed,
                factory=factory,
                base_context=base_context,
                queue=queue,
                judge=judge,
                judge_state=judge_state,
                relevance_filter=module.relevance_filter,
                persist_queue=lambda rows,
                game_name=game.name,
                backend=game.teacher_backend: store.write_context_cache(
                    game_name, backend, rows
                ),
            )
            store.write_trial(
                game.name,
                game.teacher_backend,
                f"learning-{episode_index:06d}",
                {
                    "seed": seed,
                    "cold_start_trajectory_ids": json.dumps(cold_start_ids),
                    **result.outcome(game.name).as_trial_columns(),
                },
                phase="learning",
            )
        counts[game.name] = len(
            list(
                store.trial_dir(game.name, game.teacher_backend, "learning").glob(
                    "*.parquet"
                )
            )
        )
    return counts


def _run_simple_policy(
    config: ProjectConfig,
    game: GameConfig,
    policy: Callable[[np.ndarray, int], int],
    *,
    seed: int,
) -> float:
    env = build_env(game.env_id, config.atari, config.runtime)
    observation, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(config.collection.max_decisions_per_episode):
        action_count = int(env.action_space.n)  # type: ignore[attr-defined]
        action = policy(observation, action_count)
        observation, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    env.close()
    return total


def evaluate(config: ProjectConfig) -> dict[str, Any]:
    store = DataStore(config)
    factory = TabPFNFactory(config)
    report: dict[str, Any] = {}
    for game in config.active_games:
        base_context, cold_start_ids = _cold_start_context(config, store, game)
        judge, judge_state = _judge_for_game(config, store, game)
        learning = label_trials(
            store.read_trials(game.name, game.teacher_backend, phases=("learning",)),
            game=game.name,
            judge=judge,
            state=judge_state,
        )
        required = max(config.evaluation.early_window, config.evaluation.late_window)
        if len(learning) < required:
            raise RuntimeError(
                f"{game.name} has {len(learning)} learning trials; "
                f"evaluation requires at least {required}"
            )
        rng = np.random.default_rng(config.runtime.seed + 300_000)
        random_scores = [
            _run_simple_policy(
                config,
                game,
                lambda _observation, count, _rng=rng: int(_rng.integers(0, count)),
                seed=config.runtime.seed + 300_000 + index,
            )
            for index in range(config.evaluation.random_episodes)
        ]
        teacher = _load_teacher(config, game)
        teacher_scores = [
            _run_simple_policy(
                config,
                game,
                lambda observation, _count, _teacher=teacher: _teacher.predict(
                    observation
                ),
                seed=config.runtime.seed + 400_000 + index,
            )
            for index in range(config.evaluation.teacher_episodes)
        ]
        pfn_scores: list[float] = []
        pfn_judgments: list[Judgment] = []
        module = game_module(config, game.name)
        stored_queue = (
            store.read_context_cache(game.name, game.teacher_backend).tail(
                config.context_cache.maximum_rows
            )
            if config.context_cache.enabled
            else pd.DataFrame()
        )
        frozen_actor = (
            Actor.fit(
                factory,
                base_context,
                budget=context_budget(config),
                seed=config.runtime.seed,
            )
            if not config.context_cache.enabled
            else None
        )
        for index in range(config.evaluation.final_episodes):
            seed = config.runtime.seed + 500_000 + index
            queue = ContextQueue(config.context_cache.maximum_rows, stored_queue.copy())
            actor = frozen_actor or Actor.fit(
                factory,
                base_context,
                budget=context_budget(config),
                seed=config.runtime.seed,
                additions=queue.rows,
                minimum_base_rows=config.context_cache.minimum_cold_start_rows,
            )
            result = _run_actor(
                config,
                game,
                actor,
                seed=seed,
                factory=factory,
                base_context=base_context,
                queue=queue,
                judge=judge,
                judge_state=judge_state,
                relevance_filter=module.relevance_filter,
            )
            pfn_scores.append(result.episode_return)
            pfn_judgments.append(judge.judge(result.outcome(game.name), judge_state))
        random_mean = float(np.mean(random_scores))
        teacher_mean = float(np.mean(teacher_scores))
        pfn_mean = float(np.mean(pfn_scores))
        denominator = teacher_mean - random_mean
        normalized = (
            (pfn_mean - random_mean) / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )
        early = float(
            learning["played_episode_return"]
            .head(config.evaluation.early_window)
            .mean()
        )
        late = float(
            learning["played_episode_return"].tail(config.evaluation.late_window).mean()
        )
        early_judge_score = float(
            learning["trajectory_score"].head(config.evaluation.early_window).mean()
        )
        late_judge_score = float(
            learning["trajectory_score"].tail(config.evaluation.late_window).mean()
        )
        report[game.name] = {
            "random_scores": random_scores,
            "teacher_scores": teacher_scores,
            "pfn_scores": pfn_scores,
            "random_mean": random_mean,
            "teacher_mean": teacher_mean,
            "pfn_mean": pfn_mean,
            "random_teacher_normalized_return": normalized,
            "early_learning_mean": early,
            "late_learning_mean": late,
            "judge": judge.name,
            "early_judge_score": early_judge_score,
            "late_judge_score": late_judge_score,
            "pfn_success_rate": float(
                np.mean([judgment.success for judgment in pfn_judgments])
            ),
            "pfn_judgments": [
                judgment.as_trial_columns() for judgment in pfn_judgments
            ],
            "beats_random": pfn_mean > random_mean,
            "improves_over_early": late_judge_score > early_judge_score,
            "cold_start_trajectory_ids": list(cold_start_ids),
        }
    output = config.path(config.paths.data_dir) / (
        f"evaluation_{config.learning.policy_mode}_"
        f"{config.learning.action_selection}.json"
    )
    atomic_json(output, report)
    return report


def record_video(config: ProjectConfig) -> dict[str, Any]:
    """Record one cache-adaptive TabPFN episode without loading the teacher."""
    store = DataStore(config)
    factory = TabPFNFactory(config)
    results: dict[str, Any] = {}
    for game in config.active_games:
        base_context, cold_start_ids = _cold_start_context(config, store, game)
        judge, judge_state = _judge_for_game(config, store, game)
        stored_context = (
            store.read_context_cache(game.name, game.teacher_backend).tail(
                config.context_cache.maximum_rows
            )
            if config.context_cache.enabled
            else pd.DataFrame()
        )
        queue = ContextQueue(config.context_cache.maximum_rows, stored_context)
        actor = Actor.fit(
            factory,
            base_context,
            budget=context_budget(config),
            seed=config.runtime.seed,
            additions=queue.rows,
            minimum_base_rows=config.context_cache.minimum_cold_start_rows,
        )
        module = game_module(config, game.name)
        video_dir = config.path(config.paths.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        prefix = (
            f"{config.video.name_prefix}-{game.name.lower()}-{uuid.uuid4().hex[:8]}"
        )
        before = set(video_dir.glob(f"{prefix}*.mp4"))
        base_env = build_env(game.env_id, config.atari, config.runtime, render=True)
        env = RecordVideo(
            base_env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode: episode == 0,
            video_length=config.video.max_decisions,
            name_prefix=prefix,
            fps=config.video.fps,
        )
        result = _run_actor(
            config,
            game,
            actor,
            seed=config.video.seed,
            factory=factory,
            base_context=base_context,
            queue=queue,
            judge=judge,
            judge_state=judge_state,
            relevance_filter=module.relevance_filter,
            persist_queue=(
                lambda rows, game_name=game.name, backend=game.teacher_backend: (
                    store.write_context_cache(game_name, backend, rows)
                    if config.video.persist_context_additions
                    else None
                )
            ),
            environment=env,
            decision_limit=config.video.max_decisions,
        )
        created = sorted(
            set(video_dir.glob(f"{prefix}*.mp4")) - before,
            key=lambda path: path.stat().st_mtime,
        )
        if not created:
            raise RuntimeError(f"RecordVideo did not produce an MP4 for {game.name}")
        video_path = created[-1]
        judgment = judge.judge(result.outcome(game.name), judge_state)
        results[game.name] = {
            "video": str(video_path),
            "bytes": video_path.stat().st_size,
            "cold_start_trajectory_ids": list(cold_start_ids),
            "episode_return": result.episode_return,
            "decisions": result.episode_length,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "judge": judgment.as_trial_columns(),
            "context_queue_rows": len(queue.rows),
            "context_additions_persisted": (config.video.persist_context_additions),
        }
    return results


def _concatenate_episode_videos(
    episode_videos: list[Path], output: Path
) -> None:
    """Losslessly concatenate finalized, format-identical episode MP4 files."""
    if not episode_videos:
        raise ValueError("At least one episode video is required")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to build a multi-episode video but was not found"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.concat.txt")
    temporary = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.partial.mp4")
    try:
        manifest.write_text(
            "".join(
                f"file '{str(path.resolve()).replace(chr(92), '/')}'\n"
                for path in episode_videos
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-y",
                str(temporary),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"Could not concatenate episode videos: {detail}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced an empty multi-episode video")
        os.replace(temporary, output)
    finally:
        manifest.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def record_learning_video(config: ProjectConfig) -> dict[str, Any]:
    """Run adaptive episodes, report every score, and make one joined video."""
    if not config.context_cache.enabled:
        raise RuntimeError(
            "record-learning-video requires context_cache.enabled = true"
        )
    if not config.video.persist_context_additions:
        raise RuntimeError(
            "record-learning-video requires video.persist_context_additions = true"
        )

    store = DataStore(config)
    factory = TabPFNFactory(config)
    run_id = uuid.uuid4().hex[:12]
    results: dict[str, Any] = {}
    for game in config.active_games:
        base_context, cold_start_ids = _cold_start_context(config, store, game)
        judge, judge_state = _judge_for_game(config, store, game)
        stored_context = store.read_context_cache(
            game.name, game.teacher_backend
        ).tail(config.context_cache.maximum_rows)
        queue = ContextQueue(config.context_cache.maximum_rows, stored_context)
        module = game_module(config, game.name)
        video_dir = config.path(config.paths.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        prefix = (
            f"{config.video.name_prefix}-learning-{game.name.lower()}-{run_id}"
        )
        report_path = config.path(config.paths.data_dir) / (
            f"learning_video_{game.name.lower()}_{store.policy_namespace}_{run_id}.json"
        )
        episode_reports: list[dict[str, Any]] = []
        episode_videos: list[Path] = []

        def write_progress(
            status: str,
            *,
            joined_video: Path | None = None,
            error: str | None = None,
        ) -> None:
            scores = [float(item["episode_return"]) for item in episode_reports]
            document: dict[str, Any] = {
                "status": status,
                "run_id": run_id,
                "game": game.name,
                "policy_mode": config.learning.policy_mode,
                "action_selection": config.learning.action_selection,
                "configured_episodes": config.video.learning_episodes,
                "completed_episodes": len(episode_reports),
                "max_decisions_per_episode": config.video.max_decisions,
                "cold_start_trajectory_ids": list(cold_start_ids),
                "initial_context_rows": len(stored_context),
                "final_context_rows": len(queue.rows),
                "scores": scores,
                "score_mean": float(np.mean(scores)) if scores else None,
                "score_min": float(np.min(scores)) if scores else None,
                "score_max": float(np.max(scores)) if scores else None,
                "first_to_last_score_change": (
                    scores[-1] - scores[0] if len(scores) >= 2 else None
                ),
                "episodes": episode_reports,
                "video": str(joined_video) if joined_video is not None else None,
                "error": error,
            }
            atomic_json(report_path, document)

        write_progress("running")
        joined_video = video_dir / f"{prefix}.mp4"
        try:
            for episode_index in range(config.video.learning_episodes):
                seed = config.video.seed + episode_index
                actor = Actor.fit(
                    factory,
                    base_context,
                    budget=context_budget(config),
                    seed=config.runtime.seed,
                    additions=queue.rows,
                    minimum_base_rows=config.context_cache.minimum_cold_start_rows,
                )
                episode_prefix = f"{prefix}-episode-{episode_index:03d}"
                before = set(video_dir.glob(f"{episode_prefix}*.mp4"))
                base_env = build_env(
                    game.env_id, config.atari, config.runtime, render=True
                )
                env = RecordVideo(
                    base_env,
                    video_folder=str(video_dir),
                    episode_trigger=lambda episode: episode == 0,
                    video_length=config.video.max_decisions,
                    name_prefix=episode_prefix,
                    fps=config.video.fps,
                )
                context_rows_before = len(queue.rows)
                result = _run_actor(
                    config,
                    game,
                    actor,
                    seed=seed,
                    factory=factory,
                    base_context=base_context,
                    queue=queue,
                    judge=judge,
                    judge_state=judge_state,
                    relevance_filter=module.relevance_filter,
                    persist_queue=lambda rows,
                    game_name=game.name,
                    backend=game.teacher_backend: store.write_context_cache(
                        game_name, backend, rows
                    ),
                    environment=env,
                    decision_limit=config.video.max_decisions,
                )
                created = sorted(
                    set(video_dir.glob(f"{episode_prefix}*.mp4")) - before,
                    key=lambda path: path.stat().st_mtime,
                )
                if not created:
                    raise RuntimeError(
                        f"RecordVideo did not produce episode {episode_index} for "
                        f"{game.name}"
                    )
                episode_video = created[-1]
                episode_videos.append(episode_video)
                judgment = judge.judge(result.outcome(game.name), judge_state)
                episode_report = {
                    "episode_index": episode_index,
                    "seed": seed,
                    "episode_return": result.episode_return,
                    "decisions": result.episode_length,
                    "positive_reward_events": result.positive_reward_events,
                    "initial_lives": result.initial_lives,
                    "final_lives": result.final_lives,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "context_rows_before": context_rows_before,
                    "context_rows_after": len(queue.rows),
                    "judge": judgment.as_trial_columns(),
                    "episode_video": str(episode_video),
                }
                episode_reports.append(episode_report)
                store.write_trial(
                    game.name,
                    game.teacher_backend,
                    f"learning-video-{run_id}-{episode_index:06d}",
                    {
                        "seed": seed,
                        "learning_video_run_id": run_id,
                        "learning_video_episode_index": episode_index,
                        "cold_start_trajectory_ids": json.dumps(cold_start_ids),
                        **result.outcome(game.name).as_trial_columns(),
                        **judgment.as_trial_columns(),
                    },
                    phase="learning_video",
                )
                write_progress("running")

            _concatenate_episode_videos(episode_videos, joined_video)
            if not config.video.retain_learning_episode_videos:
                for episode_video in episode_videos:
                    episode_video.unlink()
                for episode_report in episode_reports:
                    episode_report["episode_video"] = None
            write_progress("complete", joined_video=joined_video)
        except BaseException as error:
            write_progress("failed", error=f"{type(error).__name__}: {error}")
            raise
        results[game.name] = {
            "run_id": run_id,
            "video": str(joined_video),
            "report": str(report_path),
            "episode_scores": [
                report["episode_return"] for report in episode_reports
            ],
            "episodes": len(episode_reports),
            "final_context_rows": len(queue.rows),
        }
    return results


def record_teacher_video(config: ProjectConfig) -> dict[str, Any]:
    """Record a complete episode from each configured teacher backend."""
    results: dict[str, Any] = {}
    video_dir = config.path(config.paths.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    for game in config.active_games:
        teacher = _load_teacher(config, game)
        prefix = (
            f"{config.video.teacher_name_prefix}-{game.name.lower()}-"
            f"{game.teacher_backend}-{uuid.uuid4().hex[:8]}"
        )
        before = set(video_dir.glob(f"{prefix}*.mp4"))
        base_env = build_env(game.env_id, config.atari, config.runtime, render=True)
        env = RecordVideo(
            base_env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode: episode == 0,
            video_length=config.video.teacher_max_decisions,
            name_prefix=prefix,
            fps=config.video.fps,
        )
        observation, _ = env.reset(seed=config.video.seed)
        episode_return = 0.0
        decisions = 0
        terminated = truncated = False
        while (
            not (terminated or truncated)
            and decisions < config.video.teacher_max_decisions
        ):
            action = teacher.predict(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            decisions += 1
        env.close()
        created = sorted(
            set(video_dir.glob(f"{prefix}*.mp4")) - before,
            key=lambda path: path.stat().st_mtime,
        )
        if not created:
            raise RuntimeError(
                f"RecordVideo did not produce a teacher MP4 for {game.name}"
            )
        video_path = created[-1]
        results[game.name] = {
            "video": str(video_path),
            "bytes": video_path.stat().st_size,
            "teacher": teacher.identity,
            "episode_return": episode_return,
            "decisions": decisions,
            "terminated": terminated,
            "truncated": truncated,
        }
    return results


def run_pipeline(config: ProjectConfig) -> dict[str, Any]:
    flight = preflight(config)
    if not flight["ok"]:
        raise RuntimeError(f"Preflight failed: {format_result(flight)}")
    return {
        "preflight": flight,
        "teachers": fetch_teachers(config),
        "tabpfn": prepare_tabpfn(config),
        "collected": collect(config),
        "judges": bootstrap_judges(config),
        "played": play(config),
        "evaluation": evaluate(config),
    }


def format_result(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
