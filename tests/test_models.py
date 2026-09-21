import numpy as np
import pandas as pd

from tfm4atari.config import load_config
from tfm4atari.features import (
    ACTION_FEATURE_COLUMNS,
    ACTION_TARGET_COLUMN,
    DESIRED_SYMBOLIC_LABEL,
)
from tfm4atari.models import Actor, stratified_context


class RecordingClassifier:
    classes_ = np.array([False, True])

    def fit(self, x, y):
        self.x = x.copy()
        self.y = y.copy()

    def predict_proba(self, x):
        self.query = x.copy()
        return np.tile([0.5, 0.5], (len(x), 1))


class RecordingFactory:
    def __init__(
        self,
        policy_mode="behavior_cloning",
        action_selection="greedy",
        epsilon_sample_probability=0.30,
    ):
        config = load_config()
        self.config = config.model_copy(
            update={
                "learning": config.learning.model_copy(
                    update={
                        "policy_mode": policy_mode,
                        "action_selection": action_selection,
                        "epsilon_sample_probability": epsilon_sample_probability,
                    }
                )
            }
        )

    def classifier(self, feature_columns, categorical_columns=()):
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        self.model = RecordingClassifier()
        return self.model


def _action_rows(count: int, marker: int) -> pd.DataFrame:
    rows = pd.DataFrame(0, index=range(count), columns=ACTION_FEATURE_COLUMNS)
    rows["step"] = range(count)
    rows[ACTION_TARGET_COLUMN] = [index % 2 for index in range(count)]
    rows[DESIRED_SYMBOLIC_LABEL] = [index % 3 - 1 for index in range(count)]
    rows["ram_000"] = marker
    return rows


def test_actor_context_reserves_rl_additions_and_cold_start() -> None:
    factory = RecordingFactory()
    Actor.fit(
        factory,
        _action_rows(10, marker=1),
        budget=6,
        seed=1,
        additions=_action_rows(3, marker=2),
        minimum_base_rows=4,
    )
    assert len(factory.model.x) == 6
    assert factory.model.x["ram_000"].value_counts().to_dict() == {1: 4, 2: 2}


def test_cold_start_sampling_preserves_both_teacher_trajectories() -> None:
    first = _action_rows(10, marker=1).assign(trajectory_id="first")
    second = _action_rows(10, marker=2).assign(trajectory_id="second")
    context = stratified_context(
        pd.concat([first, second], ignore_index=True), budget=8, seed=1
    )
    assert set(context["trajectory_id"]) == {"first", "second"}


def test_context_sampling_preserves_action_frequency() -> None:
    common = _action_rows(900, marker=1)
    common[ACTION_TARGET_COLUMN] = 0
    rare = _action_rows(100, marker=1)
    rare[ACTION_TARGET_COLUMN] = 1
    rows = pd.concat([common, rare], ignore_index=True).assign(
        trajectory_id="teacher", desired_symbolic_label=1
    )
    context = stratified_context(rows, budget=200, seed=1)
    assert len(context) == 200
    assert context[ACTION_TARGET_COLUMN].value_counts().to_dict() == {0: 180, 1: 20}


def test_behavior_clone_does_not_train_or_query_on_symbolic_label() -> None:
    factory = RecordingFactory()
    actor = Actor.fit(factory, _action_rows(10, marker=1), budget=10, seed=1)
    actor.choose_action(
        np.zeros(128, dtype=np.uint8),
        None,
        action_count=2,
        previous_action=-1,
        previous_reward=0.0,
        lives=3,
        episode_progress=0.0,
    )
    assert DESIRED_SYMBOLIC_LABEL not in factory.model.x
    assert DESIRED_SYMBOLIC_LABEL not in factory.model.query


def test_outcome_conditioned_actor_queries_for_success_label_one() -> None:
    factory = RecordingFactory(policy_mode="outcome_conditioned")
    actor = Actor.fit(factory, _action_rows(10, marker=1), budget=10, seed=1)
    actor.choose_action(
        np.zeros(128, dtype=np.uint8),
        None,
        action_count=2,
        previous_action=-1,
        previous_reward=0.0,
        lives=3,
        episode_progress=0.0,
    )
    assert factory.model.query[DESIRED_SYMBOLIC_LABEL].tolist() == [1]


def test_probability_sampling_is_seeded_and_uses_model_probabilities() -> None:
    factory = RecordingFactory(action_selection="probability_sample")
    actor = Actor.fit(factory, _action_rows(10, marker=1), budget=10, seed=1)
    action = actor.choose_action(
        np.zeros(128, dtype=np.uint8),
        None,
        action_count=2,
        previous_action=-1,
        previous_reward=0.0,
        lives=3,
        episode_progress=0.0,
    )
    assert action == 1


def test_zero_epsilon_uses_greedy_action() -> None:
    factory = RecordingFactory(
        action_selection="epsilon_sample", epsilon_sample_probability=0.0
    )
    actor = Actor.fit(factory, _action_rows(10, marker=1), budget=10, seed=1)
    action = actor.choose_action(
        np.zeros(128, dtype=np.uint8),
        None,
        action_count=2,
        previous_action=-1,
        previous_reward=0.0,
        lives=3,
        episode_progress=0.0,
    )
    assert action == 0
